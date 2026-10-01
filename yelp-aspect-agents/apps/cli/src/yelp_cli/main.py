"""yelp-agents CLI.

  yelp-agents providers                     show which LLM backend would be used
  yelp-agents sample --tar ... -n 100       stream a stratified sample out of the Yelp tar
  yelp-agents sample-restaurants -b 50 -k 10  50 random restaurants x 10 reviews each
  yelp-agents run    -n 20                  run the agent team over the sample
  yelp-agents eval                          metrics vs stars + business roll-up
  yelp-agents compare                       reviews of the same restaurant, side by side
  yelp-agents viz                           interactive 3-D (food, service, ambience) plot

Web UI: `uv run yelp-web`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yelp_core import Settings, build_llm

DEFAULT_TAR = Path.home() / "Downloads" / "Yelp JSON" / "yelp_dataset.tar"
DEFAULT_SAMPLE = Path("data/sample_reviews.jsonl")
DEFAULT_RESULTS = Path("outputs/results.jsonl")
RESTAURANT_SAMPLE = Path("data/restaurant_reviews.jsonl")
RESTAURANT_RESULTS = Path("outputs/restaurant_results.jsonl")


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    if getattr(args, "provider", None):
        s.provider = args.provider
    if getattr(args, "model", None):
        s.model = args.model
    return s


def cmd_providers(args: argparse.Namespace) -> None:
    s = _settings(args)
    print(f"ASU key:        {'set' if s.asu_api_key else 'not set'} (ASU_API_KEY or API_KEY; {s.asu_base_url})")
    print(f"NVIDIA_API_KEY: {'set' if s.nvidia_api_key else 'not set'}")
    print(f"HF_TOKEN:       {'set' if s.hf_token else 'not set'}")
    from yelp_core.llm import OllamaClient
    print(f"Ollama:         {'reachable' if OllamaClient.reachable(s.ollama_base_url) else 'not reachable'} ({s.ollama_base_url})")
    print(f"-> LLM_PROVIDER={s.provider} resolves to {build_llm(s)!r}")


def cmd_sample(args: argparse.Namespace) -> None:
    from yelp_data import SampleConfig, sample_from_tar, write_reviews
    cfg = SampleConfig(n=args.n, category=args.category or None, city=args.city, seed=args.seed)
    reviews, _ = sample_from_tar(args.tar, cfg, log=lambda m: print(f"  {m}", file=sys.stderr))
    write_reviews(reviews, args.out)
    print(f"wrote {len(reviews)} reviews -> {args.out}")


def cmd_sample_restaurants(args: argparse.Namespace) -> None:
    from yelp_data import RestaurantSampleConfig, sample_restaurants_from_tar, write_reviews
    cfg = RestaurantSampleConfig(businesses=args.businesses, per_business=args.per_business,
                                 category=args.category or None, city=args.city, seed=args.seed)
    reviews = sample_restaurants_from_tar(args.tar, cfg, log=lambda m: print(f"  {m}", file=sys.stderr))
    write_reviews(reviews, args.out)
    print(f"wrote {len(reviews)} reviews of {len({r.business_id for r in reviews})} restaurants -> {args.out}")
    print(f"next: yelp-agents run --sample {args.out} --out {RESTAURANT_RESULTS}")


def cmd_run(args: argparse.Namespace) -> None:
    from yelp_agents import Orchestrator
    from yelp_data import load_reviews
    llm = build_llm(_settings(args))
    agg_llm = None
    if args.aggregator_model:
        s = _settings(args)
        s.model = args.aggregator_model
        agg_llm = build_llm(s)
    reviews = load_reviews(args.sample, limit=args.n)
    print(f"agents: food, service, ambience -> aggregator | backend {llm!r}"
          + (f" | aggregator {agg_llm!r}" if agg_llm else "") + f" | {len(reviews)} reviews", file=sys.stderr)
    orch = Orchestrator(llm, aggregator_llm=agg_llm, few_shot=not args.zero_shot)

    def progress(i: int, n: int, a) -> None:
        v = " ".join(f"{x:+.2f}" for x in a.vector)
        print(f"  [{i}/{n}] {a.review.stars:.0f}★ -> {a.overall.sentiment:8} "
              f"rule {a.overall.stars_rule:.2f}★  [{v}]  {a.latency_s:.1f}s", file=sys.stderr)

    results = orch.run(reviews, args.out, workers=args.workers, on_result=progress, retry_errors=args.retry_errors)
    print(f"{len(results)} analyses in {args.out}")


def cmd_eval(args: argparse.Namespace) -> None:
    from yelp_eval import business_rollup, evaluate, format_report, load_results
    results = load_results(args.results)
    m = evaluate(results)
    print(format_report(m))
    rollup = business_rollup(results, min_reviews=args.min_reviews)
    if len(rollup):
        print(f"\nBusinesses (>= {args.min_reviews} reviews), mean [food, service, ambience]:")
        for row in rollup.head(args.top).to_dict("records"):
            vec = ", ".join("  n/a" if row[a] is None or row[a] != row[a] else f"{row[a]:+.2f}"
                            for a in ("food", "service", "ambience"))
            print(f"  {row['business'][:32]:32} {row['city'][:14]:14} n={row['reviews']:<3} "
                  f"stars {row['mean_stars']:.1f}  [{vec}]")
    if args.json:
        Path(args.json).write_text(json.dumps(m, indent=2, default=float))
        print(f"\nmetrics json -> {args.json}")


def cmd_compare(args: argparse.Namespace) -> None:
    from yelp_eval import load_results
    from yelp_eval.compare import compare_restaurants, format_comparison, format_restaurant
    results = load_results(args.results)
    c = compare_restaurants(results, min_reviews=args.min_reviews)
    if args.restaurant:
        q = args.restaurant.lower()
        hits = [x for x in c["restaurants"] if q in x["business"].lower()]
        if not hits:
            sys.exit(f"no restaurant matching {args.restaurant!r}")
        print("\n\n".join(format_restaurant(x, results) for x in hits))
    else:
        print(format_comparison(c, top=args.top))
    if args.json:
        Path(args.json).write_text(json.dumps(c, indent=2, default=float))
        print(f"\ncomparison json -> {args.json}")


def cmd_viz(args: argparse.Namespace) -> None:
    from yelp_eval import load_results, results_frame
    from yelp_eval.figures import aspect_space_3d
    fig = aspect_space_3d(results_frame(load_results(args.results)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"3-D plot -> {Path(args.out).resolve()}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="yelp-agents", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def llm_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--provider", choices=["auto", "asu", "nvidia", "hf", "ollama", "mock"])
        sp.add_argument("--model", help="override model id for the chosen provider")

    sp = sub.add_parser("providers", help="show backend resolution")
    llm_flags(sp)
    sp.set_defaults(fn=cmd_providers)

    sp = sub.add_parser("sample", help="sample reviews from the Yelp tar")
    sp.add_argument("--tar", type=Path, default=DEFAULT_TAR)
    sp.add_argument("-n", type=int, default=100)
    sp.add_argument("--category", default="Restaurants", help="'' to disable")
    sp.add_argument("--city")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out", type=Path, default=DEFAULT_SAMPLE)
    sp.set_defaults(fn=cmd_sample)

    sp = sub.add_parser("sample-restaurants", help="random restaurants, several reviews of each")
    sp.add_argument("--tar", type=Path, default=DEFAULT_TAR)
    sp.add_argument("-b", "--businesses", type=int, default=50, help="restaurants to sample")
    sp.add_argument("-k", "--per-business", type=int, default=10, help="reviews per restaurant")
    sp.add_argument("--category", default="Restaurants", help="'' to disable")
    sp.add_argument("--city")
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out", type=Path, default=RESTAURANT_SAMPLE)
    sp.set_defaults(fn=cmd_sample_restaurants)

    sp = sub.add_parser("run", help="run the agent team")
    llm_flags(sp)
    sp.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    sp.add_argument("-n", type=int, help="only the first N reviews of the sample")
    sp.add_argument("--workers", type=int, default=2, help="reviews processed concurrently")
    sp.add_argument("--zero-shot", action="store_true", help="drop the few-shot examples")
    sp.add_argument("--aggregator-model", help="different model for the lead agent")
    sp.add_argument("--retry-errors", action="store_true", help="re-run reviews whose stored analysis had an agent error")
    sp.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("eval", help="metrics vs star ratings")
    sp.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    sp.add_argument("--min-reviews", type=int, default=2)
    sp.add_argument("--top", type=int, default=15)
    sp.add_argument("--json", type=Path, help="also write metrics to this file")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("compare", help="compare the reviews of each restaurant")
    sp.add_argument("--results", type=Path, default=RESTAURANT_RESULTS)
    sp.add_argument("--min-reviews", type=int, default=2)
    sp.add_argument("--top", type=int, help="only the first N restaurants (most contested first)")
    sp.add_argument("--restaurant", help="show one restaurant's reviews side by side (name substring)")
    sp.add_argument("--json", type=Path, help="also write the comparison to this file")
    sp.set_defaults(fn=cmd_compare)

    sp = sub.add_parser("viz", help="3-D scatter HTML")
    sp.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    sp.add_argument("--out", type=Path, default=Path("outputs/aspect_space.html"))
    sp.set_defaults(fn=cmd_viz)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
