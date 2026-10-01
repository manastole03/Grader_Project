"""Draw a uniform, star-stratified random sample directly out of yelp_dataset.tar.

Nothing is extracted to disk. Members are streamed in archive order (business.json
precedes review.json): businesses matching the category/city filter are indexed,
then the *entire* review file is scanned once with reservoir sampling (Algorithm R),
one reservoir per star rating. Every eligible review of a given star rating therefore
has the same probability of being chosen, wherever it sits in the file.

sample_restaurants_from_tar draws whole restaurants instead: a uniform random set of
restaurants, then a uniform random set of reviews from each, so the reviews of one
restaurant can be compared with each other.
"""
from __future__ import annotations

import json
import random
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from yelp_core.schemas import Review

BUSINESS_MEMBER = "yelp_academic_dataset_business.json"
REVIEW_MEMBER = "yelp_academic_dataset_review.json"
_BID_KEY = b'"business_id":"'


@dataclass
class SampleConfig:
    n: int = 100                      # total reviews, split evenly across 1..5 stars
    category: str | None = "Restaurants"
    city: str | None = None
    max_chars: int = 2000             # review length window (characters of review text)
    min_chars: int = 40
    seed: int = 42


@dataclass
class RestaurantSampleConfig:
    businesses: int = 50              # restaurants, uniform over those with enough eligible reviews
    per_business: int = 10            # reviews per restaurant, uniform over its eligible reviews
    category: str | None = "Restaurants"
    city: str | None = None
    max_chars: int = 2000
    min_chars: int = 40
    seed: int = 42
    oversample: int = 3               # candidates drawn per slot, to replace restaurants that fall short


@dataclass
class SampleStats:
    businesses: int = 0
    reviews_scanned: int = 0
    eligible_per_star: dict[int, int] = field(default_factory=dict)


def _lines(tar: tarfile.TarFile, info: tarfile.TarInfo) -> Iterator[bytes]:
    f = tar.extractfile(info)
    assert f is not None
    # Stream-mode members aren't seekable (TextIOWrapper needs that), so iterate bytes.
    for raw in f:
        if raw.strip():
            yield raw


def _business_ok(b: dict, cfg: SampleConfig) -> bool:
    if cfg.category and cfg.category.lower() not in [c.strip().lower() for c in (b.get("categories") or "").split(",")]:
        return False
    if cfg.city and cfg.city.lower() != (b.get("city") or "").strip().lower():
        return False
    return True


def _business_id(raw: bytes) -> str | None:
    """Cheap pre-filter: pull business_id out of the raw line without a full JSON parse."""
    i = raw.find(_BID_KEY)
    if i < 0:
        return None
    j = raw.find(b'"', i + len(_BID_KEY))
    return raw[i + len(_BID_KEY):j].decode()


def sample_from_tar(tar_path: str | Path, cfg: SampleConfig, log=print) -> tuple[list[Review], SampleStats]:
    rng = random.Random(cfg.seed)
    per_star = max(1, cfg.n // 5)
    reservoirs: dict[int, list[tuple[dict, dict]]] = {s: [] for s in range(1, 6)}
    seen = {s: 0 for s in range(1, 6)}
    businesses: dict[str, dict] = {}
    stats = SampleStats()

    with tarfile.open(tar_path, mode="r|*") as tar:  # stream mode: no random seeks
        for info in tar:
            name = Path(info.name).name
            if name == BUSINESS_MEMBER:
                for raw in _lines(tar, info):
                    b = json.loads(raw)
                    if _business_ok(b, cfg):
                        businesses[b["business_id"]] = b
                log(f"indexed {len(businesses):,} businesses matching filter")
            elif name == REVIEW_MEMBER:
                if not businesses:
                    raise RuntimeError("no businesses matched (or review file precedes business file)")
                for raw in _lines(tar, info):
                    stats.reviews_scanned += 1
                    bid = _business_id(raw)
                    if bid not in businesses:
                        continue
                    r = json.loads(raw)
                    stars = int(round(float(r["stars"])))
                    if not 1 <= stars <= 5:
                        continue
                    if not cfg.min_chars <= len((r.get("text") or "").strip()) <= cfg.max_chars:
                        continue
                    seen[stars] += 1
                    res = reservoirs[stars]
                    if len(res) < per_star:
                        res.append((r, businesses[bid]))
                    else:
                        j = rng.randrange(seen[stars])
                        if j < per_star:
                            res[j] = (r, businesses[bid])
                    if stats.reviews_scanned % 1_000_000 == 0:
                        log(f"scanned {stats.reviews_scanned:,} reviews")
                break

    stats.businesses = len(businesses)
    stats.eligible_per_star = dict(seen)
    log(f"scanned {stats.reviews_scanned:,} reviews; eligible per star: {seen}")
    short = {s: len(v) for s, v in reservoirs.items() if len(v) < per_star}
    if short:
        log(f"warning: fewer eligible reviews than requested for stars {short}")
    reviews = [
        Review(review_id=r["review_id"], business_id=r["business_id"], stars=float(s),
               text=r["text"].strip(), business_name=b.get("name", ""), city=b.get("city", ""),
               categories=b.get("categories", ""), date=r.get("date", ""))
        for s in sorted(reservoirs) for r, b in reservoirs[s]
    ]
    rng.shuffle(reviews)
    return reviews, stats


def sample_restaurants_from_tar(tar_path: str | Path, cfg: RestaurantSampleConfig, log=print) -> list[Review]:
    """cfg.businesses restaurants chosen uniformly at random, then cfg.per_business reviews of each.

    Business pass: reservoir-sample oversample x businesses candidates among the businesses that
    match the filter and list at least per_business reviews. Review pass: one reservoir per
    candidate. Candidates are then taken in random order, skipping any with fewer than
    per_business eligible reviews (rejection sampling), so the result is uniform over the
    restaurants that have enough eligible reviews. Reviews are grouped by restaurant, oldest
    first, so a partial run (`run -n`) covers whole restaurants.
    """
    rng = random.Random(cfg.seed)
    k = cfg.per_business
    slots = cfg.businesses * max(1, cfg.oversample)
    candidates: list[dict] = []
    matched = 0
    reservoirs: dict[str, list[dict]] = {}
    seen: dict[str, int] = {}
    scanned = 0

    with tarfile.open(tar_path, mode="r|*") as tar:
        for info in tar:
            name = Path(info.name).name
            if name == BUSINESS_MEMBER:
                for raw in _lines(tar, info):
                    b = json.loads(raw)
                    if not _business_ok(b, cfg) or (b.get("review_count") or 0) < k:
                        continue
                    matched += 1
                    if len(candidates) < slots:
                        candidates.append(b)
                    else:
                        j = rng.randrange(matched)
                        if j < slots:
                            candidates[j] = b
                reservoirs = {b["business_id"]: [] for b in candidates}
                seen = dict.fromkeys(reservoirs, 0)
                log(f"{matched:,} businesses match the filter with >= {k} reviews; drew {len(candidates)} candidates")
            elif name == REVIEW_MEMBER:
                if not reservoirs:
                    raise RuntimeError("no businesses matched (or review file precedes business file)")
                for raw in _lines(tar, info):
                    scanned += 1
                    if scanned % 1_000_000 == 0:
                        log(f"scanned {scanned:,} reviews")
                    bid = _business_id(raw)
                    if bid not in reservoirs:
                        continue
                    r = json.loads(raw)
                    if not 1 <= int(round(float(r["stars"]))) <= 5:
                        continue
                    if not cfg.min_chars <= len((r.get("text") or "").strip()) <= cfg.max_chars:
                        continue
                    seen[bid] += 1
                    res = reservoirs[bid]
                    if len(res) < k:
                        res.append(r)
                    else:
                        j = rng.randrange(seen[bid])
                        if j < k:
                            res[j] = r
                break

    rng.shuffle(candidates)
    chosen = [b for b in candidates if seen[b["business_id"]] >= k][: cfg.businesses]
    log(f"scanned {scanned:,} reviews; {len(chosen)} restaurants with >= {k} eligible reviews")
    if len(chosen) < cfg.businesses:
        log(f"warning: only {len(chosen)} of {cfg.businesses} restaurants have enough eligible reviews")
    return [
        Review(review_id=r["review_id"], business_id=r["business_id"], stars=float(int(round(float(r["stars"])))),
               text=r["text"].strip(), business_name=b.get("name", ""), city=b.get("city", ""),
               categories=b.get("categories", ""), date=r.get("date", ""),
               business_stars=b.get("stars"), business_review_count=b.get("review_count"))
        for b in chosen for r in sorted(reservoirs[b["business_id"]], key=lambda r: r.get("date", ""))
    ]


def write_reviews(reviews: list[Review], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in reviews:
            f.write(json.dumps(r.__dict__) + "\n")


def load_reviews(path: str | Path, limit: int | None = None) -> list[Review]:
    out = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                out.append(Review.from_dict(json.loads(line)))
                if limit and len(out) >= limit:
                    break
    return out
