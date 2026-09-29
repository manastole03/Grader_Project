"""Prompt templates. Agents return categorical judgements only (quotes, polarity,
intensity); every number is computed in Python by yelp_scoring. Inputs are wrapped
in XML-style tags so every backend (including the offline mock) can locate them."""

ASPECT_DEFINITIONS = {
    "food": "the food and drink itself: taste, freshness, temperature, portion size, menu variety, presentation. "
            "NOT how fast it arrived (service) and NOT its price.",
    "service": "the staff and the service process: friendliness, attentiveness, speed / wait times, order accuracy, "
               "how problems were handled. NOT the room or the food itself.",
    "ambience": "the physical setting: atmosphere, decor, noise level, music, cleanliness of the premises, seating, "
                "comfort, vibe. NOT staff behaviour, NOT waiting times, NOT the food.",
}

INTENSITY_RUBRIC = """intensity:
  1 = mild or hedged ("decent", "a bit slow", "okay but nothing special")
  2 = clear ("good", "friendly", "slow", "cold fries")
  3 = strong or emphatic ("amazing", "best I've had", "disgusting", "never coming back because of the staff")"""

ASPECT_SYSTEM = """You are the {aspect_upper} specialist in a team of agents analysing Yelp reviews.
You ONLY look at {aspect}: {definition}
Other agents cover the other aspects; ignore them. Price and location are not {aspect}.

List every statement in which the reviewer JUDGES the {aspect} of this business.
Include only evaluative statements. Skip plain descriptions of what was ordered, background
story, and wishes for other customers. Read polarity from the reviewer's point of view,
including sarcasm and counterfactuals ("I'd have been better off ordering X" is negative about
what they got; "I hope others have a better experience" is negative).
For each statement give:
  quote: copied EXACTLY from the review (verbatim, at least two words, one sentence or clause)
  polarity: "positive" | "negative" | "neutral"
{rubric}

If the review says nothing about {aspect}, return an empty list.
Return a single JSON object and nothing else:
{{"mentions": [{{"quote": "...", "polarity": "...", "intensity": 1}}]}}"""

# Hand-written few-shot examples (deliberately not drawn from the dataset, to avoid leakage).
FEW_SHOT = {
    "food": [
        ("The tacos were bland and the tortillas stale, but our server was a sweetheart. The salsa was incredible though!",
         '{"mentions": [{"quote": "The tacos were bland and the tortillas stale", "polarity": "negative", "intensity": 2}, '
         '{"quote": "The salsa was incredible though!", "polarity": "positive", "intensity": 3}]}'),
        ("Great patio and friendly staff. Can't wait to come back!", '{"mentions": []}'),
    ],
    "service": [
        ("Food was incredible. We waited 45 minutes for a table and nobody apologized.",
         '{"mentions": [{"quote": "We waited 45 minutes for a table and nobody apologized.", "polarity": "negative", "intensity": 3}]}'),
        ("Cozy little spot, the pho is the best in town.", '{"mentions": []}'),
    ],
    "ambience": [
        ("Super loud inside and the tables were sticky, though the burger was solid.",
         '{"mentions": [{"quote": "Super loud inside and the tables were sticky", "polarity": "negative", "intensity": 2}]}'),
        ("Slowest service ever. The waiter forgot our drinks, but the new patio is gorgeous.",
         '{"mentions": [{"quote": "the new patio is gorgeous", "polarity": "positive", "intensity": 3}]}'),
        ("Our waitress was quick and greeted us with a smile. The curry was perfectly spiced.", '{"mentions": []}'),
    ],
}

ASPECT_USER = """<aspect>{aspect}</aspect>
<review>
{review}
</review>
List the {aspect} mentions as JSON."""

AGGREGATOR_SYSTEM = """You are the lead analyst of a team of agents. Three specialists have extracted what a
Yelp reviewer said about food, service and ambience. Read their findings AND the full review
(which may also discuss price, location, parking, etc.), then judge the reviewer's OVERALL
feeling about the business.

  polarity: "positive" | "negative" | "neutral" (neutral = genuinely mixed or lukewarm, "it was okay")
  intensity (ignored when neutral):
    1 = leaning that way, with real reservations
    2 = clearly positive / clearly negative
    3 = enthusiastic / furious
  other_factors: short phrases for anything outside food/service/ambience that influenced the verdict
  rationale: one or two sentences

Return a single JSON object and nothing else:
{"polarity": "...", "intensity": 2, "other_factors": ["..."], "rationale": "..."}"""

AGGREGATOR_USER = """<review>
{review}
</review>
<aspects_json>
{aspects_json}
</aspects_json>
Give the overall verdict as JSON."""

ARBITER_SYSTEM = """You are the arbiter in a team of review-analysis agents. Several specialist agents
cited the SAME passage of a Yelp review as evidence for their aspect. Decide which aspect(s)
the passage genuinely expresses an opinion about. Definitions:
  food: {food}
  service: {service}
  ambience: {ambience}
A passage can belong to more than one aspect only if it explicitly judges each of them.

Return a single JSON object and nothing else:
{{"aspects": ["<aspect>", ...]}}"""

ARBITER_USER = """<review>
{review}
</review>
<quote>{quote}</quote>
<candidates>{candidates}</candidates>
Which of the candidate aspects does this passage express an opinion about?"""
