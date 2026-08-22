"""Red-flag screening for uploaded reports.

Surfaces findings that conventionally warrant prompt clinical review, so a
report containing something urgent is not quietly summarised as routine.

This is NOT triage and NOT a diagnosis. It reports what the document says, and
its output is always phrased as "this needs a clinician's attention", never as
a clinical decision.

Two safeguards against a false alarm, which in this context is a real harm:

1. Every flag the model proposes must carry a verbatim quote from the report,
   and the quote is checked against the source text. A flag whose evidence is
   not literally present is discarded -- so the model cannot invent an
   emergency that the document does not contain.
2. A deterministic keyword pass runs alongside the model, so the most dangerous
   terms are caught even if the model overlooks them.

Absence of a flag is never clearance. The UI must say so.
"""

import json
import re

# Terms that are dangerous enough to surface on sight. Kept deliberately
# specific: "infection" or "mass" alone are too loose and would fire on
# ordinary radiology prose.
KEYWORD_FLAGS = {
    "septic arthritis": "Possible joint infection",
    "joint sepsis": "Possible joint infection",
    "osteomyelitis": "Possible bone infection",
    "abscess": "Possible abscess",
    "cauda equina": "Possible cauda equina syndrome",
    "cord compression": "Possible spinal cord compression",
    "myelopathy": "Possible spinal cord involvement",
    "metastasis": "Possible malignancy",
    "metastases": "Possible malignancy",
    "metastatic": "Possible malignancy",
    "lytic lesion": "Possible malignancy",
    "pathological fracture": "Possible pathological fracture",
    "pathologic fracture": "Possible pathological fracture",
    "acute fracture": "Fracture reported",
    "avascular necrosis": "Possible avascular necrosis",
    "osteonecrosis": "Possible avascular necrosis",
    "deep vein thrombosis": "Possible deep vein thrombosis",
    "septic": "Possible infection",
    "full-thickness tear": "Full-thickness tendon tear",
    "full thickness tear": "Full-thickness tendon tear",
    "complete rupture": "Complete tendon rupture",
}

RED_FLAG_PROMPT = """You are screening a clinical report for findings that
conventionally require prompt medical review.

Look ONLY for these categories:
- joint or bone infection (septic arthritis, osteomyelitis, abscess)
- fracture, especially pathological or insufficiency fracture
- suspected malignancy (metastasis, destructive or lytic lesion, suspicious mass)
- avascular necrosis / osteonecrosis
- spinal cord compression, cauda equina, or a new neurological deficit
- deep vein thrombosis or limb ischaemia
- complete tendon rupture or full-thickness tear
- systemic features: unexplained fever, night sweats, unexplained weight loss

Rules:
- Report ONLY what the report text itself states. Never infer or speculate.
- Ordinary degenerative findings (osteoarthritis, osteophytes, mild
  tendinopathy, bursitis, joint space narrowing, effusion) are NOT red flags.
- For each finding, give a SHORT VERBATIM quote copied exactly from the report.
- If there is nothing in these categories, return an empty list.

Return ONLY valid JSON in exactly this shape, with no other text:
{{"flags": [{{"category": "...", "concern": "...", "quote": "..."}}]}}

REPORT:
{report}
"""


def _normalise(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _quote_is_real(quote, report):
    """True when the model's quoted evidence genuinely appears in the report."""
    quote_norm = _normalise(quote)
    if len(quote_norm) < 8:
        return False
    return quote_norm in _normalise(report)


def keyword_scan(report):
    """Deterministic pass over the report for high-danger terms.

    Where a specific term and a broader one both match -- "septic arthritis"
    and "septic" -- only the specific one is kept, so one problem produces one
    warning rather than a wall of near-duplicates.
    """
    haystack = _normalise(report)
    matched = [term for term in KEYWORD_FLAGS if _normalise(term) in haystack]

    specific = [term for term in matched
                if not any(other is not term and _normalise(term) in _normalise(other)
                           for other in matched)]

    found, seen = [], set()
    for term in specific:
        concern = KEYWORD_FLAGS[term]
        if concern in seen:
            continue
        seen.add(concern)
        found.append({"category": term, "concern": concern,
                      "quote": term, "source": "keyword"})
    return found


def screen_report(report, chat=None):
    """Screen a report for red flags.

    `chat` is an optional callable taking a prompt and returning text -- the
    application passes backend._chat. Without it, only the keyword pass runs.

    Returns a list of {category, concern, quote, source} dicts. Every entry has
    been verified to correspond to text actually present in the report.
    """
    if not (report or "").strip():
        return []

    flags = keyword_scan(report)
    seen = {_normalise(flag["concern"]) for flag in flags}

    if chat is None:
        return flags

    try:
        raw = chat(RED_FLAG_PROMPT.format(report=report[:20000]))
        match = re.search(r"\{.*\}", raw, re.S)
        proposed = json.loads(match.group(0))["flags"] if match else []
    except Exception:
        # Screening is best-effort; the keyword pass still stands.
        return flags

    for item in proposed or []:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote", ""))
        concern = str(item.get("concern", "")).strip()
        # The safeguard: unverifiable evidence means the flag is dropped.
        if not concern or not _quote_is_real(quote, report):
            continue
        if _normalise(concern) in seen:
            continue
        # The keyword pass already reported this problem in its own words.
        if any(_normalise(flag["quote"]) in _normalise(quote) for flag in flags):
            continue
        seen.add(_normalise(concern))
        flags.append({"category": str(item.get("category", "")).strip(),
                      "concern": concern, "quote": quote.strip(),
                      "source": "model"})

    return flags


BANNER_TITLE = "This report contains findings that usually need prompt review"

BANNER_BODY = (
    "Discuss this report with the treating clinician without waiting for a "
    "routine appointment. If the patient is unwell now - fever with a hot, "
    "painful joint, sudden loss of function, new numbness or weakness, or a "
    "hot swollen calf - arrange urgent assessment."
)

NO_FLAG_NOTE = (
    "No red-flag terms were found in this report. That is not clearance: this "
    "screen only checks for a fixed list of terms in the text, and it cannot "
    "see anything the report does not say. Clinical judgement still decides."
)
