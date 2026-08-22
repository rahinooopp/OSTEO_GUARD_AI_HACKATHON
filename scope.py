"""Scope gate for uploaded documents.

OsteoGuard's evidence base is two osteoarthritis guidelines. Summarising a
cardiology report or a utility bill would produce fluent output with nothing
behind it, so a document is classified before any summary is generated:

    in_scope    osteoarthritis or a joint / musculoskeletal report -> summarise
    healthcare  a clinical document about something else           -> refuse
    unrelated   not a medical document at all                      -> refuse

A document that plainly names osteoarthritis is admitted without calling the
language model at all, so the common case costs nothing. The model is only
consulted to separate "other healthcare" from "not healthcare", which keyword
matching alone does badly.
"""

import json
import re

IN_SCOPE = "in_scope"
HEALTHCARE = "healthcare"
UNRELATED = "unrelated"

# Naming any of these is treated as sufficient evidence of osteoarthritis or a
# joint assessment. Written to catch the report vocabulary radiologists and
# rheumatologists actually use, in English and Arabic.
OA_TERMS = [
    "osteoarthritis", "osteo-arthritis", "osteoarthrosis", "arthrosis",
    "degenerative joint disease", "degenerative change", "degenerative disc",
    "joint space narrowing", "joint-space narrowing", "osteophyte",
    "chondromalacia", "kellgren", "cartilage loss", "cartilage thinning",
    "subchondral sclerosis", "subchondral cyst", "meniscal", "meniscus",
    "rotator cuff", "acromio-clavicular", "acromioclavicular",
    "patellofemoral", "tibiofemoral", "glenohumeral", "carpometacarpal",
    "synovitis", "joint effusion", "arthroplasty", "knee replacement",
    "hip replacement", "rheumatology", "rheumatologist", "arthritis",
    # Arabic
    "خشونة",              # roughening - the common Egyptian term for OA
    "الفصال العظمي",   # osteoarthritis
    "التهاب المفاصل",  # arthritis
    "الغضروف",          # cartilage
    "المفصل",            # the joint
    "الركبة",            # the knee
]

# Vocabulary that marks a document as clinical, used only as a fallback when
# the model cannot be reached.
MEDICAL_TERMS = [
    "patient", "diagnosis", "diagnostic", "clinical", "symptom", "treatment",
    "prescription", "prescribed", "dose", "dosage", " mg ", "medication",
    "mri", "ct scan", "radiograph", "x-ray", "ultrasound", "biopsy",
    "physician", "doctor", "clinic", "hospital", "referral", "follow-up",
    "examination", "findings", "impression", "history of", "blood pressure",
    "laboratory", "haemoglobin", "hemoglobin", "creatinine", "therapy",
    "mellitus", "mmhg", "hba1c", "ecg", "echocardiogram", "ejection fraction",
    "vaccination", "vaccine", "surgery", "operative", "nurse", "tablet",
    "screening", "mg daily", "syndrome", "chronic", "acute",
    "المريض", "تشخيص", "علاج", "الطبيب", "مستشفى", "أشعة", "جرعة",
]

CLASSIFY_PROMPT = """Classify the document below into exactly one category.

in_scope    - it concerns osteoarthritis, or is a joint / musculoskeletal
              report (imaging, orthopaedic or rheumatology) where
              osteoarthritis could reasonably be assessed.
healthcare  - it is a medical or health document, but about something else
              entirely (for example cardiology, diabetes, obstetrics,
              dermatology, a pharmacy invoice, a vaccination card).
unrelated   - it is not a medical document at all.

Judge only what the document is about. Do not guess beyond the text.

Return ONLY valid JSON, no other text:
{{"category": "in_scope|healthcare|unrelated", "subject": "<3-6 words naming the document's actual subject>"}}

DOCUMENT:
{excerpt}
"""

# Enough text to identify a document without paying for the whole thing.
EXCERPT_CHARS = 3000


def _normalise(text):
    return re.sub(r"\s+", " ", (text or "").lower())


def mentions_osteoarthritis(text):
    haystack = _normalise(text)
    return any(term in haystack for term in OA_TERMS)


# Two independent clinical terms is enough. The threshold is deliberately low
# because this path only runs when the model is unreachable, and the safer
# error there is to call a document "healthcare" (refused with the clinical
# message) rather than "unrelated".
MEDICAL_TERM_THRESHOLD = 2


def looks_medical(text):
    haystack = _normalise(text)
    return sum(1 for term in MEDICAL_TERMS if term in haystack) >= MEDICAL_TERM_THRESHOLD


def classify_document(text, chat=None):
    """Decide whether a document may be summarised.

    `chat` is an optional callable taking a prompt and returning text (the
    application passes backend._chat). Without it, classification falls back to
    vocabulary matching.

    Returns {"category", "subject", "source"}.
    """
    if not (text or "").strip():
        return {"category": UNRELATED, "subject": "empty document",
                "source": "empty"}

    # Fast path: the document names osteoarthritis or a joint assessment.
    if mentions_osteoarthritis(text):
        return {"category": IN_SCOPE, "subject": "osteoarthritis / joint report",
                "source": "keyword"}

    if chat is not None:
        try:
            raw = chat(CLASSIFY_PROMPT.format(excerpt=text[:EXCERPT_CHARS]))
            match = re.search(r"\{.*\}", raw, re.S)
            if match:
                data = json.loads(match.group(0))
                category = str(data.get("category", "")).strip().lower()
                if category in (IN_SCOPE, HEALTHCARE, UNRELATED):
                    subject = str(data.get("subject", "")).strip() or "unspecified"
                    return {"category": category, "subject": subject[:80],
                            "source": "model"}
        except Exception:
            pass  # fall through to the vocabulary check

    if looks_medical(text):
        return {"category": HEALTHCARE, "subject": "a clinical document",
                "source": "keyword"}
    return {"category": UNRELATED, "subject": "non-medical content",
            "source": "keyword"}


REFUSAL = {
    HEALTHCARE: {
        "title": "This is a clinical document, but not about osteoarthritis",
        "body": ("OsteoGuard is grounded in two osteoarthritis guidelines "
                 "(NICE NG226 and ACR/AF 2019) and holds no evidence on any "
                 "other condition. Summarising it here would produce a "
                 "confident-looking summary with nothing behind it, so no "
                 "summary has been generated. Use a tool built for that "
                 "specialty, or upload an osteoarthritis or joint report."),
    },
    UNRELATED: {
        "title": "This does not look like a medical document",
        "body": ("OsteoGuard summarises clinical reports only. No summary has "
                 "been generated. If this is a medical report that was scanned "
                 "poorly, check the extracted text below and correct it, or "
                 "paste the report text directly."),
    },
}
