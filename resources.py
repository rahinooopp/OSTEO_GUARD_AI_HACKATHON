"""Exercise video suggestions.

Both NICE NG226 and the ACR/AF guideline put therapeutic exercise first for
osteoarthritis, so when an answer is about exercise it helps to show the
patient what the movement looks like.

These are YouTube *search* links, not links to particular videos. That is
deliberate: the application cannot watch a video, so it cannot vouch for what
any specific one demonstrates, and a made-up video id is either a dead link or
- worse in a clinical tool - the wrong content presented as a recommendation.
A search link always resolves, and it leaves the clinician choosing the video.
"""

from urllib.parse import quote_plus

# Topic -> (label, search terms). Ordered roughly as the guidelines present
# them: local strengthening, then range of movement, then general activity.
EXERCISE_TOPICS = {
    "knee": [
        ("Quadriceps strengthening for knee OA",
         "quadriceps strengthening exercises knee osteoarthritis physiotherapy"),
        ("Straight leg raises",
         "straight leg raise exercise knee osteoarthritis"),
        ("Sit-to-stand and step-ups",
         "sit to stand step up exercise knee osteoarthritis"),
    ],
    "hip": [
        ("Hip abduction and glute strengthening",
         "hip abduction glute strengthening exercises hip osteoarthritis"),
        ("Hip range of movement",
         "hip range of motion exercises osteoarthritis physiotherapy"),
    ],
    "hand": [
        ("Hand and finger range of movement",
         "hand finger range of motion exercises osteoarthritis"),
        ("Grip strengthening",
         "grip strengthening exercises hand osteoarthritis"),
    ],
    "shoulder": [
        ("Pendulum and range of movement",
         "pendulum exercise shoulder range of motion physiotherapy"),
        ("Rotator cuff and scapular strengthening",
         "rotator cuff scapular strengthening exercises physiotherapy"),
    ],
    "general": [
        ("Low-impact aerobic exercise",
         "low impact aerobic exercise arthritis walking cycling"),
        ("Aquatic exercise",
         "aquatic exercise therapy osteoarthritis pool"),
        ("Tai chi for arthritis",
         "tai chi for arthritis beginners"),
    ],
}

# Arabic terms are listed alongside the English ones so that a question asked
# in Arabic still matches. They are matched as substrings, which handles the
# definite article for free ("الركبة" contains "ركبة"). Only stems specific
# enough to avoid false matches are included -- bare "يد" (hand), for example,
# appears inside many unrelated Arabic words and is deliberately left out.
JOINT_TERMS = {
    "knee": ["knee", "tibiofemoral", "patellofemoral", "quadriceps",
             "ركبة", "ركبه", "الرضفة"],
    "hip": ["hip", "acetabul", "trochanter", "gluteal",
            "الورك", "ورك", "الحوض"],
    "hand": ["hand", "finger", "thumb", "carpometacarpal", "cmc", "wrist",
             "اليد", "أصابع", "إبهام", "الإبهام", "رسغ"],
    "shoulder": ["shoulder", "rotator cuff", "supraspinat", "acromio",
                 "glenohumeral", "biceps tendon",
                 "كتف", "الكتف", "الكفة المدورة"],
}

EXERCISE_TERMS = [
    "exercise", "exercises", "physiotherapy", "physical therapy", "strengthening",
    "stretching", "aerobic", "activity", "rehabilitation", "rehab", "movement",
    "walking", "swimming", "cycling", "tai chi", "yoga", "muscle",
    # Arabic
    "تمارين", "تمرين", "رياض", "علاج طبيعي", "العلاج الطبيعي",
    "تأهيل", "إطالة", "تقوية", "مشي", "سباحة", "حركة", "نشاط", "يوجا", "عضل",
]


def is_exercise_related(*texts):
    """True when the question or answer is about exercise or physiotherapy."""
    haystack = " ".join(t or "" for t in texts).lower()
    return any(term in haystack for term in EXERCISE_TERMS)


def detect_joints(*texts):
    """Which joints the text is about, for picking relevant videos."""
    haystack = " ".join(t or "" for t in texts).lower()
    found = [joint for joint, terms in JOINT_TERMS.items()
             if any(term in haystack for term in terms)]
    return found


def youtube_search_url(terms):
    return "https://www.youtube.com/results?search_query=" + quote_plus(terms)


def exercise_videos(*texts, limit=5):
    """Suggested exercise searches for the joints mentioned in the text.

    Returns [] when the text is not about exercise, so the caller can simply
    hide the panel.
    """
    if not is_exercise_related(*texts):
        return []

    joints = detect_joints(*texts)
    topics = []
    for joint in joints:
        topics.extend(EXERCISE_TOPICS.get(joint, []))
    # General activity advice applies whatever the joint, and is the fallback
    # when no specific joint was named.
    topics.extend(EXERCISE_TOPICS["general"])

    seen = set()
    videos = []
    for label, terms in topics:
        if label in seen:
            continue
        seen.add(label)
        videos.append({"label": label, "terms": terms,
                       "url": youtube_search_url(terms)})
        if len(videos) == limit:
            break
    return videos


DISCLAIMER = (
    "These open a YouTube search, not a specific video - the app cannot vet "
    "individual videos. Review a video before giving it to a patient, and stop "
    "any exercise that causes new or worsening pain."
)
