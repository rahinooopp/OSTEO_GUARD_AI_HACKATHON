"""OsteoGuard AI - retrieval backend.

Hybrid retrieval (BM25 + PubMedBERT embeddings) fused with RRF and reranked by
a biomedical cross-encoder, then answered by the configured LLM over the
retrieved context.

Also serves `summarize_report`, which condenses a free-text clinical report
and pulls the guideline evidence relevant to it.

Benchmarked on a 50-query clinical eval set:

    configuration                       Precision@5   Confidence
    -------------------------------------------------------------
    previous (ms-marco-MiniLM-L-6)         78.00%       71.03%
    current  (MedCPT + fixes)              87.60%       88.20%
"""

import os
import re

import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import config

# The notebook writes the populated database here.
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "osteoarthritis_db")
COLLECTION_NAME = "osteoarthritis_guidelines_v2"

# Biomedical reranker (trained on PubMed query/article pairs). Replaces the
# generic web-search reranker, which was the single biggest source of error.
RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"

# --- Retrieval presets ------------------------------------------------------
# Calibration constants (A, B) are fitted per preset so the reported confidence
# tracks the precision actually measured for that setting.
#
#   MODE        Precision@5   Confidence   Latency/query (CPU)
#   ---------------------------------------------------------
#   accurate      87.60%        88.20%          ~14 s
#   balanced      84.40%        85.10%          ~5.9 s
#   fast          82.80%        83.50%          ~4.4 s
RETRIEVAL_PRESETS = {
    "accurate": {"fetch_k": 60, "rerank_pool": 40, "A": 1.8037, "B": 0.4720},
    "balanced": {"fetch_k": 50, "rerank_pool": 20, "A": 2.1946, "B": -0.0692},
    "fast": {"fetch_k": 40, "rerank_pool": 15, "A": 2.1389, "B": -0.1519},
}
# "demo" was the former name of the balanced preset; still accepted.
RETRIEVAL_PRESETS["demo"] = RETRIEVAL_PRESETS["balanced"]

MODE = config.RETRIEVAL_MODE if config.RETRIEVAL_MODE in RETRIEVAL_PRESETS else "accurate"

# Only true boilerplate is dropped. The previous list also removed "METHODS",
# "Context" and "Rationale and impact", which hold real recommendation text --
# that is why bisphosphonate / lidocaine / stem-cell queries used to fail.
JUNK_SECTIONS = [
    "contents", "references", "acknowledgment",
    "your responsibility", "update information",
    # Front matter and change logs. These carry the words "guideline" and
    # "management" without any clinical content, so a topic-level query used
    # to rank them above the actual recommendations.
    "minor changes since publication", "terms used in this guideline",
]

# Clinical abbreviations, so the keyword leg can match their expanded forms.
QUERY_EXPANSIONS = {
    "tens": "transcutaneous electrical nerve stimulation",
    "prp": "platelet rich plasma",
    "cmc": "carpometacarpal thumb",
    "pemf": "pulsed electromagnetic field",
    "ia": "intraarticular intra articular",
    "nsaid": "nsaid nsaids nonsteroidal",
    "oa": "oa osteoarthritis",
    "corticosteroid": "corticosteroid glucocorticoid",
    "cane": "cane walking stick assistive device",
    "rollator": "rollator walking frame walking aid assistive device",
    "acetaminophen": "acetaminophen paracetamol",
}

# Reports longer than this are trimmed before being sent to the model.
REPORT_CHAR_LIMIT = 60000

_collection = None
_bm25 = None
_all_docs = None
_all_ids = None
_all_metadatas = None
_id_to_index = None
_cross_encoder = None


def tokenize_clean(text):
    return re.sub(r"\W+", " ", text).lower().split()


def expand_query_tokens(query):
    """Add clinical synonyms so BM25 matches abbreviations to their full forms."""
    tokens = tokenize_clean(query)
    extra = []
    for token in tokens:
        if token in QUERY_EXPANSIONS:
            extra += QUERY_EXPANSIONS[token].split()
    return tokens + extra


def score_to_confidence(score, mode=None):
    """Map a raw cross-encoder score to a calibrated 0-100% confidence."""
    preset = RETRIEVAL_PRESETS[mode or MODE]
    return float(1.0 / (1.0 + np.exp(-(preset["A"] * score + preset["B"]))) * 100.0)


def init_backend():
    global _collection, _bm25, _all_docs, _all_ids, _all_metadatas
    global _id_to_index, _cross_encoder

    if _collection is not None:
        return

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    _collection = client.get_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    data = _collection.get(include=["documents", "metadatas"])
    _all_docs = data["documents"]
    _all_ids = data["ids"]
    _all_metadatas = data["metadatas"]
    _id_to_index = {cid: i for i, cid in enumerate(_all_ids)}  # O(1) lookup

    _bm25 = BM25Okapi([tokenize_clean(doc) for doc in _all_docs])
    _cross_encoder = CrossEncoder(RERANKER_MODEL)


def reciprocal_rank_fusion(semantic_ranks, keyword_ranks, k=60):
    scores = {}
    for rank, chunk_id in enumerate(semantic_ranks):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, chunk_id in enumerate(keyword_ranks):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def reranked_hybrid_search(query, top_n=5, mode=None):
    init_backend()
    preset = RETRIEVAL_PRESETS[mode or MODE]
    fetch_k = preset["fetch_k"]
    rerank_pool = preset["rerank_pool"]

    # Leg A: dense / semantic retrieval
    semantic_ids = _collection.query(query_texts=[query], n_results=fetch_k)["ids"][0]

    # Leg B: sparse / keyword retrieval, with clinical expansion
    keyword_scores = _bm25.get_scores(expand_query_tokens(query))
    keyword_ids = [_all_ids[i] for i in np.argsort(keyword_scores)[::-1][:fetch_k]]

    fused = reciprocal_rank_fusion(semantic_ids, keyword_ids)

    cross_inp = []
    candidates = []
    for chunk_id, _ in fused:
        idx = _id_to_index[chunk_id]
        metadata = _all_metadatas[idx]
        text = _all_docs[idx]

        section = (metadata.get("section_title") or "").lower()
        if any(junk in section for junk in JUNK_SECTIONS):
            continue

        # Context injection: give the reranker the section heading too.
        cross_inp.append([query, f"Section: {metadata.get('section_title', '')}. {text}"])
        candidates.append((chunk_id, metadata, text))

        if len(cross_inp) == rerank_pool:
            break

    if not cross_inp:
        return []

    scores = _cross_encoder.predict(cross_inp)
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return ranked[:top_n]


def retrieve_evidence(query, top_n=5, mode=None):
    """Retrieval only - no LLM call. Returns a list of source dicts."""
    sources = []
    for score, (chunk_id, metadata, text) in reranked_hybrid_search(query, top_n, mode):
        sources.append({
            "doc_name": metadata.get("document_name", "Unknown Document"),
            "page": metadata.get("page_number", "N/A"),
            "section": metadata.get("section_title", ""),
            "url": metadata.get("source_url", "#"),
            "score": float(score),
            "confidence": score_to_confidence(score, mode),
            "text": text,
        })
    return sources


class NotConfigured(RuntimeError):
    """No API key is available from the deployment's configuration."""


_client = None


def _get_client(api_key=None):
    """The generation client for the configured provider.

    `api_key` is only an override for callers that manage their own credentials
    (scripts, notebooks). The application passes nothing: the key comes from the
    environment via config, never from the interface.
    """
    global _client

    key = api_key or config.get_api_key()
    if not key:
        raise NotConfigured(config.MISSING_KEY_MESSAGE)

    if api_key is None and _client is not None:
        return _client

    if config.LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=key)
        client = genai.GenerativeModel(config.GEMINI_MODEL)
    else:
        from groq import Groq
        client = Groq(api_key=key)

    if api_key is None:
        _client = client
    return client


# openai/gpt-oss intermittently falls into a run of filler characters
# (zero-width and narrow no-break spaces). It is not cosmetic: the run eats the
# completion budget and the real answer is truncated mid-sentence. Measured at
# roughly 1 in 6 replies on this corpus at temperature 0.2, so every reply is
# checked and resampled; three attempts puts the residual failure rate well
# under 1%.
_FILLER_RUN = re.compile("[​‌‍  ﻿ ]{8,}")
_DROP = {ord(c): None for c in "​‌‍﻿"}
_TO_SPACE = {ord(c): " " for c in "   "}


def detect_language(text):
    """The language an answer should be written in, decided in Python.

    Asking the model to infer "the same language as the question" made it
    occasionally drift into a third language entirely. The script is something
    we can determine exactly, so we do it here and give the model an explicit
    instruction instead of an inference task.
    """
    arabic = sum(1 for ch in (text or "") if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in (text or "") if ch.isascii() and ch.isalpha())
    return "Arabic" if arabic > latin else "English"


def looks_degenerate(text):
    """True when a reply contains a filler run rather than an answer."""
    return bool(_FILLER_RUN.search(text or ""))


# gpt-oss also splits acronyms with a narrow space -- "NSA<nnbsp>ID" -- which
# a naive space conversion turns into "NSA ID". Inside a run of capitals the
# separator is always an artifact, so it is deleted rather than widened. The
# rule requires capitals on both sides, leaving "5<nnbsp>mg" as "5 mg".
_ACRONYM_SPLIT = re.compile("(?<=[A-Z])[   ](?=[A-Z])")


def _clean_output(text):
    """Remove invisible filler and normalise exotic spaces to plain ones."""
    text = (text or "").translate(_DROP)
    text = _ACRONYM_SPLIT.sub("", text)
    text = text.translate(_TO_SPACE)
    return re.sub(r"[ 	]{3,}", " ", text)


def _chat(prompt, api_key=None, job="assistant", attempts=3):
    """Send one prompt to the model configured for `job` and return its text.

    `job` is "assistant" (guideline answering) or "summary" (report
    summarisation); each carries its own model, temperature and token budget so
    that tuning one cannot regress the other.

    Temperature is held low: this is clinical text handling, not creative
    writing, and on gpt-oss a high temperature makes the filler-run failure far
    more likely. Retrieval never goes through here -- embeddings and reranking
    run locally.
    """
    client = _get_client(api_key)

    if config.LLM_PROVIDER == "gemini":
        return _clean_output(client.generate_content(prompt).text)

    if job == "summary":
        model = config.SUMMARY_MODEL
        temperature = config.SUMMARY_TEMPERATURE
        max_tokens = config.SUMMARY_MAX_TOKENS
    else:
        model = config.ASSISTANT_MODEL
        temperature = config.ASSISTANT_TEMPERATURE
        max_tokens = config.ASSISTANT_MAX_TOKENS

    reply = ""
    for attempt in range(attempts):
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            # Retries change the seed, not the temperature. Raising the
            # temperature also drew the model off-language, so the ladder now
            # varies only the sample.
            temperature=temperature,
            seed=attempt,
            max_completion_tokens=max_tokens,
        )
        # gpt-oss also exposes a `reasoning` field; the answer is `content`.
        reply = completion.choices[0].message.content or ""
        if not looks_degenerate(reply):
            break

    return _clean_output(reply)


def _model_error(exc):
    """A readable message for the interface, not a raw provider traceback."""
    text = str(exc)
    if "rate_limit" in text or "429" in text:
        wait = re.search(r"try again in ([\dhms.]+)", text)
        when = f" Capacity returns in about {wait.group(1).rstrip('.')}." if wait else ""
        return ("The language service has reached its usage limit for now, so "
                f"no new answer can be generated.{when} Retrieval and saved "
                "records are unaffected.")
    if "api key" in text.lower() or "401" in text:
        return ("The language service rejected the configured API key. Check "
                f"{config.key_variable()} in the .env file.")
    return f"Could not reach the {config.LLM_PROVIDER} language service: {exc}"


def generate_response(query, api_key=None, top_n=5, mode=None):
    """Retrieve guideline evidence and answer grounded in it.

    Returns (answer_text, sources, query_in_english). The third value is the
    translated query used for retrieval; the interface uses it so that
    language-specific features work for a question asked in any language.
    """
    # The index is English, so a non-English question is translated first.
    # A question already in English skips that call entirely -- it saves a
    # round trip of latency and, on a metered plan, roughly half the tokens a
    # simple question costs.
    if detect_language(query) == "English":
        translated_query = query
    else:
        translate_prompt = (
            "Translate the following text to English for a clinical database "
            "search. If it is already in English, return it exactly as is. "
            "Output ONLY the translated text." + chr(10) + f"Text: {query}"
        )
        try:
            translated_query = _chat(translate_prompt, api_key).strip()
        except NotConfigured as exc:
            return str(exc), [], query
        except Exception as exc:
            return _model_error(exc), [], query

    sources = retrieve_evidence(translated_query, top_n=top_n, mode=mode)
    if not sources:
        return ("I could not find any relevant information in the guidelines "
                "to answer your query."), [], translated_query

    context_text = "".join(
        f"--- Source: {s['doc_name']} (Page {s['page']}) ---\n{s['text']}\n\n"
        for s in sources
    )

    language = detect_language(query)

    prompt = f"""
You are an expert clinical AI assistant for osteoarthritis management. Use the provided clinical guidelines to answer the user's question.
If the answer is not contained in the provided guidelines, clearly state that you do not have that information based on the guidelines.
Please be concise, objective, and cite the document names where appropriate.

LANGUAGE: Write your entire answer in {language}. Every heading, sentence and bullet must be in {language}. Do not use any other language.

Context from Guidelines:
{context_text}

User Question: {query}
"""

    try:
        return _chat(prompt, api_key), sources, translated_query
    except Exception as exc:
        return _model_error(exc), sources, translated_query


# --- Report summarisation ---------------------------------------------------

ENGLISH_HEADINGS = """### Patient snapshot
### Reason for the report
### Key findings
### Diagnoses stated in the report
### Current management
### Abnormal or urgent findings
### Follow-up and next steps"""

ARABIC_HEADINGS = """### لمحة عن المريض
### سبب التقرير
### أهم النتائج
### التشخيصات المذكورة في التقرير
### العلاج الحالي
### نتائج غير طبيعية أو عاجلة
### المتابعة والخطوات التالية"""

# The summary language is the clinician's choice, independent of the language
# the report happens to be written in.
SUMMARY_LANGUAGES = {
    "auto": {
        "headings": ENGLISH_HEADINGS,
        "not_stated": "Not stated in the report.",
        "rule": ("- Write in the same language the report is written in, and "
                 "translate the headings into that language."),
    },
    "english": {
        "headings": ENGLISH_HEADINGS,
        "not_stated": "Not stated in the report.",
        "rule": ("- Write the summary in English, whatever language the report "
                 "is written in."),
    },
    "arabic": {
        "headings": ARABIC_HEADINGS,
        "not_stated": "غير مذكور في التقرير.",
        "rule": ("- Write the summary in Arabic, whatever language the report is "
                 "written in. Keep drug names, dosages, measurements, dates and "
                 "identifiers exactly as they appear in the report, in their "
                 "original script -- never transliterate or convert them."),
    },
}

SUMMARY_PROMPT = """You are a clinical documentation assistant. Summarise the
medical report below for a busy clinician.

Rules you must follow:
- Use ONLY information that is present in the report. Never infer, estimate,
  extrapolate or invent anything.
- If the report does not contain something a heading asks for, write exactly:
  *{not_stated}*
- Reproduce all numbers, doses, dates and units exactly as written.
- Do not add a diagnosis, grade or measurement the report does not state.
- Do not give treatment advice here; only report what the document says.
{rule}

Return markdown using exactly these headings, in this order:

{headings}

Keep each section to short bullet points.

--- REPORT START ---
{report}
--- REPORT END ---
"""

TOPIC_PROMPT = """Read the clinical report below and list the clinical terms
that should be looked up in a treatment guideline.

Give the affected joint, the diagnosis, and the treatments that are mentioned
or that the findings raise. Clinical terms only, separated by spaces, 15 words
maximum. Always answer in ENGLISH, translating if the report is in another
language -- the guideline index being searched is English.

Do NOT use the words "guideline", "recommendation", "management", "patient" or
"report" -- those words match title pages rather than clinical text. Output the
terms only, nothing else.

{report}
"""

TRUNCATION_NOTE = """

---
*Note: the report was longer than {limit:,} characters, so only the first
{limit:,} characters were summarised.*"""


def summarize_report(report_text, api_key=None, with_evidence=True, top_n=4,
                     mode=None, language="auto"):
    """Summarise a free-text medical report.

    The summary is constrained to the content of the report itself. When
    `with_evidence` is set, guideline passages relevant to the report's topic
    are retrieved as well -- these are clearly separate from the summary and
    must be labelled as guideline text in the UI.

    Returns (summary_markdown, sources).
    """
    text = (report_text or "").strip()
    if not text:
        return "No report text was provided.", []

    truncated = len(text) > REPORT_CHAR_LIMIT
    text = text[:REPORT_CHAR_LIMIT]

    try:
        preset = SUMMARY_LANGUAGES.get(language, SUMMARY_LANGUAGES["auto"])
        prompt = SUMMARY_PROMPT.format(report=text, rule=preset["rule"],
                                       headings=preset["headings"],
                                       not_stated=preset["not_stated"])
        summary = _chat(prompt, api_key, job="summary")
    except NotConfigured as exc:
        return str(exc), []
    except Exception as exc:
        return _model_error(exc), []

    if truncated:
        summary += TRUNCATION_NOTE.format(limit=REPORT_CHAR_LIMIT)

    sources = []
    if with_evidence:
        # Evidence is a bonus -- a failure here must not lose the summary.
        try:
            topic = _chat(TOPIC_PROMPT.format(report=text[:4000]), api_key,
                          job="summary").strip()
            sources = retrieve_evidence(topic, top_n=top_n, mode=mode)
        except Exception:
            sources = []

    return summary, sources


def corpus_stats():
    """Chunk and document counts read straight from the Chroma sqlite file.

    Deliberately does not touch the embedding models, so the UI can show this
    instantly. Returns None if the database cannot be read.
    """
    import sqlite3

    path = os.path.join(DB_PATH, "chroma.sqlite3")
    if not os.path.exists(path):
        return None

    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        documents = con.execute(
            "SELECT string_value, COUNT(*) FROM embedding_metadata "
            "WHERE key = 'document_name' GROUP BY string_value ORDER BY 2 DESC"
        ).fetchall()
        sections = con.execute(
            "SELECT COUNT(DISTINCT string_value) FROM embedding_metadata "
            "WHERE key = 'section_title'"
        ).fetchone()[0]
        pages = con.execute(
            "SELECT MAX(int_value) FROM embedding_metadata WHERE key = 'page_number'"
        ).fetchone()[0]
        con.close()
    except Exception:
        return None

    return {
        "documents": [{"name": name, "chunks": count} for name, count in documents],
        "chunks": sum(count for _, count in documents),
        "sections": sections,
        "max_page": pages,
    }
