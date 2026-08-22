"""OsteoGuard AI - osteoarthritis clinical decision support.

Two working tools, both driven by the same guideline-grounded LLM backend:

    1. Clinical Assistant - ask a question, get an answer grounded in retrieved
       NICE NG226 / ACR-AF 2019 guideline text, with citations.
    2. Report Summary    - upload or paste a medical report, get a structured
       summary plus the guideline evidence relevant to it.

Run with:  python -m streamlit run app.py
"""
#python -m streamlit run "C:\Users\joeel\OneDrive\Desktop\OSTEO_GUARD_AI_HACKATHON\app.py"

import json
import os
from contextlib import contextmanager

import streamlit as st

import config
import records
import reports
import resources
import risk
import safety
import scope

st.set_page_config(page_title="OsteoGuard AI", page_icon="\U0001FA7A", layout="wide",
                   initial_sidebar_state="expanded")

THEME_PATH = os.path.join(os.path.dirname(__file__), "theme.css")


@st.cache_data
def load_theme(mtime):
    """Read theme.css, keyed on the file's mtime so editing the stylesheet
    shows up on the next rerun instead of serving a stale cached copy.

    The parameter must NOT start with an underscore: st.cache_data excludes
    underscore-prefixed arguments from the cache key, which would cache the
    first read forever.
    """
    with open(THEME_PATH, encoding="utf-8") as handle:
        return handle.read()


st.markdown("<style>" + load_theme(os.path.getmtime(THEME_PATH)) + "</style>",
            unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Small view helpers
# --------------------------------------------------------------------------
@contextmanager
def card(title=None, icon=""):
    """A bordered container with an optional icon heading."""
    box = st.container(border=True)
    with box:
        if title:
            st.markdown(
                f"<div class='card-h'><span class='ico'>{icon}</span>"
                f"<span class='t'>{title}</span></div>",
                unsafe_allow_html=True,
            )
        yield box


def is_rtl(text):
    """True when a block of text is predominantly right-to-left.

    Counts Arabic letters against Latin ones rather than looking only at the
    first character, so a summary that opens with a Latin drug name still reads
    as Arabic.
    """
    text = text or ""
    arabic = sum(1 for ch in text if "؀" <= ch <= "ۿ")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return arabic > 20 and arabic > latin * 0.3


def render_markdown(text):
    """Render markdown, flipping the whole block to RTL when it is Arabic."""
    if is_rtl(text):
        st.markdown("<span class='rtl-mark'></span>", unsafe_allow_html=True)
    st.markdown(text)


def row(key, value_html):
    st.markdown(
        f"<div class='row'><span class='k'>{key}</span>"
        f"<span class='v'>{value_html}</span></div>",
        unsafe_allow_html=True,
    )


def bar(percent, start="#38bdf8", end="#10b981"):
    st.markdown(
        f"<div class='bar-bg'><div class='bar-fill' style='width:{percent:.0f}%;"
        f"background:linear-gradient(90deg,{start},{end});'></div></div>",
        unsafe_allow_html=True,
    )


def empty_state(message):
    st.markdown(f"<div class='empty'>{message}</div>", unsafe_allow_html=True)


def render_evidence(sources, caption=None):
    """Guideline passages, shown as expandable cited cards."""
    if not sources:
        return
    if caption:
        st.markdown(f"<div class='note-blue'>{caption}</div>", unsafe_allow_html=True)
        st.write("")
    for index, source in enumerate(sources, 1):
        header = (f"[{index}]  {source['doc_name']}  ·  p.{source['page']}"
                  f"  ·  confidence {source['confidence']:.0f}%")
        with st.expander(header):
            if source.get("section"):
                st.markdown(
                    f"<span class='pill pill-blue'>{source['section']}</span>",
                    unsafe_allow_html=True,
                )
                st.write("")
            bar(source["confidence"])
            st.write("")
            st.write(source["text"])
            st.markdown(f"[Open the source guideline]({source['url']})")


def render_exercise_videos(*texts):
    """Show exercise demonstrations when the text is about exercise.

    Both guidelines put therapeutic exercise first, so an answer that
    recommends it is more useful with something to show the patient.
    """
    videos = resources.exercise_videos(*texts)
    if not videos:
        return
    with card("Exercise demonstrations", "🎬"):
        video_links(videos)


def video_links(videos):
    st.markdown("".join(
        f"<a class='vid' href='{video['url']}' target='_blank' "
        f"rel='noopener noreferrer'><span class='play'>&#9654;</span>"
        f"<span>{video['label']}</span></a>" for video in videos),
        unsafe_allow_html=True)
    st.caption(resources.DISCLAIMER)


BAR_COLOR = {"High": "#ef4444", "Moderate": "#f59e0b", "Low": "#10b981"}
RISK_PILL = {"High": "pill-red", "Moderate": "pill-amber", "Low": "pill-green"}


# --------------------------------------------------------------------------
# Navigation. The clinical LLM assistant is the first entry and the landing
# page - it is the product's primary tool.
# --------------------------------------------------------------------------
NAV = [
    ("Clinical Assistant", "\U0001F9E0"),
    ("Report Summary", "\U0001F4DD"),
    ("Patient Records", "\U0001F464"),
    ("Statistics", "\U0001F4CA"),
    ("About", "ℹ️"),
]

HEADERS = {
    "Clinical Assistant": (
        "Clinical Assistant",
        "Ask any osteoarthritis management question. Every answer is generated "
        "only from guideline passages retrieved for that question, and each "
        "passage is shown with its source and page.",
    ),
    "Report Summary": (
        "Medical Report Summary",
        "Upload or paste a clinical report. The assistant condenses it into a "
        "structured summary and pulls the guideline evidence that matches it.",
    ),
    "Patient Records": (
        "Patient Records",
        "Saved assessments and patient history.",
    ),
    "Statistics": (
        "Retrieval Statistics",
        "How the evidence engine performs, measured on a 50-query clinical "
        "evaluation set.",
    ),
    "About": (
        "About OsteoGuard AI",
        "What this system does, what it is grounded in, and where its limits are.",
    ),
}

DEFAULTS = {
    "page": "Clinical Assistant",
    "question": "",
    "question_en": "",
    "report_input": "",
    "report_note": "",
    "report_ocr": False,
    "out_of_scope": None,
    "red_flags": [],
    "screened": False,
    "saved_id": None,
    "answer": "",
    "sources": [],
    "summary": "",
    "summary_sources": [],
    "recommendations": "",
    "recommendation_sources": [],
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

# Deep-linking: /?page=Report+Summary opens that page directly. Applied once
# per browser session so in-app navigation is not overridden on rerun.
_page_param = st.query_params.get("page")
if _page_param in dict(HEADERS) and not st.session_state.get("_deeplinked"):
    st.session_state.page = _page_param
    st.session_state._deeplinked = True


def use_example(text):
    """Callback: fill the question box. Runs before the rerun, which is the
    only point at which a widget-backed key may be written."""
    st.session_state.question = text


def load_uploaded_report():
    """Callback: pull text out of the uploaded file into the paste box.

    Scans go through OCR inside extract_text, which can take a few seconds per
    page; Streamlit shows the file uploader's own spinner meanwhile.
    """
    upload = st.session_state.get("report_file")
    if upload is None:
        st.session_state.report_note = ""
        st.session_state.report_ocr = False
        return
    text, note, used_ocr = reports.extract_text(upload.name, upload.getvalue())
    st.session_state.report_note = note
    st.session_state.report_ocr = used_ocr
    if text:
        st.session_state.report_input = text


def clear_report():
    st.session_state.report_input = ""
    st.session_state.report_note = ""
    st.session_state.report_ocr = False
    st.session_state.summary = ""
    st.session_state.summary_sources = []
    st.session_state.red_flags = []
    st.session_state.screened = False
    st.session_state.saved_id = None
    st.session_state.out_of_scope = None


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='brand'><div class='brand-logo'>\U0001FA7A</div>"
        "<div><div class='brand-name'>OsteoGuard AI</div>"
        "<div class='brand-tag'>Smarter insights<br>for healthier joints</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    for name, icon in NAV:
        active = st.session_state.page == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}",
                     type="primary" if active else "secondary",
                     use_container_width=True):
            st.session_state.page = name
            st.rerun()

    st.divider()

    # Credentials and engine settings come from deployment configuration
    # (see config.py), never from this interface.
    configured = config.api_key_configured()

    status = ("<span class='dot'></span> Evidence engine online" if configured
              else "<span class='dot' style='background:#f59e0b'></span> "
                   "Evidence engine offline")
    st.markdown(f"<div class='chip' style='margin:4px 0 16px'>{status}</div>",
                unsafe_allow_html=True)

    st.markdown(
        "<div class='side-quote'>&ldquo;Better joints for a more active "
        "tomorrow&rdquo;</div>",
        unsafe_allow_html=True,
    )

page = st.session_state.page
title, subtitle = HEADERS[page]

st.markdown(
    f"<div class='hero'><div class='h'>{title}</div>"
    f"<div class='s'>{subtitle}</div>"
    f"<div class='chips'>"
    f"<span class='chip'><span class='dot'></span>NICE NG226 &middot; ACR/AF 2019</span>"
    f"<span class='chip'>\U0001F50E Hybrid clinical retrieval</span>"
    f"<span class='chip'>\U0001F4CE Every claim cited to page</span>"
    f"</div></div>",
    unsafe_allow_html=True,
)

if not configured:
    st.markdown(f"<div class='note-amber'>{config.MISSING_KEY_MESSAGE}</div>",
                unsafe_allow_html=True)
    st.write("")


# ==========================================================================
# 1. CLINICAL ASSISTANT  (the LLM - first in the nav, landing page)
# ==========================================================================
if page == "Clinical Assistant":
    left, right = st.columns([1.75, 1])

    with left:
        with card("Ask a clinical question", "\U0001F9E0"):
            st.text_area(
                "Question", key="question", height=120, label_visibility="collapsed",
                placeholder="Example: What is the clinical role of duloxetine in "
                            "managing knee osteoarthritis?",
            )

            st.caption("Or start from an example:")
            examples = [
                "Are topical NSAIDs recommended before oral NSAIDs?",
                "What type of exercise is recommended for knee osteoarthritis?",
                "When should a patient be referred for joint replacement surgery?",
            ]
            for column, example in zip(st.columns(3), examples):
                column.button(example, key=f"ex_{example[:18]}",
                              on_click=use_example, args=(example,),
                              use_container_width=True)

            st.write("")
            ask = st.button("⚡  Retrieve evidence & answer", type="primary",
                            use_container_width=True)

        if ask:
            question = st.session_state.question.strip()
            if not question:
                st.warning("Enter a question first.")
            elif not configured:
                st.error(config.MISSING_KEY_MESSAGE)
            else:
                with st.spinner("Searching the guidelines and composing an answer..."):
                    try:
                        from backend import generate_response
                        answer, sources, question_en = generate_response(
                            question, top_n=5)
                        st.session_state.answer = answer
                        st.session_state.sources = sources
                        st.session_state.question_en = question_en
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        if st.session_state.answer:
            with card("Answer", "\U0001F4AC"):
                render_markdown(st.session_state.answer)
            if st.session_state.sources:
                with card("Retrieved evidence", "\U0001F4DA"):
                    render_evidence(st.session_state.sources)
            # Detection runs on the English translation as well as the
            # original, so an Arabic question still matches.
            render_exercise_videos(st.session_state.question_en,
                                   st.session_state.question,
                                   st.session_state.answer)

    with right:
        with card("How an answer is built", "\U0001F9EA"):
            steps = [
                "Your question is translated to English for the search index.",
                "Keyword (BM25) and clinical-embedding search run in parallel.",
                "Both result lists are fused with reciprocal rank fusion.",
                "A biomedical cross-encoder reranks what survived.",
                "The model answers using only the top passages, in your language.",
            ]
            st.markdown(
                "<div class='checklist'>" + "".join(
                    f"<div class='check' style='animation-delay:{i * 60}ms'>"
                    f"<span class='tick'>{i + 1}</span><span>{step}</span></div>"
                    for i, step in enumerate(steps)) + "</div>",
                unsafe_allow_html=True,
            )

        with card("Guideline coverage", "\U0001F4D6"):
            try:
                from backend import corpus_stats
                stats = corpus_stats()
            except Exception:
                stats = None

            if stats:
                for document in stats["documents"]:
                    share = document["chunks"] / max(stats["chunks"], 1) * 100
                    row(document["name"], f"{document['chunks']} passages")
                    bar(share)
                    st.write("")
                st.caption(f"{stats['chunks']} indexed passages across "
                           f"{stats['sections']} guideline sections.")
            else:
                st.caption("Guideline index not readable from disk.")

        with card("Safe use", "\U0001F6E1️"):
            st.markdown(
                "<div class='note-blue'>OsteoGuard AI supports clinical "
                "decision-making. It does not diagnose, and it does not replace "
                "professional judgement. Always confirm against the current "
                "published guideline before acting.</div>",
                unsafe_allow_html=True,
            )


# ==========================================================================
# 2. REPORT SUMMARY  (replaces the old X-ray module)
# ==========================================================================
elif page == "Report Summary":
    # Filled in after screening runs below. Reserved here so an urgent finding
    # appears at the top of the page rather than below the fold.
    alert_slot = st.empty()

    top_left, top_right = st.columns([1, 1.45])

    with top_left:
        with card("1. Load the report", "\U0001F4C4"):
            st.file_uploader(
                "Report file", type=["pdf", "txt", "md"], key="report_file",
                on_change=load_uploaded_report, label_visibility="collapsed",
                help="PDF or plain text. Scanned PDFs need OCR, which is not connected.",
            )
            if st.session_state.report_note:
                tone = "note-amber" if st.session_state.report_ocr else "note-green"
                st.markdown(f"<div class='{tone}'>{st.session_state.report_note}</div>",
                            unsafe_allow_html=True)
            if st.session_state.report_ocr:
                st.markdown(
                    "<div class='note-amber'><b>Check the text below before "
                    "summarising.</b> OCR misreads characters, and a wrong digit "
                    "in a dose or a measurement carries clinical risk. Correct "
                    "anything that looks wrong -- the summary uses this text, not "
                    "the original image.</div>",
                    unsafe_allow_html=True)
            st.caption("Or paste the report text directly:")
            st.text_area("Report text", key="report_input", height=210,
                         label_visibility="collapsed",
                         placeholder="Paste consultation notes, radiology reports, "
                                     "discharge summaries...")

            language_choice = st.radio(
                "Summary language",
                ["Match the report", "English", "العربية"],
                horizontal=True,
                help="The summary is written in this language whatever language "
                     "the report itself uses. Drug names, doses and figures are "
                     "always kept exactly as the report writes them.")
            summary_language = {"Match the report": "auto", "English": "english",
                                "العربية": "arabic"}[language_choice]

            with_evidence = st.checkbox("Also retrieve matching guideline evidence",
                                        value=True)
            action, reset = st.columns([2, 1])
            summarize = action.button("⚡  Summarise report", type="primary",
                                      use_container_width=True)
            reset.button("Clear", on_click=clear_report, use_container_width=True)

    if summarize:
        report_text = st.session_state.report_input.strip()
        if not report_text:
            st.warning("Upload a file or paste some report text first.")
        elif not configured:
            st.error(config.MISSING_KEY_MESSAGE)
        else:
            with st.spinner("Reading the report and summarising..."):
                try:
                    from backend import summarize_report, _chat as backend_chat
                    summarise_chat = lambda p: backend_chat(p, job="summary")

                    # Scope gate. A document outside the guideline corpus is
                    # refused before anything is generated, so the app never
                    # produces a fluent summary it has no evidence for.
                    verdict = scope.classify_document(report_text,
                                                      chat=summarise_chat)
                    st.session_state.out_of_scope = (
                        None if verdict["category"] == scope.IN_SCOPE else verdict)

                    if st.session_state.out_of_scope:
                        st.session_state.summary = ""
                        st.session_state.summary_sources = []
                        st.session_state.red_flags = []
                        st.session_state.screened = False
                        st.session_state.saved_id = None
                    else:
                        summary, summary_sources = summarize_report(
                            report_text, with_evidence=with_evidence,
                            language=summary_language)
                        st.session_state.summary = summary
                        st.session_state.summary_sources = summary_sources
                        st.session_state.saved_id = None
                        # Screen the report itself, not the summary, so nothing
                        # the summariser dropped can hide an urgent finding.
                        st.session_state.red_flags = safety.screen_report(
                            report_text, chat=summarise_chat)
                        st.session_state.screened = True
                except Exception as exc:
                    st.error(f"Error: {exc}")

    if st.session_state.out_of_scope:
        verdict = st.session_state.out_of_scope
        refusal = scope.REFUSAL[verdict["category"]]
        detected = (f"<div class='flag'><div><div class='n'>Detected subject</div>"
                    f"<div class='q'>{verdict['subject']}</div></div>"
                    f"<div class='src'>{verdict['source']}</div></div>")
        alert_slot.markdown(
            "<div class='alert-red'><div class='t'>&#9888;&#65039; "
            + refusal["title"] + "</div><div class='b'>"
            + refusal["body"] + "</div>" + detected + "</div>",
            unsafe_allow_html=True)
    elif st.session_state.red_flags:
        rows = "".join(
            f"<div class='flag'><div><div class='n'>{flag['concern']}</div>"
            f"<div class='q'>&ldquo;{flag['quote']}&rdquo;</div></div>"
            f"<div class='src'>{flag['source']}</div></div>"
            for flag in st.session_state.red_flags)
        alert_slot.markdown(
            "<div class='alert-red'><div class='t'>&#9888;&#65039; "
            + safety.BANNER_TITLE + "</div><div class='b'>"
            + safety.BANNER_BODY + "</div>" + rows + "</div>",
            unsafe_allow_html=True)
    elif st.session_state.screened:
        alert_slot.markdown(f"<div class='note-green'>{safety.NO_FLAG_NOTE}</div>",
                            unsafe_allow_html=True)

    with top_right:
        with card("2. Structured summary", "\U0001F9FE"):
            if st.session_state.summary:
                render_markdown(st.session_state.summary)
                st.markdown(
                    "<div class='note-amber'>The summary is restricted to what the "
                    "report itself says. Anything missing is marked as not stated "
                    "rather than filled in.</div>",
                    unsafe_allow_html=True,
                )
            elif st.session_state.out_of_scope:
                empty_state("<b>No summary generated.</b><br>This document is "
                            "outside OsteoGuard’s osteoarthritis evidence "
                            "base — see the notice above.")
            else:
                empty_state("Load a report and press <b>Summarise report</b>.<br>"
                            "The summary appears here, split into findings, "
                            "diagnoses, current management and follow-up.")

        if st.session_state.summary_sources:
            with card("Guideline evidence for this report", "\U0001F4DA"):
                render_evidence(
                    st.session_state.summary_sources,
                    caption="Retrieved from the guidelines because they match the "
                            "report's topic. This is guideline text, not content "
                            "from the report.",
                )

    # ---------------- patient context ----------------
    mid_left, mid_mid, mid_right = st.columns([1.1, 1, 1])

    with mid_left:
        with card("Patient information", "\U0001F464"):
            age_col, age_unit = st.columns([2, 1])
            age = age_col.number_input("Age", 18, 100, 62)
            age_unit.markdown("<div style='padding-top:34px;color:#6b8fa3;"
                              "font-size:.84rem'>years</div>", unsafe_allow_html=True)
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            bmi_col, bmi_unit = st.columns([2, 1])
            bmi = bmi_col.number_input("BMI", 12.0, 60.0, 28.4, step=0.1)
            bmi_unit.markdown("<div style='padding-top:34px;color:#6b8fa3;"
                              "font-size:.84rem'>kg/m&sup2;</div>", unsafe_allow_html=True)
            knee = st.selectbox("Affected joint", ["Right knee", "Left knee", "Both knees",
                                                   "Hip", "Hand"])
            dur_col, dur_unit = st.columns([2, 1])
            duration = dur_col.number_input("Symptom duration", 0, 600, 12)
            dur_unit.markdown("<div style='padding-top:34px;color:#6b8fa3;"
                              "font-size:.84rem'>months</div>", unsafe_allow_html=True)
            previous_injury = st.checkbox("Previous joint injury")
            family_history = st.checkbox("Family history of OA")
            heavy_load = st.checkbox("High physical / occupational load")

    with mid_mid:
        with card("Risk factors", "⚠️"):
            factors = risk.risk_factors(age, bmi, previous_injury,
                                        family_history, heavy_load)
            for factor in factors:
                colour = BAR_COLOR.get(factor["level"], "#6b8fa3")
                st.markdown(
                    f"<div class='risk'><div class='n'>{factor['name']}</div>"
                    f"<div class='bar-bg'><div class='bar-fill' "
                    f"style='width:{factor['value'] * 100:.0f}%;background:{colour};'>"
                    f"</div></div>"
                    f"<div class='l' style='color:{colour};'>{factor['level']}</div></div>",
                    unsafe_allow_html=True,
                )
            level, score = risk.overall_risk(factors)
            st.write("")
            row("Overall risk profile",
                f"<span class='pill {RISK_PILL.get(level, 'pill-grey')}'>{level}</span>")
            st.caption("Transparent thresholds from established OA risk factors - "
                       "not a trained model.")

    with mid_right:
        with card("Export", "\U0001F4E5"):
            summary_text = st.session_state.summary or "No summary generated yet."
            report_lines = f"""OsteoGuard AI - clinical summary

Patient: {age}y {gender}, BMI {bmi}, {knee}, symptoms {duration} months
Risk profile: {risk.overall_risk(factors)[0]}
Risk factors: {", ".join(f"{f['name']}={f['level']}" for f in factors)}

--- REPORT SUMMARY ---
{summary_text}

--- NOTE ---
Generated by an AI assistant to support clinical decision-making.
Verify against the current published guideline before acting.
"""
            st.download_button("\U0001F4C4  Download summary (TXT)", report_lines,
                               file_name="osteoguard_summary.txt",
                               type="primary", use_container_width=True)
            st.write("")
            patient_ref = st.text_input(
                "Patient reference", key="patient_ref",
                placeholder="e.g. MRN 44821 or initials",
                help="Your own identifier for this record. Saved locally.")
            if st.button("Save to patient records", use_container_width=True):
                if not st.session_state.summary:
                    st.warning("Generate a summary before saving.")
                else:
                    record_id = records.save_assessment(
                        patient_ref=patient_ref or "Unnamed",
                        age=age, gender=gender, bmi=bmi, joint=knee,
                        duration=duration,
                        risk_level=risk.overall_risk(factors)[0],
                        summary=st.session_state.summary,
                        red_flags=st.session_state.red_flags,
                        sources=st.session_state.summary_sources,
                        notes=st.session_state.get("clinician_notes", ""))
                    st.session_state.saved_id = record_id
                    st.success(f"Saved as record #{record_id}.")
            st.write("")
            st.markdown(
                "<div class='note-blue'>The export contains the summary and the "
                "entered patient context. Retrieved guideline passages stay in the "
                "app, where their citations remain clickable.</div>",
                unsafe_allow_html=True,
            )

    # ---------------- recommendations ----------------
    bottom_left, bottom_right = st.columns([1.6, 1])

    with bottom_left:
        with card("Clinical recommendations", "\U0001F4CB"):
            if st.button("⚡  Generate guideline-cited recommendations",
                         type="primary", use_container_width=True):
                if not configured:
                    st.error(config.MISSING_KEY_MESSAGE)
                else:
                    question = (f"What are the recommended management options for a "
                                f"{age}-year-old patient with {knee.lower()} "
                                f"osteoarthritis, BMI {bmi}, symptoms for "
                                f"{duration} months?")
                    with st.spinner("Retrieving guideline evidence..."):
                        try:
                            from backend import generate_response
                            answer, sources, _ = generate_response(
                                question, top_n=5)
                            st.session_state.recommendations = answer
                            st.session_state.recommendation_sources = sources
                        except Exception as exc:
                            st.error(f"Error: {exc}")

            if st.session_state.recommendations:
                render_markdown(st.session_state.recommendations)
                render_evidence(st.session_state.recommendation_sources)
                videos = resources.exercise_videos(
                    st.session_state.recommendations, knee)
                if videos:
                    st.markdown("**Exercise demonstrations**")
                    video_links(videos)
            else:
                items = [
                    "Exercise and weight management are first-line for knee OA.",
                    "Topical NSAIDs are considered before oral NSAIDs.",
                    "Physiotherapy to maintain joint function and strength.",
                    "Review analgesia regularly and at the lowest effective dose.",
                    "Consider orthopaedic referral if symptoms limit daily life.",
                ]
                st.markdown(
                    "<div class='checklist'>" + "".join(
                        f"<div class='check' style='animation-delay:{i * 60}ms'>"
                        f"<span class='tick'>✓</span><span>{item}</span></div>"
                        for i, item in enumerate(items)) + "</div>",
                    unsafe_allow_html=True,
                )
                st.caption("General orientation only. Press the button above for "
                           "recommendations retrieved and cited from the guidelines "
                           "for this specific patient.")

    with bottom_right:
        with card("Clinician notes", "\U0001F4AC"):
            st.text_area("Notes", key="clinician_notes",
                         placeholder="Add your own notes...",
                         label_visibility="collapsed", height=150)
            st.markdown(
                "<div class='note-green'>Notes are kept for this session, and "
                "are written to the local record file if you press <b>Save to "
                "patient records</b>. Report text is sent to the generation API "
                "only when you press a generate button.</div>",
                unsafe_allow_html=True,
            )


# ==========================================================================
# 3. PATIENT RECORDS
# ==========================================================================
elif page == "Patient Records":
    saved = records.list_assessments()
    db_path, db_bytes = records.store_location()

    left, right = st.columns([1.7, 1])

    with left:
        with card(f"Saved assessments ({len(saved)})", "👤"):
            if not saved:
                empty_state("No assessments saved yet.<br>Summarise a report, "
                            "then press <b>Save to patient records</b>.")
            for entry in saved:
                flags = json.loads(entry["red_flags"] or "[]")
                when = entry["created_at"].replace("T", " ")
                pill = (f"<span class='pill pill-red'>{len(flags)} red flag"
                        f"{'' if len(flags) == 1 else 's'}</span>"
                        if flags else
                        f"<span class='pill {RISK_PILL.get(entry['risk_level'], 'pill-grey')}'>"
                        f"{entry['risk_level'] or 'no risk score'}</span>")
                st.markdown(
                    f"<div class='rec'><div><div class='who'>#{entry['id']} &middot; "
                    f"{entry['patient_ref']}</div><div class='when'>{when} &middot; "
                    f"{entry['age'] or '?'}y {entry['gender'] or ''} &middot; "
                    f"{entry['joint'] or 'joint not recorded'}</div></div>{pill}</div>",
                    unsafe_allow_html=True)

                with st.expander(f"Open record #{entry['id']}"):
                    full = records.get_assessment(entry["id"])
                    if full:
                        if full["red_flags"]:
                            st.markdown(
                                "<div class='alert-red'><div class='t'>"
                                "&#9888;&#65039; Red flags recorded</div>"
                                + "".join(
                                    f"<div class='flag'><div><div class='n'>"
                                    f"{flag['concern']}</div><div class='q'>"
                                    f"&ldquo;{flag['quote']}&rdquo;</div></div></div>"
                                    for flag in full["red_flags"]) + "</div>",
                                unsafe_allow_html=True)
                        st.markdown(full["summary"] or "_No summary stored._")
                        if full["notes"]:
                            st.markdown("**Clinician notes**")
                            st.write(full["notes"])
                        if full["sources"]:
                            st.markdown("**Guideline citations**")
                            for source in full["sources"]:
                                st.markdown(
                                    f"- {source.get('doc_name')} p.{source.get('page')} "
                                    f"({source.get('section') or 'no section'})")
                        st.download_button(
                            "Download this record", full["summary"] or "",
                            file_name=f"osteoguard_record_{full['id']}.txt",
                            key=f"dl_{full['id']}")
                        if st.button("Delete this record", key=f"del_{full['id']}"):
                            records.delete_assessment(full["id"])
                            st.rerun()

    with right:
        with card("Where records are kept", "🗄"):
            st.markdown(
                "<div class='note-blue'>Records are stored in a single SQLite "
                "file on this machine. Nothing is uploaded, and no cloud "
                "service is involved.</div>",
                unsafe_allow_html=True)
            st.write("")
            row("Records", str(len(saved)))
            row("File size", f"{db_bytes / 1024:.0f} KB" if db_bytes else "empty")
            st.caption(db_path)

        with card("Handle with care", "🔒"):
            st.markdown(
                "<div class='note-amber'>This file holds clinical text and "
                "whatever reference you typed, so treat it as patient data: it "
                "is not encrypted, it is not backed up, and anyone with access "
                "to this machine can read it. It is git-ignored so it cannot be "
                "committed by accident.</div>",
                unsafe_allow_html=True)


elif page == "Statistics":
    try:
        from backend import corpus_stats
        stats = corpus_stats()
    except Exception:
        stats = None

    tiles = st.columns(4)
    figures = [
        ("Precision@5", "87.6%", "measured on 50 clinical queries"),
        ("Calibrated confidence", "88.2%", "Platt-scaled to match precision"),
        ("Indexed passages", str(stats["chunks"]) if stats else "n/a",
         "from the guideline corpus"),
        ("Guideline sections", str(stats["sections"]) if stats else "n/a",
         "distinct section headings"),
    ]
    for column, (label, value, caption) in zip(tiles, figures):
        column.markdown(
            f"<div class='stat'><div class='t'>{label}</div>"
            f"<div class='v'>{value}</div><div class='c'>{caption}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns(2)

    with left:
        with card("Retrieval quality", "\U0001F4C8"):
            st.markdown("""
| Configuration | Precision@5 | Confidence |
|---|---|---|
| Previous (ms-marco-MiniLM-L-6) | 78.0% | 71.0% |
| Current (MedCPT + fixes) | **87.6%** | **88.2%** |
""")
            st.caption("Measured over a 50-query evaluation set covering "
                       "pharmacological, non-pharmacological and surgical topics.")

        with card("Speed / accuracy modes", "⚙️"):
            st.markdown("""
| Mode | Precision@5 | Latency per query |
|---|---|---|
| accurate | 87.6% | ~14 s |
| demo | 84.4% | ~6 s |
| fast | 82.8% | ~4 s |
""")
            st.caption("Set with OSTEOGUARD_MODE in configuration. Latency measured on CPU.")

    with right:
        with card("Corpus composition", "\U0001F4DA"):
            if stats:
                for document in stats["documents"]:
                    share = document["chunks"] / max(stats["chunks"], 1) * 100
                    row(document["name"], f"{document['chunks']} passages "
                                          f"({share:.0f}%)")
                    bar(share)
                    st.write("")
                st.caption(f"{stats['chunks']} passages · {stats['sections']} sections "
                           f"· up to page {stats['max_page']}.")
            else:
                st.caption("Guideline index not readable from disk.")

        with card("What is measured", "\U0001F50D"):
            st.markdown(
                "<div class='note-blue'><b>Precision@5</b> is the share of the top "
                "five retrieved passages judged relevant to the query. It measures "
                "the <i>retrieval</i> step - the part this system controls. It is "
                "not a measure of clinical correctness of the final wording, which "
                "still needs a clinician's review.</div>",
                unsafe_allow_html=True,
            )


# ==========================================================================
# 5. ABOUT
# ==========================================================================
else:
    left, right = st.columns([1.4, 1])

    with left:
        with card("What this system does", "ℹ️"):
            st.markdown("""
**Evidence retrieval.** Hybrid search over NICE NG226 and the ACR/AF 2019
osteoarthritis guideline: PubMedBERT embeddings and BM25 keyword search run in
parallel, are fused with reciprocal rank fusion, reranked by the MedCPT
biomedical cross-encoder, and answered strictly over the retrieved text. Confidence is Platt-calibrated so the reported figure tracks measured
precision rather than a raw model score.

**Report summarisation.** Digital PDFs are read from their text layer; scans
and phone photographs are detected by text density and read with local OCR, so
nothing but the recovered text ever leaves the machine. A dedicated
summarisation model - configured and
tuned separately from the assistant - condenses an uploaded or pasted clinical
report into a fixed set of headings: findings, diagnoses, current management,
follow-up. The prompt forbids inference, so anything the report does not state
is returned as *not stated in the report* rather than guessed. Guideline
evidence matching the report's clinical terms is retrieved separately, by the
same hybrid search that serves the assistant, and labelled as guideline text.
""")

        with card("Limits worth knowing", "⚠️"):
            st.markdown(
                "<div class='note-amber'>"
                "<b>No imaging analysis.</b> This system does not read X-rays or "
                "any other image, and does not produce Kellgren-Lawrence grades.<br><br>"
                "<b>OCR is approximate.</b> Scanned reports are read locally with OCR, which misreads characters. Check the extracted text before summarising "
                "-- a wrong digit in a dose is a clinical risk.<br><br>"
                "<b>Two guidelines only.</b> Answers reflect NICE NG226 and ACR/AF "
                "2019 and nothing else - including nothing published since.<br><br>"
                "<b>Records are stored unencrypted, locally.</b> Saving an "
                "assessment writes it to a SQLite file on this machine. It is "
                "not encrypted and not backed up, and anyone with access to the "
                "machine can read it. Nothing is uploaded."
                "</div>",
                unsafe_allow_html=True,
            )

    with right:
        with card("Sources", "\U0001F4D6"):
            st.markdown("""
- [NICE NG226 - Osteoarthritis in over 16s](https://www.nice.org.uk/guidance/ng226)
- [ACR/AF 2019 Osteoarthritis Guideline](https://rheumatology.org/osteoarthritis-guideline)
- [World Health Organization - Osteoarthritis in over 16s](https://www.who.int/publications/i/item/9789241550044)
- [National Institute for Health and Care Excellence - Osteoarthritis in over 16s](https://www.nice.org.uk/guidance/ng226)
- [Egyptian Ministry of Health and Population](https://www.mohp.gov.eg/Content/Guidelines/osteoarthritis.pdf)   
""")

        with card("Stack", "\U0001F9F0"):
            for label, value in [
                ("Interface", "Streamlit"),
                ("Embeddings", "NeuML/pubmedbert-base-embeddings"),
                ("Keyword search", "BM25 Okapi"),
                ("Reranker", "ncbi/MedCPT-Cross-Encoder"),
                ("Vector store", "ChromaDB"),
                ("Assistant model", config.model_name("assistant")),
                ("Summary model", config.model_name("summary")),
                ("Inference", config.LLM_PROVIDER.title()),
            ]:
                row(label, value)

        with card("Intended use", "\U0001F6E1️"):
            st.markdown(
                "<div class='note-blue'>A decision-support aid for clinicians. It "
                "does not diagnose, does not prescribe, and does not replace "
                "professional medical judgement.</div>",
                unsafe_allow_html=True,
            )


st.markdown(
    "<div class='footer-bar'><span>OsteoGuard AI &middot; guideline-grounded "
    "decision support</span>"
    "<span><i>Improving lives through smarter musculoskeletal care</i></span></div>",
    unsafe_allow_html=True,
)
