# 🩺 OsteoGuard AI

**Built for the AI Hackathon by Orange Digital Center, Creativa, and ITIDA**

Clinical decision support for osteoarthritis management, grounded in NICE NG226
and the ACR/AF 2019 osteoarthritis guideline.

## 👥 Team: The Innovators

* **Youssef Elbasiouny** — AI / Software Engineering
* **Ibrahim Ahmed** — Team Leader
* **Ali Sherif** — QA Testing

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials**

   The API key is configured once at deployment and is never entered through the
   interface. Copy `.env.example` to `.env` beside the application and fill it in:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Real environment variables work too and take precedence over the file. `.env`
   is git-ignored, so a key cannot be committed by accident.

   Until a key is present, the interface reports the evidence engine as offline
   and states what to set. Other optional settings:

   | Variable | Purpose |
   |---|---|
   | `ASSISTANT_MODEL` | Model answering guideline questions |
   | `SUMMARY_MODEL` | Model summarising reports, tuned separately |
   | `OSTEOGUARD_MODE` | Retrieval preset: `accurate` (default), `balanced`, `fast` |
   | `LLM_PROVIDER` | `groq` (default) or `gemini` |

3. **Run the application**

   ```bash
   python -m streamlit run app.py
   ```

   Run it from the repository root so that `.env` and `.streamlit/config.toml`
   (which pins the light clinical theme) are picked up.

## What it does

**Clinical Assistant** — ask an osteoarthritis management question in any
language. The question is translated for search, answered only from retrieved
guideline passages, and every passage is shown with its document, page and
calibrated confidence.

**Report Summary** — upload a PDF or text clinical report (or paste it) and get
a structured summary under fixed headings: findings, diagnoses, current
management, flagged items, follow-up. Digital PDFs are read from their text
layer; scans and phone photographs are detected by text density and passed
through local OCR, so the document itself never leaves the machine. The prompt forbids inference, so anything
the report does not state comes back as *not stated in the report* rather than
being filled in. Guideline evidence matching the report's topic is retrieved
separately and labelled as guideline text.

Before anything is generated, the document is classified against the scope of
the evidence base. Only osteoarthritis and joint / musculoskeletal reports are
summarised; a clinical document about another specialty, or a non-medical file,
is refused with an explanation and no summary is produced. A document that
plainly names osteoarthritis is admitted without an extra model call.

Every summarised report is also screened for red flags — infection, fracture,
suspected malignancy, avascular necrosis, cord compression, DVT, complete
tendon rupture, systemic features. A finding proposed by the model must carry a
verbatim quote that is then checked against the report text, so the screen
cannot raise an alarm the document does not support.

When an answer is about exercise, YouTube search links for the relevant joint
are offered alongside it. They are searches rather than specific videos: the
app cannot vet an individual video's content.

**Patient Records** — assessments saved to a local SQLite file. No cloud, no
account, no upload. Browse, open, export or delete them from the page.

**Statistics** — retrieval performance and live corpus composition read from the
vector store.

## Architecture

- **Frontend**: Streamlit (`app.py`, styled by `theme.css`)
- **Retrieval**: BM25 keyword search with clinical abbreviation expansion, run
  alongside `NeuML/pubmedbert-base-embeddings` dense search in ChromaDB, fused
  with reciprocal rank fusion and reranked by `ncbi/MedCPT-Cross-Encoder`
- **Generation**: Groq (`openai/gpt-oss-120b`), answering strictly over
  retrieved context. Answering and summarising are configured separately so
  tuning one cannot regress the other.
- **Data**: NICE NG226 and ACR/AF 2019, 238 indexed passages

Retrieval runs entirely locally — the embedding model, the reranker and the
vector store are all on disk. Only the generation step calls out to an API.

Measured on a 50-query clinical evaluation set:

| Configuration | Precision@5 | Confidence |
|---|---|---|
| Previous (ms-marco-MiniLM-L-6) | 78.0% | 71.0% |
| Current (MedCPT + fixes) | **87.6%** | **88.2%** |

Retrieval presets trade latency against precision:

| Preset | Precision@5 | Latency per query |
|---|---|---|
| `accurate` (default) | 87.6% | ~14 s |
| `balanced` | 84.4% | ~6 s |
| `fast` | 82.8% | ~4 s |

## Files

```
app.py                    the application
theme.css                 clinical green / blue / white theme
config.py                 credentials and engine settings, from the environment
backend.py                retrieval, answering, report summarisation
reports.py                PDF / text extraction, with OCR for scans
scope.py                  scope gate - refuses out-of-scope documents
safety.py                 red-flag screening with quote verification
records.py                local SQLite patient record store
resources.py              exercise video search links
risk.py                   rule-based OA risk factor display
data/
  osteoarthritis_db/      Chroma index the app reads (238 passages)
  guidelines/             the two source PDFs
notebooks/
  Rag_model.ipynb         corpus build and evaluation
docs/                     hackathon briefing material
```

The notebook builds `data/osteoarthritis_db`; the application reads it. The
retrieval logic in `backend.py` is a port of the notebook's Cell 8, so a change
to one needs mirroring in the other.

## Limits

- **No imaging analysis.** The system does not read X-rays and does not produce
  Kellgren–Lawrence grades. The earlier simulated X-ray module was removed.
- **OCR output is approximate.** Scans and phone photographs are read with
  local OCR, which misreads characters. The extracted text is shown for review
  before summarising, and it should be checked — a wrong digit in a dose or a
  measurement carries clinical risk. OCR is capped at the first 20 pages.
- **Two guidelines only**, including nothing published since them.
- **The scope gate is a topic check, not a clinical judgement.** It decides
  whether a document is about osteoarthritis, nothing more.
- **Red-flag screening is not triage.** It reports terms found in the report
  text against a fixed list. It cannot see what a report does not say, and the
  absence of a flag is never clearance.
- **Saved records are unencrypted.** `osteoguard_records.db` is a plain SQLite
  file on the machine, git-ignored and not backed up. Anyone with access to the
  machine can read it.

A decision-support aid for clinicians. It does not diagnose, does not prescribe,
and does not replace professional medical judgement.
