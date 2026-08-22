"""OsteoGuard AI - runtime configuration.

Credentials and engine settings are read from the environment here, never from
the user interface. Deployment sets them once in a `.env` file next to this
module (or as real environment variables) and the application picks them up on
start.

This module is deliberately dependency-light: the UI imports it to check
whether the system is configured without pulling in the retrieval stack.
"""

import os

from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))

# A .env beside the application wins; a .env in the working directory is also
# honoured so the app can be launched from either location. Neither overrides
# a variable that is already set in the real environment.
load_dotenv(os.path.join(_HERE, ".env"))
load_dotenv()


# --- Language model ---------------------------------------------------------
# Groq serves the generation step: it is markedly faster than the previous
# Gemini path, and the Gemini SDK this project used is end-of-life.
# Retrieval is unaffected -- embeddings and reranking run locally.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# openai/gpt-oss-120b is the default after testing the alternatives on this
# corpus: the smaller gpt-oss-20b mistranslated the Arabic term for
# osteoarthritis as "bone fractures", which is not acceptable clinically.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- The two jobs are configured separately ---------------------------------
# Answering guideline questions and summarising a report are different tasks
# and are tuned independently, so changing one cannot regress the other.
#
# Both default to gpt-oss-120b because it was the only candidate that held up
# on a report with deliberate omissions: it kept all seven headings, reproduced
# every figure exactly, and returned "not stated in the report" for the missing
# follow-up section instead of inventing one. qwen3.6-27b leaked its reasoning
# into the answer, dropped every heading and ran past the token limit.
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", GROQ_MODEL)
ASSISTANT_TEMPERATURE = float(os.getenv("ASSISTANT_TEMPERATURE", "0.2"))
ASSISTANT_MAX_TOKENS = int(os.getenv("ASSISTANT_MAX_TOKENS", "4096"))

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", GROQ_MODEL)
# 0.2 is the measured setting: at 1.0 this model family drops into runs of
# filler characters that truncate the output. Do not raise it without retesting.
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.2"))
# Report summaries are longer than a guideline answer, so they get more room.
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "3000"))

# Optional fallback provider, kept working for deployments with a Gemini key.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

_KEY_VARIABLE = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}


def key_variable():
    """Name of the environment variable this deployment reads its key from."""
    return _KEY_VARIABLE.get(LLM_PROVIDER, "GROQ_API_KEY")


def model_name(job="assistant"):
    """The model id serving a given job: "assistant" or "summary"."""
    if LLM_PROVIDER == "gemini":
        return GEMINI_MODEL
    return SUMMARY_MODEL if job == "summary" else ASSISTANT_MODEL


def get_api_key():
    """The provider's API key, or None when nothing has been configured."""
    key = os.getenv(key_variable(), "").strip()
    return key or None


def api_key_configured():
    return get_api_key() is not None


# Retrieval preset: "accurate" (default), "balanced" or "fast".
RETRIEVAL_MODE = os.getenv("OSTEOGUARD_MODE", "accurate")

MISSING_KEY_MESSAGE = (
    f"OsteoGuard is not configured: no API key was found. Set "
    f"{key_variable()} in the .env file beside the application, or as an "
    f"environment variable, and restart."
)
