"""Medical report ingestion.

Turns an uploaded report (PDF or plain text) into the raw text that
`backend.summarize_report` works from. No interpretation happens here -- this
module only recovers characters.

Two kinds of PDF arrive in practice:

* Digital PDFs, which carry a text layer that is read directly.
* Scans and phone photographs (CamScanner and similar), which carry no text
  layer at all. These are recognised and passed through local OCR.

A scan is detected by text density, not by an empty page: a CamScanner export
typically contains a single word -- its own watermark -- which is enough to
look like a successful extraction while carrying no clinical content.
"""

# Below this many characters per page a PDF is treated as a scan rather than a
# document with a text layer. A real page of clinical text runs to hundreds.
SCAN_CHARS_PER_PAGE = 100

# OCR settings. 200 dpi is a good balance: high enough for the small print on a
# phone scan, low enough to keep a page around five seconds on CPU.
OCR_DPI = 200
OCR_MAX_PAGES = 20

_ocr_engine = None


def ocr_available():
    """True when the local OCR engine can be loaded."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _get_ocr():
    """Load the OCR engine once and reuse it across reruns."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _open_pdf(data):
    try:
        import pymupdf
    except ImportError:  # older installs only expose the deprecated alias
        import fitz as pymupdf
    return pymupdf.open(stream=data, filetype="pdf")


def _read_pdf_text(data):
    """Text layer of a PDF, with the page count."""
    with _open_pdf(data) as doc:
        pages = [page.get_text("text") for page in doc]
        return "\n\n".join(pages), doc.page_count


def _ocr_pdf(data):
    """Render each page and read it with local OCR.

    Returns (text, pages_read, pages_total). Nothing leaves the machine.
    """
    engine = _get_ocr()
    chunks = []

    with _open_pdf(data) as doc:
        total = doc.page_count
        for index, page in enumerate(doc):
            if index >= OCR_MAX_PAGES:
                break
            image = page.get_pixmap(dpi=OCR_DPI).tobytes("png")
            result, _ = engine(image)
            if result:
                chunks.append(" ".join(line[1] for line in result))

    return "\n\n".join(chunks), min(total, OCR_MAX_PAGES), total


def _read_text(data):
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text(filename, data):
    """Extract text from an uploaded report.

    Returns (text, note, used_ocr). `note` is a short human-readable line about
    what was read, or an error message when extraction failed. `used_ocr` is
    True when the text came from OCR and should be treated as approximate.
    """
    if not data:
        return "", "The uploaded file is empty.", False

    name = (filename or "").lower()

    if not name.endswith(".pdf"):
        try:
            text = _read_text(data).strip()
        except Exception as exc:
            return "", f"Could not read the file: {exc}", False
        if not text:
            return "", "That file contains no text.", False
        return text, f"Read {len(text.split()):,} words from the text file.", False

    try:
        text, pages = _read_pdf_text(data)
    except Exception as exc:
        return "", f"Could not read the PDF: {exc}", False

    text = text.strip()
    if len(text) >= SCAN_CHARS_PER_PAGE * max(pages, 1):
        words = len(text.split())
        return text, f"Read {words:,} words from the PDF.", False

    # Sparse text layer: a scan, a photograph, or a watermark-only export.
    if not ocr_available():
        return "", ("This PDF has no readable text layer -- it is a scan or a "
                    "photograph. OCR is not installed, so its text cannot be "
                    "recovered. Install rapidocr-onnxruntime, or paste the "
                    "report text into the box below."), False

    try:
        text, read, total = _ocr_pdf(data)
    except Exception as exc:
        return "", f"This PDF is a scan and OCR failed: {exc}", False

    text = text.strip()
    if not text:
        return "", ("This PDF is a scan and OCR could not find any text in it. "
                    "Try a sharper scan, or paste the report text below."), False

    plural = "page was" if read == 1 else "pages were"
    note = (f"No text layer found, so {read} scanned {plural} read with OCR "
            f"({len(text.split()):,} words).")
    if total > read:
        note += f" Only the first {read} of {total} pages were processed."
    return text, note, True
