import logging
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

TOKEN_CHUNK_SIZE = 320
TOKEN_CHUNK_OVERLAP = 50
_SECTION_PATTERN = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)*)\s+(?P<title>[A-Za-z0-9].*)")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_ENCODING = None
try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENCODING = None

try:
    import ocrmypdf  # type: ignore[import]
except Exception:  # noqa: BLE001
    ocrmypdf = None  # type: ignore[assignment]


def _encoding():
    global _ENCODING
    if _ENCODING is None:
        try:
            import tiktoken

            _ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            _ENCODING = None
    return _ENCODING


def _token_length(text: str) -> int:
    enc = _encoding()
    if enc is None:
        return max(1, len(text.split()))
    return len(enc.encode(text))


def _split_into_chunks(text: str, chunk_size: int = TOKEN_CHUNK_SIZE, overlap: int = TOKEN_CHUNK_OVERLAP) -> Iterable[str]:
    sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT.split(text) if sentence.strip()]
    if not sentences:
        stripped = text.strip()
        if stripped:
            yield stripped
        return

    enc = _encoding()
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence)) if enc else max(1, len(sentence.split()))
        if current and current_tokens + sentence_tokens > chunk_size:
            yield " ".join(current).strip()
            # Build overlap
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for prev_sent in reversed(current):
                prev_len = len(enc.encode(prev_sent)) if enc else max(1, len(prev_sent.split()))
                if overlap_tokens + prev_len > overlap:
                    break
                overlap_sentences.append(prev_sent)
                overlap_tokens += prev_len
            current = list(reversed(overlap_sentences))
            current_tokens = sum(len(enc.encode(sent)) if enc else max(1, len(sent.split())) for sent in current)
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        yield " ".join(current).strip()


def _squeeze_text(text: str) -> str:
    """Remove whitespace and hyphen separators to speed up keyword matches."""
    return re.sub(r"[\s\-]+", "", text.lower())


def _detect_section_heading(text: str) -> Tuple[Optional[str], Optional[str]]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _SECTION_PATTERN.match(stripped)
        if match:
            number = match.group("number")
            title = match.group("title").strip()
            normalized = f"{number} {title}"
            return stripped, normalized
    return None, None


def _build_chunk_record(
    chunk_text: str,
    document_title: str,
    source: str,
    page: Optional[int],
    last_heading: Tuple[Optional[str], Optional[str]],
) -> Tuple[Dict[str, Any], Tuple[Optional[str], Optional[str]]]:
    raw_heading, normalized_heading = _detect_section_heading(chunk_text)
    if normalized_heading:
        last_heading = (raw_heading, normalized_heading)

    section_heading = raw_heading or (last_heading[0] if last_heading else None)
    normalized = normalized_heading or (last_heading[1] if last_heading else None)
    squeezed_content = _squeeze_text(chunk_text)
    heading_squeezed = _squeeze_text(normalized) if normalized else None

    record = {
        "content": chunk_text,
        "content_squeezed": squeezed_content,
        "source": source,
        "page": page,
        "document_title": document_title,
        "section_heading": section_heading,
        "normalized_section_heading": normalized,
        "normalized_section_heading_squeezed": heading_squeezed,
    }
    return record, last_heading


def _load_pdf_documents(pdf_path: str) -> List[Any]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    if any(doc.page_content.strip() for doc in documents):
        return documents

    if ocrmypdf is None:
        logger.warning("OCR fallback gewuenscht, aber ocrmypdf ist nicht installiert. Verwende Original-PDF.")
        return documents

    tmp_path = None
    try:
        with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        ocrmypdf.ocr(pdf_path, str(tmp_path), force_ocr=True, quiet=True)
        loader = PyPDFLoader(str(tmp_path))
        ocr_docs = loader.load()
        if not any(doc.page_content.strip() for doc in ocr_docs):
            logger.warning("OCR lieferte keine verwertbaren Texte. Verwende Original-PDF.")
            return documents
        return ocr_docs
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR-Verarbeitung fuer %s fehlgeschlagen: %s", pdf_path, exc)
        return documents
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


def _resolve_chunking() -> Tuple[int, int]:
    strategy = os.getenv("SUPPORTAPP_CHUNK_STRATEGY", "").strip().lower()
    if strategy == "fixed_500":
        return 500, 0
    if strategy == "semantic_overlap20":
        base = int(os.getenv("SUPPORTAPP_SEMANTIC_CHUNK_SIZE", "420"))
        overlap = int(max(1, round(base * 0.2)))
        return base, overlap
    size = int(os.getenv("SUPPORTAPP_CHUNK_SIZE", str(TOKEN_CHUNK_SIZE)))
    overlap = int(os.getenv("SUPPORTAPP_CHUNK_OVERLAP", str(TOKEN_CHUNK_OVERLAP)))
    return size, overlap


def get_active_chunk_config() -> Tuple[int, int]:
    return _resolve_chunking()


def load_and_chunk_pdf(pdf_path: str, chunk_size_tokens: int | None = None, overlap_tokens: int | None = None) -> List[Dict[str, Any]]:
    if chunk_size_tokens is None or overlap_tokens is None:
        chunk_size_tokens, overlap_tokens = _resolve_chunking()
    documents = _load_pdf_documents(pdf_path)

    document_title = os.path.splitext(os.path.basename(pdf_path))[0]
    enriched_chunks: List[Dict[str, Any]] = []
    last_heading: Tuple[Optional[str], Optional[str]] = (None, None)

    for doc in documents:
        metadata = doc.metadata or {}
        source = metadata.get("source") or pdf_path
        page = metadata.get("page")
        page_text = doc.page_content or ""

        for chunk_text in _split_into_chunks(page_text, chunk_size_tokens, overlap_tokens):
            record, last_heading = _build_chunk_record(
                chunk_text,
                document_title,
                source,
                page,
                last_heading,
            )
            record["token_length"] = _token_length(chunk_text)
            enriched_chunks.append(record)

    return enriched_chunks
