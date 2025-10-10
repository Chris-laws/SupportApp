import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

_SECTION_PATTERN = re.compile(r"^\s*(?P<number>\d+(?:\.\d+)*)\s+(?P<title>[A-Za-z0-9].*)")


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


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 200, overlap: int = 50) -> List[Dict[str, Any]]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(documents)

    document_title = os.path.splitext(os.path.basename(pdf_path))[0]
    enriched_chunks: List[Dict[str, Any]] = []
    last_heading: Tuple[Optional[str], Optional[str]] = (None, None)

    for chunk in chunks:
        metadata = chunk.metadata or {}
        source = metadata.get("source") or pdf_path
        page = metadata.get("page")
        record, last_heading = _build_chunk_record(
            chunk.page_content,
            document_title,
            source,
            page,
            last_heading,
        )
        enriched_chunks.append(record)

    return enriched_chunks
