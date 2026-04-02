"""Load classical corpus files from disk."""

from __future__ import annotations

import logging
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from app.domain.models import LoadedDocument

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".txt", ".md", ".epub"}
CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
OPF_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from XHTML/HTML fragments."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"script", "style"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "section", "article", "br", "li", "h1", "h2", "h3", "h4"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in {"p", "div", "section", "article", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def _parse_front_matter(raw_text: str) -> tuple[dict[str, str], str]:
    """Parse very small YAML-like front matter without extra dependencies."""

    stripped = raw_text.strip()
    if not stripped.startswith("---"):
        return {}, raw_text

    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_text

    metadata: dict[str, str] = {}
    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')

    if end_index is None:
        return {}, raw_text

    body = "\n".join(lines[end_index + 1 :]).strip()
    return metadata, body


def _infer_metadata_from_path(path: Path) -> tuple[str, str]:
    stem = path.stem.replace("-", "_")
    if "__" in stem:
        author, work = stem.split("__", 1)
        return author.replace("_", " ").title(), work.replace("_", " ").title()
    return "Unknown", stem.replace("_", " ").title()


def _extract_epub_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


def _load_epub(path: Path) -> LoadedDocument:
    """Load an EPUB file using only the Python standard library."""

    fallback_author, fallback_work = _infer_metadata_from_path(path)
    with zipfile.ZipFile(path, "r") as archive:
        container_xml = archive.read("META-INF/container.xml")
        container_root = ET.fromstring(container_xml)
        rootfile = container_root.find(".//c:rootfile", CONTAINER_NS)
        if rootfile is None:
            raise ValueError(f"EPUB container missing rootfile entry: {path}")

        opf_path = rootfile.attrib.get("full-path", "")
        if not opf_path:
            raise ValueError(f"EPUB rootfile path is empty: {path}")

        opf_root = ET.fromstring(archive.read(opf_path))
        opf_dir = posixpath.dirname(opf_path)

        title = opf_root.findtext(".//dc:title", default=fallback_work, namespaces=OPF_NS)
        author = opf_root.findtext(".//dc:creator", default=fallback_author, namespaces=OPF_NS)
        language = opf_root.findtext(".//dc:language", default="unknown", namespaces=OPF_NS)

        manifest: dict[str, str] = {}
        for item in opf_root.findall(".//opf:manifest/opf:item", OPF_NS):
            item_id = item.attrib.get("id", "")
            href = item.attrib.get("href", "")
            media_type = item.attrib.get("media-type", "")
            if not item_id or not href:
                continue
            if media_type in {"application/xhtml+xml", "text/html"} or href.endswith(
                (".xhtml", ".html", ".htm")
            ):
                manifest[item_id] = posixpath.normpath(posixpath.join(opf_dir, href))

        spine_paths: list[str] = []
        for itemref in opf_root.findall(".//opf:spine/opf:itemref", OPF_NS):
            idref = itemref.attrib.get("idref", "")
            href = manifest.get(idref)
            if href:
                spine_paths.append(href)

        if not spine_paths:
            spine_paths = list(manifest.values())

        chapters: list[str] = []
        for chapter_path in spine_paths:
            try:
                chapter_html = archive.read(chapter_path).decode("utf-8", errors="ignore")
            except KeyError:
                LOGGER.warning("EPUB chapter not found in archive: %s (%s)", path, chapter_path)
                continue
            text = _extract_epub_text(chapter_html)
            if text:
                chapters.append(text)

        body = "\n\n".join(chapters).strip()
        return LoadedDocument(
            source_path=str(path),
            text=body,
            author=author.strip() or fallback_author,
            work=title.strip() or fallback_work,
            reference_prefix="section",
            metadata={
                "corpus_type": "epub",
                "source_note": "Extracted from EPUB container.",
                "language": language.strip() or "unknown",
            },
        )


def load_corpus(input_path: Path) -> list[LoadedDocument]:
    """Load supported documents from a directory."""

    if not input_path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {input_path}")

    documents: list[LoadedDocument] = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".epub":
            document = _load_epub(path)
        else:
            raw_text = path.read_text(encoding="utf-8")
            metadata, body = _parse_front_matter(raw_text)
            author, work = _infer_metadata_from_path(path)
            document = LoadedDocument(
                source_path=str(path),
                text=body.strip(),
                author=metadata.get("author", author),
                work=metadata.get("work", work),
                reference_prefix=metadata.get("reference_prefix", "fragment"),
                metadata={
                    "corpus_type": metadata.get("corpus_type", "unknown"),
                    "source_note": metadata.get("source_note", ""),
                    "language": metadata.get("language", "en"),
                },
            )
        if not document.text:
            LOGGER.warning("Skipping empty corpus file: %s", path)
            continue
        documents.append(document)

    return documents
