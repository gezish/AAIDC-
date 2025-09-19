# ingest/clean.py
import re
from typing import List, Tuple

DIVIDER_PATTERN = re.compile(r"\s*-*DIVIDER-*\s*", flags=re.I)

def strip_md_images(text: str) -> str:
    # Remove markdown images: ![alt](path)
    return re.sub(r"!```math [^```]*```KATEX_INLINE_OPEN[^)]+KATEX_INLINE_CLOSE", "", text)

def strip_admonitions(text: str) -> str:
    # Remove :::info{...} tokens but keep inner content; drop raw ::: tokens
    text = re.sub(r":::\s*info\{[^}]*\}", "", text, flags=re.I)
    text = text.replace(":::", "\n")
    return text

def normalize_breaks(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = text.replace("</br>", "\n")
    # Convert horizontal rules to single newline
    text = re.sub(r"\n-{3,}\n", "\n", text)
    return text

def normalize_dividers(text: str) -> str:
    # Unify any variant of DIVIDER markup into newline-bounded token
    text = DIVIDER_PATTERN.sub("\nDIVIDER\n", text)
    text = text.replace("--DIVIDER--", "\nDIVIDER\n")
    return text

def basic_clean(text: str) -> str:
    if not text:
        return ""
    text = normalize_breaks(text)
    text = strip_admonitions(text)
    text = strip_md_images(text)
    # Fix stray hyphenation (mostly for PDFs; harmless for JSON)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def segment_by_divider_and_headings(markdown: str) -> List[Tuple[str, str]]:
    """
    Split publication_description into segments by DIVIDER markers,
    then determine a section title from first Markdown heading if present.
    Returns list of (section_title, content) pairs.
    """
    if not markdown:
        return []
    text = normalize_dividers(markdown)
    parts = [p.strip() for p in re.split(r"\n?DIVIDER\n?", text) if p and p.strip()]

    sections: List[Tuple[str, str]] = []
    for i, part in enumerate(parts):
        # Find first heading as section title
        m = re.search(r"^\s*#{1,6}\s+(.+)$", part, flags=re.M)
        section_title = m.group(1).strip() if m else f"Section {i+1}"
        # Remove first heading line from content to avoid duplication
        if m:
            content = re.sub(r"^\s*#{1,6}\s+.+$", "", part, count=1, flags=re.M)
        else:
            content = part
        content = basic_clean(content)
        # Skip segments that became empty (e.g., images only)
        if len(content.split()) < 5:
            continue
        sections.append((section_title, content))
    return sections

def normalize_for_dedup(text: str) -> str:
    t = text.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()