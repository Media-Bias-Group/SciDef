import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from lxml import etree  # type: ignore

from scidef.extraction.dataclass import ChunkMode
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


class SectionData(TypedDict):
    level: int
    title: Optional[str]
    paragraphs: List[str]
    subsections: List["SectionData"]


def split_into_sentences(text: str) -> list[str]:
    """
    SOURCE: https://stackoverflow.com/a/31505798

    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    alphabets = "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co|al)"
    starters = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r"\.{2,}"

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(
        multiple_dots,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>",
        text,
    )
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def extract_text_from_grobid(
    file_path: Path,
    chunk_mode: ChunkMode,
):
    tree = etree.parse(file_path)
    root = tree.getroot()

    # todo check if legal
    # if root.tag != "{http://www.tei-c.org/ns/1.0}TEI":
    #    raise TextExtractionError(f"Not a TEI XML file: {file_path}")

    body = _parse_body(root)

    return _chunk_body(body, chunk_mode)


#     section_data = {
#    "level": level,
#    "title": None,
#    "paragraphs": [],
#    "subsections": [],
# }


def _chunk_body(body, chunk_mode):
    """
    TODO add subsections (I don't think they exist though)
    """
    chunk = []

    if chunk_mode == ChunkMode.SENTENCE:
        for section in body:
            for paragraph in section["paragraphs"]:
                chunk.extend(split_into_sentences(paragraph))
    elif chunk_mode == ChunkMode.PARAGRAPH:
        for section in body:
            chunk.extend(section["paragraphs"])
    elif chunk_mode == ChunkMode.SECTION:
        for section in body:
            chunk.append(" ".join(section["paragraphs"]))
    elif chunk_mode == ChunkMode.FULL:
        temp_body = ""
        for section in body:
            temp_body += " ".join(section["paragraphs"])
        chunk = [temp_body]
    elif chunk_mode == ChunkMode.THREE_SENTENCE:
        all_sentences = []
        for section in body:
            for paragraph in section["paragraphs"]:
                all_sentences.extend(split_into_sentences(paragraph))

        for i in range(len(all_sentences) - 2):
            window = " ".join(all_sentences[i : i + 3])
            chunk.append(window)

        if len(all_sentences) == 2:
            chunk.append(" ".join(all_sentences))
        elif len(all_sentences) == 1:
            chunk.append(all_sentences[0])
    else:
        raise ValueError(f"Unsupported chunk mode: {chunk_mode}")

    return chunk


def _parse_body(root):
    sections: List[SectionData] = []
    body = root.xpath(".//tei:body", namespaces=NS)
    if not body:
        return sections

    for div in body[0].xpath("./tei:div", namespaces=NS):
        section_data = _parse_section_structure(div, level=1)
        if section_data:
            sections.append(section_data)

    return sections


def _parse_section_structure(div, level: int) -> SectionData:
    section_data: SectionData = {
        "level": level,
        "title": None,
        "paragraphs": [],
        "subsections": [],
    }

    head = div.xpath("./tei:head", namespaces=NS)
    if head:
        title = "".join(head[0].itertext())
        section_data["title"] = title

    for child in div:
        if child.tag == f"{{{NS['tei']}}}p":
            paragraph_text = "".join(child.itertext())
            if paragraph_text:
                section_data["paragraphs"].append(paragraph_text)
        elif child.tag == f"{{{NS['tei']}}}div":
            subsection = _parse_section_structure(child, level + 1)
            if subsection and (
                subsection["title"]
                or subsection["paragraphs"]
                or subsection["subsections"]
            ):
                section_data["subsections"].append(subsection)

    return section_data


def _parse_title(root):
    title_elem = root.xpath(
        './/tei:titleStmt/tei:title[@level="a"][@type="main"]',
        namespaces=NS,
    )
    if title_elem:
        return "".join(title_elem[0].itertext())

    title_elem = root.xpath(".//tei:titleStmt/tei:title", namespaces=NS)
    if title_elem:
        return "".join(title_elem[0].itertext())
    return ""


def _parse_year(root):
    date_elem = root.xpath(
        ".//tei:publicationStmt/tei:date[@type='published']",
        namespaces=NS,
    )
    if date_elem and date_elem[0].get("when"):
        try:
            year = int(date_elem[0].get("when")[:4])
            return year
        except (ValueError, TypeError):
            return ""


def _parse_abstract(root):
    abstract_elem = root.xpath(".//tei:abstract", namespaces=NS)
    if abstract_elem:
        paragraphs = []
        for p in abstract_elem[0].xpath(".//tei:p", namespaces=NS):
            text = "".join(p.itertext())
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)
    return ""


def _parse_authors(root):
    authors = []
    author_elems = root.xpath(".//tei:fileDesc//tei:author", namespaces=NS)

    for author_elem in author_elems:
        fn_elems = author_elem.xpath(".//tei:forename", namespaces=NS)
        ln_elems = author_elem.xpath(".//tei:surname", namespaces=NS)

        firstnames = []
        for fname_elem in fn_elems:
            if fname_elem.text:
                firstnames.append(fname_elem.text.strip())
        lastname = (
            ln_elems[0].text.strip() if ln_elems and ln_elems[0].text else ""
        )
        if firstnames and lastname:
            full_firstname = " ".join(firstnames)
            authors.append(f"{full_firstname} {lastname}")
        elif lastname:
            authors.append(lastname)

        return authors


def extract_metadata_from_grobid(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from GROBID XML file."""
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()

        metadata = {}
        metadata["title"] = _parse_title(root)
        metadata["authors"] = _parse_authors(root)
        metadata["year"] = _parse_year(root)

        return metadata
    except Exception:
        return {}


def _extract_element_text(element: ET.Element) -> str:
    """Extract clean text from an XML element (recursively).

    Uses ElementTree.itertext() to gather all nested text, then normalizes whitespace.
    """
    try:
        raw_text = "".join(element.itertext())
    except Exception:
        # Fallback to previous shallow extraction if itertext is unavailable
        text = element.text or ""
        for child in element:
            if child.text:
                text += child.text
            if child.tail:
                text += child.tail
        raw_text = text

    return re.sub(r"\s+", " ", raw_text.strip())


if __name__ == "__main__":
    path_file = Path(
        "ManualPDFsGROBID/manual_pdfs_grobid/paper_ffeb14ba3c28d84f87102577b4d35d2a2f80d608.grobid.tei.xml",
    )
