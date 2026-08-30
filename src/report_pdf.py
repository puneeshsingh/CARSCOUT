from fpdf import FPDF

# fpdf2's built-in fonts (Helvetica etc.) only support latin-1 - GPT output
# occasionally includes characters outside that (smart quotes, en/em dashes)
# which would raise a UnicodeEncodeError mid-render. Simpler to normalize to
# a plain-ASCII equivalent than to bundle a Unicode TTF font just for this.
_CHAR_REPLACEMENTS = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
}

TILE_LABELS = {
    "reliability": "Reliability", "price": "Price fairness",
    "recalls": "Recall history", "safety": "Safety rating",
}


def _sanitize(text: str) -> str:
    for char, replacement in _CHAR_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _write_line(pdf: FPDF, text: str, size: int, bold: bool = False, line_height: int = 6) -> None:
    """multi_cell() doesn't reset x back to the left margin when it
    finishes (it ends near wherever the last line's text stopped) - the
    next multi_cell(0, ...) call then has almost no width left to work
    with and raises FPDFException. Reset x explicitly before every call."""
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B" if bold else "", size)
    pdf.multi_cell(0, line_height, _sanitize(text))


def build_report_pdf(
    vehicle_line: str,
    listing_line: str,
    tiles: dict,
    full_answer: str,
) -> bytes:
    """Builds a simple one-report PDF: title, listing details, the tile
    headlines, then the full report text. No markdown rendering (bold/
    headers) - fpdf2's core fonts don't need it, and the plain text is
    already readable on its own."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    _write_line(pdf, vehicle_line, size=16, bold=True, line_height=10)
    _write_line(pdf, listing_line, size=11, line_height=7)
    pdf.ln(4)

    _write_line(pdf, "At a glance", size=12, bold=True, line_height=8)
    for signal, label in TILE_LABELS.items():
        tile = tiles.get(signal, {"headline": "No data"})
        _write_line(pdf, f"{label}: {tile['headline']}", size=10)
    pdf.ln(4)

    _write_line(pdf, "Full report", size=12, bold=True, line_height=8)
    clean_answer = full_answer.replace("**", "").replace("#", "").replace("`", "")
    _write_line(pdf, clean_answer, size=10)

    return bytes(pdf.output())
