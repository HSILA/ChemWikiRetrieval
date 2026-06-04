from pathlib import Path

from nq.scripts.extract_article_paragraphs import extract_file


FIXTURE = Path("nq/html_pages/1_2-rearrangement__oldid_742212898.html")
MOHS_FIXTURE = Path("nq/html_pages/Mohs_scale_of_mineral_hardness__oldid_845315263.html")
ASBESTOS_FIXTURE = Path("nq/html_pages/Asbestos__oldid_818240066.html")


def rows_by_section():
    rows = extract_file(FIXTURE)
    return {tuple(row["section_path"]): row for row in rows}


def test_12_rearrangement_merges_html_paragraphs_into_section_blocks():
    by_section = rows_by_section()

    lead = by_section[("Lead",)]
    assert "A 1,2-rearrangement" in lead["text"]
    assert "structural isomers" in lead["text"]
    assert "1,2-hydride shift" in lead["text"]

    reaction = by_section[("Reaction mechanism",)]
    assert "The driving force for the actual migration" in reaction["text"]
    assert "Hückel's rule" in reaction["text"]
    assert "Wagner–Meerwein rearrangement" in reaction["text"]


def test_12_rearrangement_preserves_list_text_inside_sections():
    by_section = rows_by_section()

    reaction = by_section[("Reaction mechanism",)]
    assert "reactive intermediate such as" in reaction["text"]
    assert "carbocation by heterolysis" in reaction["text"]
    assert "carbanion" in reaction["text"]
    assert "free radical by homolysis" in reaction["text"]
    assert "nitrene" in reaction["text"]

    rearrangements = by_section[("1,2 rearrangements",)]
    assert "The following mechanisms involve a 1,2-rearrangement" in rearrangements["text"]
    assert "Beckmann rearrangement" in rearrangements["text"]
    assert "Wagner–Meerwein rearrangement" in rearrangements["text"]
    assert "Wolff rearrangement" in rearrangements["text"]

    rearrangements_13 = by_section[("1,3-Rearrangements",)]
    assert "1,3-rearrangements take place over 3 carbon atoms" in rearrangements_13["text"]
    assert "Fries rearrangement" in rearrangements_13["text"]
    assert "verbenone" in rearrangements_13["text"]


def test_12_rearrangement_emits_one_row_per_content_section_and_skips_references():
    rows = extract_file(FIXTURE)
    sections = [tuple(row["section_path"]) for row in rows]

    assert sections == [
        ("Lead",),
        ("Reaction mechanism",),
        ("Radical 1,2-rearrangements",),
        ("1,2 rearrangements",),
        ("1,3-Rearrangements",),
    ]
    assert all("References" not in row["section_path"] for row in rows)


def test_table_only_intro_section_is_not_emitted_as_clipped_text():
    rows = extract_file(MOHS_FIXTURE)
    sections = [tuple(row["section_path"]) for row in rows]

    assert ("Comparison with Vickers scale",) not in sections


def test_numeric_heading_text_is_preserved():
    rows = extract_file(ASBESTOS_FIXTURE)
    history = next(row for row in rows if row["section"] == "History of use")

    assert "11 September 2001 attacks" in history["subsections"]
    assert "September 2001 attacks" not in history["subsections"]


def test_headings_are_metadata_not_text():
    rows = extract_file(FIXTURE)

    assert rows[0]["section"] == "summary"
    assert rows[0]["section_path"] == ["Lead"]
    assert not rows[0]["text"].startswith("Lead")

    for row in rows[1:]:
        assert row["section"] != "summary"
        assert f"== {row['section']} ==" not in row["text"]
        assert f"=== {row['section']} ===" not in row["text"]
