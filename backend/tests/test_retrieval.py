import pytest

from docqa.bm25 import BM25, stem, tokenize
from docqa.chunking import chunk_markdown, pack, split_sections, window
from docqa.index import HybridIndex


def test_headings_drop_anchor_and_nest_under_title():
    md = "# Title { #t }\n\nintro text\n\n## Part { #p }\n\nbody text\n"
    assert split_sections(md) == [("Title", "intro text"), ("Title > Part", "body text")]


def test_hash_inside_code_block_is_not_a_heading():
    md = "# T\n\n```bash\n# not a heading\npip install x\n```\n"
    sections = split_sections(md)
    assert len(sections) == 1 and "# not a heading" in sections[0][1]


def test_window_overlap_and_validation():
    words = [str(i) for i in range(10)]
    parts = window(words, max_words=4, overlap=1)
    assert parts[0] == ["0", "1", "2", "3"] and parts[1][0] == "3"
    with pytest.raises(ValueError):
        window(words, max_words=2, overlap=2)


def test_pack_keeps_code_blocks_whole():
    body = "short intro.\n\n```python\nline one\n\nline two after blank\n```\n\nafter code."
    parts = pack(body, max_words=12, overlap=2)
    assert any("line one" in p and "line two after blank" in p for p in parts)


def test_chunk_ids_are_unique(chunks):
    assert len({c.id for c in chunks}) == len(chunks)


def test_tokenize_drops_stopwords_and_optionally_stems():
    assert tokenize("How do I return the errors?") == ["return", "errors"]
    assert tokenize("queries and errors", stemmed=True) == ["query", "error"]
    assert stem("status") == "status"


def test_bm25_ranks_matching_document_first():
    bm = BM25().fit(["cats purr softly", "dogs bark loudly at night", "a cat and a dog"])
    assert bm.search("dogs bark", k=1)[0][0] == 1
    assert bm.idf("bark") > bm.idf("missing-term") - 10  # idf is defined for unseen terms
    assert bm.search("zebra") == []


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
@pytest.mark.parametrize(
    "question,doc",
    [
        ("How do I return a 404 when an item is not found?", "errors.md"),
        ("Let my Vite frontend on another origin call the API", "cors.md"),
        ("send an email after the response without waiting", "background.md"),
    ],
)
def test_each_mode_finds_the_right_doc(index, mode, question, doc):
    assert index.search(question, k=1, mode=mode)[0].chunk.doc == doc


def test_unknown_mode_raises(index):
    with pytest.raises(ValueError):
        index.search("x", mode="nope")


def test_empty_index_rejected():
    with pytest.raises(ValueError):
        HybridIndex().build([])


def test_save_and_load_round_trip(index, tmp_path):
    p = tmp_path / "idx.pkl"
    index.save(p)
    again = HybridIndex.load(p)
    q = "CORS origins"
    assert [h.chunk.id for h in again.search(q)] == [h.chunk.id for h in index.search(q)]


def test_chunk_markdown_records_doc_and_heading():
    c = chunk_markdown("a.md", "# A\n\nhello world here")[0]
    assert (c.doc, c.heading, c.id) == ("a.md", "A", "a.md#0")
