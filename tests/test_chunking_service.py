from app.services.chunking_service import chunk_pages


def test_heading_stays_attached_to_following_section():
    text = (
        "Introduction\n\n"
        "This paper studies something interesting about search algorithms "
        "and how they perform.\n\n"
        "Methodology\n\n"
        "We used a standard approach involving several well-known "
        "techniques for evaluation."
    )
    # chunk_size must be small enough to actually force the two
    # heading+paragraph sections into separate chunks (combined they're
    # ~196 chars, so chunk_size=200 lets them fit in one chunk together --
    # trivially "passing" without ever exercising a chunk boundary between
    # them at all). 150 sits between the ~99-char first section and the
    # ~196-char combined total, so a split is guaranteed.
    chunks = chunk_pages([(1, text)], chunk_size=150, overlap=20)
    assert any(c.content.startswith("Introduction") for c in chunks)
    assert any(c.content.startswith("Methodology") for c in chunks)


def test_long_single_block_falls_back_to_char_slicing_within_chunk_size():
    long_paragraph = "word " * 400  # ~2000 chars, no blank lines -> one block
    chunks = chunk_pages([(1, long_paragraph)], chunk_size=800, overlap=150)
    assert len(chunks) > 1
    assert all(len(c.content) <= 800 for c in chunks)


def test_empty_pages_are_skipped():
    chunks = chunk_pages([(1, "   \n\n  "), (2, "Real content here.")], chunk_size=200, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_index_is_monotonic_across_the_document():
    text = "\n\n".join(f"Paragraph number {i} with some filler content to pad it out." for i in range(20))
    chunks = chunk_pages([(1, text)], chunk_size=150, overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_carry_over_never_pushes_a_chunk_past_chunk_size():
    """Regression test: a run of blocks each already close to chunk_size
    used to carry the *full* overlap into every chunk after the first,
    landing chunks at roughly chunk_size + overlap instead of at
    chunk_size — e.g. ~926 chars against an 800-char cap with 150-char
    overlap. The fix caps the carried-over tail to however much room is
    actually left for the next piece."""
    word_block = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod " * 12).strip()[:780]
    text = "\n\n".join([word_block] * 10)
    chunks = chunk_pages([(1, text)], chunk_size=800, overlap=150)
    assert all(len(c.content) <= 800 for c in chunks)
