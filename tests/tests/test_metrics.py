from rtms_rag.chunking import chunk_text


def test_chunk_text():
    chunks = chunk_text("one two three four five", chunk_size=2, overlap=1)
    assert chunks
    assert chunks[0] == "one two"
