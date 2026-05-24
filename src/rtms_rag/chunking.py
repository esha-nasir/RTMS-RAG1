def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100):
    """
    Chunk text by words. chunk_size and overlap are in words.
    """
    words = text.split()
    chunks = []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks