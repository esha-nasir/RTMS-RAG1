import inspect

from rtms_rag.retrieve import retrieve


def test_retrieve_signature():
    assert callable(retrieve)
    assert str(inspect.signature(retrieve)).startswith("(query: str")
