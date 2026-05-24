import inspect

from rtms_rag.rag import generate_answer, generate_answer_with_reflexion


def test_generate_answer_signature():
    assert callable(generate_answer)
    assert str(inspect.signature(generate_answer)).startswith("(question: str")


def test_generate_answer_with_reflexion_signature():
    assert callable(generate_answer_with_reflexion)
    assert str(inspect.signature(generate_answer_with_reflexion)).startswith("(question: str")
