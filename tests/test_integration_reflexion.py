from unittest.mock import patch
import json

import rtms_rag.rag as rag


def fake_retrieve(query: str, top_k: int = 5, **kwargs):
    # return a single simple context
    return [
        {
            "text": "According to document A, the answer is 42.",
            "chunk_id": "A-1",
            "file_name": "docA.pdf",
            "source_file_path": "data/docA.pdf",
        }
    ]


def test_generate_answer_with_reflexion_mocked():
    # Patch retrieval, completion, and memory appenders to run a deterministic reflexion flow
    def fake_call_completion(prompt: str, temperature: float = 0.2, max_tokens: int = 1000, json_mode: bool = False):
        # JSON-mode calls (skeptic) should return a JSON string
        if json_mode:
            return json.dumps({
                "supported_claims": ["According to document A, the answer is 42."],
                "unsupported_claims": [],
                "verdict": "accept",
                "guidance": "Answer is grounded.",
            })
        # Non-JSON calls return a plain-text answer
        return "Initial answer: 42"

    with patch("rtms_rag.rag.retrieve", side_effect=fake_retrieve) as mock_retrieve, \
        patch("rtms_rag.rag._call_completion", side_effect=fake_call_completion) as mock_llm, \
        patch("rtms_rag.rag._append_memory") as mock_append_mem:

        # Run the reflexion-enabled API surface
        out = rag.generate_answer_with_reflexion("What is the answer?", top_k=1, max_iters=2)

        # function returns (answer_text, contexts, trace)
        assert isinstance(out, tuple) and len(out) == 3
        answer_text, contexts, trace = out
        assert isinstance(answer_text, str)
        assert contexts and contexts[0].get("file_name") == "docA.pdf"
        # ensure retrieve and LLM were called during the process
        assert mock_retrieve.called
        assert mock_llm.called
        # memory appender was patched to avoid I/O; ensure no exception and callable invoked zero-or-more times
        assert mock_append_mem is not None
