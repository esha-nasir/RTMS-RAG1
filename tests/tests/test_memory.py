import json

from rtms_rag.compute_hpr import compute_hpr


def test_compute_hpr(tmp_path):
    path = tmp_path / "predictions.jsonl"
    row = {
        "trace": {
            "trace": {
                "attempts": [
                    {"skeptic": {"unsupported_claims": ["unsupported claim"]}}
                ]
            }
        },
        "predicted_answer": "final answer",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = compute_hpr(path, threshold=0.75)
    assert result["records"] == 1
    assert result["initial_unsupported_claims"] == 1
    assert result["hpr_any"] == 0.0
