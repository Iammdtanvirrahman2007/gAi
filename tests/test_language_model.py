from pathlib import Path

from brain.language_model import NGramLanguageModel


def test_model_learns_and_persists(tmp_path: Path):
    path = tmp_path / "model.json"
    model = NGramLanguageModel(path)
    learned = model.train("Neural networks learn useful patterns.")
    assert learned > 0
    assert path.exists()

    restored = NGramLanguageModel(path)
    assert restored.trained_tokens > 0
    generated = restored.generate("neural networks", max_tokens=5)
    assert isinstance(generated, str)
