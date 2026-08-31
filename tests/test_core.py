from brain.core import GrowingBrain


def test_learning_persists(tmp_path):
    brain = GrowingBrain()
    lesson = brain.learn("neuron", "A neuron uses weights and bias.")
    assert lesson.topic == "neuron"
    assert brain.search_memory("weights")


def test_code_request_is_human_only(tmp_path):
    brain = GrowingBrain()
    path = brain.request_code(
        "Attention module",
        "The current brain needs attention.",
        ["Accept embeddings", "Return transformed representation"],
        "brain/modules/attention.py",
    )
    text = path.read_text(encoding="utf-8")
    assert "WAITING_FOR_HUMAN_CODE" in text
    assert "must NOT write the implementation code" in text
