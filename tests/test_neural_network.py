from brain.neural_network import MLP


def test_mlp_forward_shape():
    net = MLP(2, 4, 1)
    output = net.forward([0.0, 1.0])
    assert len(output) == 1
    assert 0.0 < output[0] < 1.0


def test_mlp_reduces_xor_loss():
    net = MLP(2, 4, 1, seed=7)
    samples = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    initial = sum(net.train_step(x, y, 0.8) for x, y in samples) / len(samples)
    history = net.train(samples, epochs=800, learning_rate=0.8)
    assert history[-1].loss < initial
