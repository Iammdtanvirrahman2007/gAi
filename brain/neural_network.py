"""Small dependency-free trainable multilayer perceptron for gAi."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class TrainingPoint:
    epoch: int
    loss: float


class MLP:
    """A tiny dense neural network with sigmoid activations and backpropagation."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, seed: int = 7) -> None:
        if min(input_size, hidden_size, output_size) <= 0:
            raise ValueError("Network sizes must be positive.")
        rng = random.Random(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        scale_in = 1.0 / math.sqrt(input_size)
        scale_hidden = 1.0 / math.sqrt(hidden_size)
        self.w1 = [[rng.uniform(-scale_in, scale_in) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[rng.uniform(-scale_hidden, scale_hidden) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size
        self.last_input = [0.0] * input_size
        self.last_hidden = [0.0] * hidden_size
        self.last_output = [0.0] * output_size

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = max(-60.0, min(60.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, x: list[float]) -> list[float]:
        if len(x) != self.input_size:
            raise ValueError("Input size mismatch.")
        self.last_input = [float(v) for v in x]
        hidden = []
        for j in range(self.hidden_size):
            z = self.b1[j] + sum(self.last_input[i] * self.w1[i][j] for i in range(self.input_size))
            hidden.append(self._sigmoid(z))
        output = []
        for k in range(self.output_size):
            z = self.b2[k] + sum(hidden[j] * self.w2[j][k] for j in range(self.hidden_size))
            output.append(self._sigmoid(z))
        self.last_hidden = hidden
        self.last_output = output
        return output

    def train_step(self, x: list[float], target: list[float], learning_rate: float = 0.5) -> float:
        output = self.forward(x)
        if len(target) != self.output_size:
            raise ValueError("Target size mismatch.")

        output_delta = [
            (output[k] - float(target[k])) * output[k] * (1.0 - output[k])
            for k in range(self.output_size)
        ]
        hidden_delta = []
        for j in range(self.hidden_size):
            carry = sum(output_delta[k] * self.w2[j][k] for k in range(self.output_size))
            h = self.last_hidden[j]
            hidden_delta.append(carry * h * (1.0 - h))

        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.w2[j][k] -= learning_rate * self.last_hidden[j] * output_delta[k]
        for k in range(self.output_size):
            self.b2[k] -= learning_rate * output_delta[k]

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.w1[i][j] -= learning_rate * self.last_input[i] * hidden_delta[j]
        for j in range(self.hidden_size):
            self.b1[j] -= learning_rate * hidden_delta[j]

        return sum((output[k] - float(target[k])) ** 2 for k in range(self.output_size)) / self.output_size

    def train(self, samples: list[tuple[list[float], list[float]]], epochs: int = 1000, learning_rate: float = 0.5) -> list[TrainingPoint]:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        history: list[TrainingPoint] = []
        for epoch in range(1, epochs + 1):
            total = 0.0
            for x, target in samples:
                total += self.train_step(x, target, learning_rate)
            loss = total / len(samples)
            if epoch == 1 or epoch % max(1, epochs // 100) == 0 or epoch == epochs:
                history.append(TrainingPoint(epoch, loss))
        return history

    def predict(self, x: list[float]) -> list[float]:
        return self.forward(x)

    def snapshot(self) -> dict:
        return {
            "architecture": [self.input_size, self.hidden_size, self.output_size],
            "layers": [
                {"name": "input", "size": self.input_size, "values": self.last_input},
                {"name": "hidden", "size": self.hidden_size, "values": self.last_hidden},
                {"name": "output", "size": self.output_size, "values": self.last_output},
            ],
            "weights": {"input_hidden": self.w1, "hidden_output": self.w2},
            "biases": {"hidden": self.b1, "output": self.b2},
        }
