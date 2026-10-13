"""
Train a Restricted Boltzmann Machine on Fashion-MNIST.

The RBM is implemented directly from its energy and conditional formulas.
NumPy is used for arrays and matrix products. TensorFlow is used only to
download Fashion-MNIST; the model, gradients, and training loop do not
use it.

Model
-----
Binary visible units v in {0, 1}^n and binary hidden units h in {0, 1}^m.

    E(v, h) = - a·v - b·h - v^T W h

    P(h_j = 1 | v) = sigmoid(b_j + sum_i v_i W_ij)
    P(v_i = 1 | h) = sigmoid(a_i + sum_j W_ij h_j)

    sigmoid(x) = 1 / (1 + exp(-x))

Training uses one-step contrastive divergence (CD-1) on mini-batches.
For a data matrix V (rows are examples) the updates are

    W <- W + lr * (V^T p(h|V) - V_neg^T p(h|V_neg)) / batch_size
    a <- a + lr * mean(V - V_neg)
    b <- b + lr * mean(p(h|V) - p(h|V_neg))

which are the CD approximation to the log-likelihood gradients

    d log P(v) / d W_ij  ≈  v_i h_j - v'_i h'_j
    d log P(v) / d a_i   ≈  v_i - v'_i
    d log P(v) / d b_j   ≈  h_j - h'_j

Hidden activities in the gradient are probabilities, not samples.
Samples are used only to drive the Gibbs step, following Hinton's
practical guide to training RBMs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist():
    """Download Fashion-MNIST with TensorFlow and return NumPy arrays.

    ``tf.keras.datasets.fashion_mnist.load_data`` is the only TensorFlow call.
    Images arrive as uint8 arrays of shape (n, 28, 28). They are scaled to
    [0, 1] and flattened to (n, 784) before training.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.asarray(x_train, dtype=np.float32).reshape(len(x_train), -1) / 255.0
    x_test = np.asarray(x_test, dtype=np.float32).reshape(len(x_test), -1) / 255.0
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    return x_train, y_train, x_test, y_test


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


class RestrictedBoltzmannMachine:
    """Binary-binary RBM trained with explicit CD-k updates."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.1,
        cd_k: int = 1,
        seed: int = 0,
    ):
        if cd_k < 1:
            raise ValueError("cd_k must be at least 1")
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.cd_k = cd_k
        self.rng = np.random.default_rng(seed)

        # Small random weights. Visible bias is set from the data mean in fit().
        self.W = self.rng.normal(0.0, 0.01, size=(n_visible, n_hidden)).astype(np.float32)
        self.a = np.zeros(n_visible, dtype=np.float32)  # visible bias
        self.b = np.zeros(n_hidden, dtype=np.float32)  # hidden bias

    def init_visible_bias(self, data: np.ndarray) -> None:
        """a_i = log(p_i / (1 - p_i)), with p_i the mean activation of unit i."""
        p = np.clip(data.mean(axis=0), 1e-3, 1.0 - 1e-3)
        self.a = np.log(p / (1.0 - p)).astype(np.float32)

    def p_h_given_v(self, v: np.ndarray) -> np.ndarray:
        """P(h = 1 | v) = sigmoid(v W + b). v is (batch, n_visible)."""
        return sigmoid(v @ self.W + self.b)

    def p_v_given_h(self, h: np.ndarray) -> np.ndarray:
        """P(v = 1 | h) = sigmoid(h W^T + a). h is (batch, n_hidden)."""
        return sigmoid(h @ self.W.T + self.a)

    def sample_bernoulli(self, probs: np.ndarray) -> np.ndarray:
        return (self.rng.random(probs.shape) < probs).astype(np.float32)

    def gibbs_hvh(self, h_sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        One full Gibbs step starting from a hidden sample:
            v ~ P(v | h), then return (v_sample, P(h | v)).
        """
        v_prob = self.p_v_given_h(h_sample)
        v_sample = self.sample_bernoulli(v_prob)
        h_prob = self.p_h_given_v(v_sample)
        return v_sample, h_prob

    def cd_gradients(self, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Contrastive-divergence gradients for one mini-batch.

        Positive phase uses the data (binarized pixel intensities) and the
        hidden probabilities. Negative phase runs cd_k Gibbs steps.
        """
        # Positive phase: h_j = P(h_j = 1 | v)
        h0_prob = self.p_h_given_v(v0)
        h_sample = self.sample_bernoulli(h0_prob)

        # Negative phase: k steps of Gibbs sampling.
        # Intermediate steps use samples; the last hidden state used in the
        # gradient is the probability P(h | v_k).
        v_neg = v0
        h_neg_prob = h0_prob
        for step in range(self.cd_k):
            v_neg, h_neg_prob = self.gibbs_hvh(h_sample)
            if step < self.cd_k - 1:
                h_sample = self.sample_bernoulli(h_neg_prob)

        batch = v0.shape[0]
        # <v h^T>_data - <v h^T>_model
        dW = (v0.T @ h0_prob - v_neg.T @ h_neg_prob) / batch
        # <v>_data - <v>_model
        da = (v0 - v_neg).mean(axis=0)
        # <h>_data - <h>_model
        db = (h0_prob - h_neg_prob).mean(axis=0)
        return dW, da, db

    def update(self, v0: np.ndarray) -> None:
        """Apply the explicit CD parameter updates."""
        dW, da, db = self.cd_gradients(v0)
        lr = self.learning_rate
        self.W += lr * dW
        self.a += lr * da
        self.b += lr * db

    def reconstruct(self, v: np.ndarray) -> np.ndarray:
        """Mean-field reconstruction: v -> P(h|v) -> P(v|h)."""
        return self.p_v_given_h(self.p_h_given_v(v))

    def reconstruction_cross_entropy(self, v: np.ndarray) -> float:
        recon = np.clip(self.reconstruct(v), 1e-7, 1.0 - 1e-7)
        ce = -(v * np.log(recon) + (1.0 - v) * np.log(1.0 - recon))
        return float(ce.mean())

    def fit(
        self,
        data: np.ndarray,
        epochs: int = 5,
        batch_size: int = 128,
        binarize: bool = True,
    ) -> list[float]:
        """
        Train on rows of `data` with values in [0, 1].

        If binarize is true, each presentation draws
            v_i ~ Bernoulli(pixel_i)
        so the binary RBM sees samples consistent with the gray level.
        """
        n = data.shape[0]
        history = []
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = data[order[start : start + batch_size]]
                if binarize:
                    batch = self.sample_bernoulli(batch)
                self.update(batch)

            # Monitor on a fixed subset so the curve is comparable across epochs.
            probe = data[:2048]
            if binarize:
                probe = (probe >= 0.5).astype(np.float32)
            loss = self.reconstruction_cross_entropy(probe)
            history.append(loss)
            print(f"epoch {epoch:3d}/{epochs}  reconstruction cross-entropy {loss:.4f}")
        return history


def save_pgm(path: Path, image: np.ndarray) -> None:
    """Write one grayscale image in [0, 1] as a binary PGM file."""
    img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    height, width = img.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + img.tobytes())


def save_comparison_grid(
    path: Path,
    originals: np.ndarray,
    reconstructions: np.ndarray,
    image_shape: tuple[int, int] = (28, 28),
) -> None:
    """Stack original (top) and reconstruction (bottom) into one PGM."""
    n = originals.shape[0]
    h, w = image_shape
    gap = 2
    canvas_h = h * 2 + gap
    canvas_w = n * w
    canvas = np.ones((canvas_h, canvas_w), dtype=np.float32)
    for i in range(n):
        canvas[0:h, i * w : (i + 1) * w] = originals[i].reshape(h, w)
        canvas[h + gap :, i * w : (i + 1) * w] = reconstructions[i].reshape(h, w)
    save_pgm(path, canvas)


def main() -> None:
    x_train, _y_train, x_test, y_test = load_fashion_mnist()
    print(f"train {x_train.shape}, test {x_test.shape}")

    rbm = RestrictedBoltzmannMachine(
        n_visible=x_train.shape[1],
        n_hidden=128,
        learning_rate=0.1,
        cd_k=1,
        seed=0,
    )
    rbm.init_visible_bias(x_train)

    history = rbm.fit(x_train, epochs=5, batch_size=128, binarize=True)

    # Reconstruct one test image from each class.
    picks = []
    for label in range(10):
        index = int(np.flatnonzero(y_test == label)[0])
        picks.append(index)
    originals = (x_test[picks] >= 0.5).astype(np.float32)
    reconstructions = rbm.reconstruct(originals)

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = out_dir / "reconstructions.pgm"
    save_comparison_grid(grid_path, originals, reconstructions)
    np.savez(
        out_dir / "rbm_weights.npz",
        W=rbm.W,
        a=rbm.a,
        b=rbm.b,
        history=np.asarray(history, dtype=np.float32),
    )

    test_probe = (x_test[:2048] >= 0.5).astype(np.float32)
    test_ce = rbm.reconstruction_cross_entropy(test_probe)
    print(f"test reconstruction cross-entropy {test_ce:.4f}")
    print("class order in the grid:", ", ".join(CLASS_NAMES))
    print(f"wrote {grid_path}")
    print(f"wrote {out_dir / 'rbm_weights.npz'}")


if __name__ == "__main__":
    main()
