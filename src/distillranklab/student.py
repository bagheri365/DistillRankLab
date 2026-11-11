"""A tiny linear ranking student trained from frozen teacher scores.

Shapes: features (queries, candidates, features), teacher (queries, candidates).
The teacher scores are fixed; only the student weights are updated.
"""

from dataclasses import dataclass
import numpy as np

from .distillation import distillation_loss, softmax


def _validate(features, teacher_logits):
    x = np.asarray(features, dtype=np.float64)
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError("features must have shape (queries, candidates, features), all nonempty")
    if teacher.shape != x.shape[:2]:
        raise ValueError("teacher_logits must have shape (queries, candidates)")
    if not np.isfinite(x).all() or not np.isfinite(teacher).all():
        raise ValueError("features and teacher_logits must be finite")
    return x, teacher


@dataclass
class LinearStudent:
    """Shared linear scoring function for each query–candidate pair."""

    weights: np.ndarray

    def predict(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 3 or x.shape[-1] != self.weights.size:
            raise ValueError("features have incompatible shape")
        if not np.isfinite(x).all():
            raise ValueError("features must be finite")
        return x @ self.weights


def loss_and_gradient(features, teacher_logits, weights, temperature=1.0):
    """Return T² KL loss and exact gradient for a linear student.

    For N queries, dL/d(student_logits) = T * (p_student - p_teacher) / N.
    A shared bias is intentionally omitted: softmax is invariant to per-query shifts.
    """
    x, teacher = _validate(features, teacher_logits)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (x.shape[-1],) or not np.isfinite(w).all():
        raise ValueError("weights must be a finite vector matching feature count")
    logits = x @ w
    loss = distillation_loss(teacher, logits, temperature)
    delta = temperature * (softmax(logits, temperature) - softmax(teacher, temperature))
    gradient = np.einsum("qcf,qc->f", x, delta) / x.shape[0]
    return loss, gradient


def train_student(features, teacher_logits, *, steps=300, learning_rate=0.1,
                  temperature=2.0, seed=0):
    """Full-batch gradient descent, returning (student, loss_history).

    Deterministic initialization and full-batch updates keep the experiment easy
    to reproduce. No teacher parameters are changed.
    """
    x, teacher = _validate(features, teacher_logits)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    weights = np.random.default_rng(seed).normal(0, 0.1, size=x.shape[-1])
    history = [distillation_loss(teacher, x @ weights, temperature)]
    for _ in range(steps):
        _, gradient = loss_and_gradient(x, teacher, weights, temperature)
        weights -= learning_rate * gradient
        history.append(distillation_loss(teacher, x @ weights, temperature))
    return LinearStudent(weights), np.asarray(history)
