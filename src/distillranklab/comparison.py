"""Controlled hard-label, distillation, and hybrid ranking experiments.

Each query is a group of candidate feature vectors. The student assigns a
linear score to each candidate and is trained with listwise cross-entropy.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray  # (queries, candidates, dimensions)
    labels: np.ndarray    # (queries,) index of relevant candidate
    teacher_logits: np.ndarray  # (queries, candidates)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    z = np.asarray(logits, dtype=float) / temperature
    if z.ndim < 1 or z.shape[-1] == 0 or not np.all(np.isfinite(z)):
        raise ValueError('logits must be finite with a nonempty last axis')
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def make_dataset(n_queries: int, *, seed: int, teacher_error_rate: float = 0.0,
                 n_candidates: int = 4) -> Dataset:
    """Synthetic ranking task with an imperfect, higher-capacity teacher.

    Ground truth depends on linear and nonlinear features. Students only see
    the first two features, so the teacher has information they cannot fit.
    """
    if n_queries <= 0 or n_candidates < 2 or not 0 <= teacher_error_rate <= 1:
        raise ValueError('invalid dataset parameters')
    rng = np.random.default_rng(seed)
    full = rng.normal(size=(n_queries, n_candidates, 3))
    relevance = 1.5 * full[..., 0] - 0.6 * full[..., 1] + 0.8 * full[..., 2]
    labels = relevance.argmax(axis=1)
    teacher = relevance + rng.normal(scale=0.25, size=relevance.shape)
    corrupt = rng.random(n_queries) < teacher_error_rate
    # Corrupt selected queries by boosting a known incorrect candidate.
    for q in np.flatnonzero(corrupt):
        wrong = (int(labels[q]) + 1) % n_candidates
        teacher[q, wrong] = np.max(teacher[q]) + 3.0
    return Dataset(full[..., :2], labels, teacher)


def train_student(data: Dataset, *, mode: str, steps: int = 500,
                  learning_rate: float = 0.15, temperature: float = 2.0,
                  alpha: float = 0.5) -> np.ndarray:
    """Fit a linear ranker; alpha weights KD in hybrid mode.

    Hard loss: CE(one-hot, softmax(scores)).
    KD loss: T^2 CE(softmax(teacher/T), softmax(scores/T)).
    The constant teacher entropy is irrelevant to the gradient.
    """
    if mode not in {'hard', 'soft', 'hybrid'}:
        raise ValueError('mode must be hard, soft, or hybrid')
    if steps < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('invalid optimization parameters')
    if not 0 <= alpha <= 1 or not np.isfinite(alpha):
        raise ValueError('alpha must be between 0 and 1')
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    x = np.asarray(data.features, dtype=float)
    y = np.asarray(data.labels)
    t = np.asarray(data.teacher_logits, dtype=float)
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] < 2 or x.shape[2] == 0:
        raise ValueError('invalid feature shape')
    if t.shape != x.shape[:2] or y.shape != (x.shape[0],):
        raise ValueError('labels or teacher shape mismatch')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= x.shape[1])):
        raise ValueError('invalid dataset values')
    weights = np.zeros(x.shape[-1], dtype=float)
    targets = np.eye(x.shape[1])[y]
    teacher_probs = softmax(t, temperature)
    for _ in range(steps):
        scores = x @ weights
        hard_gradient = softmax(scores) - targets
        kd_gradient = temperature * (softmax(scores, temperature) - teacher_probs)
        if mode == 'hard':
            gradient = hard_gradient
        elif mode == 'soft':
            gradient = kd_gradient
        else:
            gradient = (1 - alpha) * hard_gradient + alpha * kd_gradient
        weights -= learning_rate * np.einsum('qcd,qc->d', x, gradient) / len(x)
    return weights


def evaluate(data: Dataset, weights: np.ndarray) -> dict[str, float]:
    """Top-1 relevance accuracy and teacher top-1 agreement."""
    predicted = (data.features @ weights).argmax(axis=1)
    return {
        'accuracy': float(np.mean(predicted == data.labels)),
        'teacher_agreement': float(np.mean(predicted == data.teacher_logits.argmax(axis=1))),
    }
