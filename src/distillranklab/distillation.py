"""Foundational probability and knowledge-distillation operations.

Scores are grouped by query; the final dimension enumerates candidate passages.
"""

import numpy as np


def _scores(scores, name):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonempty candidate dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def softmax(logits, temperature=1.0):
    """Return stable softmax probabilities over the last dimension."""
    scores = _scores(logits, "logits")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scaled = scores / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def kl_divergence(teacher_probs, student_probs):
    """Compute KL(teacher || student) per query (last axis: candidates).

    Both inputs must be strictly positive probability distributions.
    """
    teacher = _scores(teacher_probs, "teacher_probs")
    student = _scores(student_probs, "student_probs")
    if teacher.shape != student.shape:
        raise ValueError("teacher and student probabilities must have equal shapes")
    for name, probs in (("teacher", teacher), ("student", student)):
        if np.any(probs <= 0) or not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-8):
            raise ValueError(f"{name} must be a strictly positive probability distribution")
    return np.sum(teacher * (np.log(teacher) - np.log(student)), axis=-1)


def distillation_loss(teacher_logits, student_logits, temperature=1.0):
    """Mean temperature-scaled KL loss, with the conventional T**2 factor."""
    teacher = _scores(teacher_logits, "teacher_logits")
    student = _scores(student_logits, "student_logits")
    if teacher.shape != student.shape:
        raise ValueError("teacher and student logits must have equal shapes")
    teacher_probs = softmax(teacher, temperature)
    student_probs = softmax(student, temperature)
    return float(np.mean(kl_divergence(teacher_probs, student_probs)) * temperature**2)
