"""Experiment 02: distill a frozen synthetic teacher into a linear student.

Run: python examples/02_train_linear_student.py
"""
import numpy as np

from distillranklab.student import train_student
from distillranklab.distillation import softmax


def main():
    # Three queries, four candidates each, two observable features per candidate.
    # Teacher is deliberately expressible by a linear function of these features.
    features = np.array([
        [[2, 0], [1, 1], [0, 2], [0, 0]],
        [[0, 2], [2, 0], [1, 0], [1, 2]],
        [[1, 1], [0, 0], [2, 1], [0, 2]],
    ], dtype=float)
    teacher_weights = np.array([1.5, -0.5])  # Used only to synthesize labels.
    teacher_logits = features @ teacher_weights

    student, history = train_student(features, teacher_logits, steps=500,
                                     learning_rate=0.1, temperature=2.0, seed=42)
    student_logits = student.predict(features)
    print(f"Initial distillation loss: {history[0]:.6f}")
    print(f"Final distillation loss:   {history[-1]:.6f}")
    print("Learned student weights:", np.round(student.weights, 3))
    print("Teacher top-ranked IDs: ", np.argmax(teacher_logits, axis=1))
    print("Student top-ranked IDs: ", np.argmax(student_logits, axis=1))
    print("Teacher probabilities for query 1:", np.round(softmax(teacher_logits[0], 2), 3))
    print("Student probabilities for query 1:", np.round(softmax(student_logits[0], 2), 3))
    print("\nTry changing temperature, learning_rate, or feature capacity.")


if __name__ == "__main__":
    main()
