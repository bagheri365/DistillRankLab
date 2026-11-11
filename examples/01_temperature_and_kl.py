"""Experiment 01: inspect soft targets and teacher–student disagreement.

Run: python examples/01_temperature_and_kl.py
"""

import numpy as np

from distillranklab.distillation import distillation_loss, softmax


def main():
    teacher = np.array([3.0, 2.0, 0.0])
    student = np.array([1.0, 2.0, 0.0])
    print("Teacher logits:", teacher)
    print("Student logits:", student)
    for temperature in (0.5, 1.0, 2.0, 4.0):
        print(f"\nTemperature = {temperature:g}")
        print("Teacher probabilities:", np.round(softmax(teacher, temperature), 4))
        print("Student probabilities:", np.round(softmax(student, temperature), 4))
        print("Distillation loss:", round(distillation_loss(teacher, student, temperature), 6))
    print("\nTry setting student = np.array([3.0, 2.0, 0.0]).")


if __name__ == "__main__":
    main()
