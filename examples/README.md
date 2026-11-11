# Experiment 01: Temperature and KL divergence

Run `python examples/01_temperature_and_kl.py` after `pip install -e ".[dev]"`.

Each query has three candidate passages. The teacher and student assign one logit
per candidate. Softmax converts these logits into a distribution *within that
query*. Temperature changes how sharp the distribution is. KL divergence
measures disagreement, and `distillation_loss` multiplies KL by `T**2`, a
common convention for temperature-scaled distillation.

## Exercises

1. Replace student scores `[1, 2, 0]` with `[3, 2, 0]`. What happens to loss?
2. Add the same constant to all teacher logits. Do the probabilities change?
3. Swap the first two student scores. Does the teacher's preferred passage
   move to first place in the student ranking?
4. Explain why a low KL loss is not, by itself, proof of good relevance ranking.

This example **does not train** a student; the next experiment will add an
optimization loop. KL is calculated across candidates belonging to the same
query, not across unrelated queries.

## Experiment 02 — Train a linear student

Run `python examples/02_train_linear_student.py` after `pip install -e ".[dev]"`.
The synthetic teacher is frozen. A student with two learned weights scores each
candidate, and full-batch gradient descent minimizes temperature-scaled KL.

**Explore:** Change the temperature, learning rate, or number of steps. Check
whether training loss falls and whether top-ranked passage IDs agree. Then try
removing one feature: the student may no longer have enough information to
imitate the teacher. This is a controlled demonstration, **not** evidence of
real-world generalization; we train and inspect the same synthetic queries.
