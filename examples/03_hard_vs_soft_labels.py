"""Compare supervised, distillation, and hybrid linear rankers on held-out data."""
from distillranklab.comparison import evaluate, make_dataset, train_student


def main() -> None:
    train = make_dataset(300, seed=11, teacher_error_rate=0.20)
    test = make_dataset(1000, seed=12, teacher_error_rate=0.20)
    print('Held-out evaluation (teacher error rate: 20%; student has fewer features)')
    print(f"{'Method':<10} {'Accuracy':>10} {'Teacher agreement':>19}")
    for mode in ('hard', 'soft', 'hybrid'):
        weights = train_student(train, mode=mode)
        metrics = evaluate(test, weights)
        print(f"{mode:<10} {metrics['accuracy']:>10.3f} {metrics['teacher_agreement']:>19.3f}")
    print('All methods share identical train/test splits and optimization budgets.')
    print('Teacher agreement measures imitation, not ground-truth relevance.')


if __name__ == '__main__':
    main()
