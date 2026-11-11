# Frozen teacher scoring (Milestone 6B)

Run the offline, dependency-free demo:

```bash
python examples/07_teacher_scoring.py
```

For actual CPU inference, install the optional dependency and run:

```bash
python -m pip install sentence-transformers
python examples/07_teacher_scoring.py --real --cache teacher_scores_real.json
```

The first real run downloads pretrained model weights. Later runs reuse the
JSON cache, keyed by model identifier and exact query/passage texts. The default
cache uses a separate demo model identifier; never treat demo scores as neural
teacher scores. The bundled fixture is illustrative, not MS MARCO data.

Do not commit generated score caches or downloaded models. For larger datasets,
use a cache path outside the repository or add the file to `.gitignore`.
