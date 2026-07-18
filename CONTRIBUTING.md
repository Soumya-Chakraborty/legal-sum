# Contributing to LegalSum

Thank you for your interest in contributing. This document covers how to set up a development environment, code standards, and the pull-request process.

---

## Development Setup

```bash
git clone https://github.com/Soumya-Chakraborty/legal-sum.git
cd legal-sum

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision h5py numpy scipy tabulate

# Verify the codebase parses cleanly
python -c "import ast; ast.parse(open('main.py').read()); print('main.py OK')"
python -c "import ast; ast.parse(open('models.py').read()); print('models.py OK')"
python -c "import ast; ast.parse(open('rewards.py').read()); print('rewards.py OK')"
```

---

## Repository Map

| File | Responsibility |
|---|---|
| `main.py` | Training pipeline, PPO loop, evaluation, CLI |
| `models.py` | All neural network architectures |
| `rewards.py` | All reward functions and attribution |
| `vsum_tools.py` | Knapsack solver, F1 evaluation |
| `utils.py` | Logger, checkpointing, JSON I/O |
| `run_all_experiments.py` | 5-fold cross-validation runner |
| `knapsack.py` | 0/1 knapsack implementation |
| `create_split.py` | Generate train/test split JSON |
| `demo/legal_dataset.py` | Courtroom dataset loader |
| `demo/plotting_utils.py` | Training curve visualisation |

---

## Code Standards

### Style

- Python 3.8+ compatible
- PEP 8 formatting (4-space indent, 100-char line limit)
- Type hints encouraged but not required
- Docstrings required for all public functions and classes

### Docstring Format

```python
def my_function(arg1, arg2):
    """
    One-line summary.

    Longer description if needed.

    Args:
        arg1 (type): Description.
        arg2 (type): Description.

    Returns:
        type: Description.
    """
```

### Novel Component Tagging

Any new architectural or algorithmic contribution must be tagged with a `[NOVEL-X]` comment at the top of its class/function and listed in the module-level docstring novelty map. This makes ablation studies and paper citations straightforward.

```python
# [NOVEL-X] Short name (ClassName or function, line ~N)
#     One-sentence description of what makes this novel vs prior work.
class MyNewComponent(nn.Module):
    ...
```

---

## Where to Add New Components

### New Model Architecture

1. Add the class to `models.py`
2. Register it in `build_model()` in `main.py` with a new `--model-type` option
3. Add the `[NOVEL-X]` tag and update the module docstring novelty map

### New Reward Component

1. Add a private helper function `_my_component(...)` to `rewards.py`
2. Integrate it into `compute_reward()` or create a new top-level function
3. Add the `[NOVEL-RX]` tag and update the module docstring
4. If it has a configurable weight, add the corresponding `--my-weight` argument to `parser` in `main.py`

### New Training Feature

1. Add the feature inside `train_one_phase()` or `main()` in `main.py`
2. Add a CLI argument to `parser` with a sensible default
3. Tag with `[NOVEL-TX]` in comments

---

## Testing

There are no mock unit tests. All testing is done on real data:

```bash
# Quick sanity check (6 epochs, SumMe split 0)
python main.py \
  -d datasets/eccv16_dataset_summe_google_pool5.h5 \
  -s datasets/summe_splits.json \
  --split-id 0 -m summe --use-cpu \
  --max-epoch 6 --phase2-epochs 2 --pretrain-epochs 2 \
  --num-episode 2 --ensemble-k 3 --patience 99 \
  --ppo-inner-steps 1 --save-dir log/test_run

# Expected: F1-max >= 30% by epoch 3
```

All pull requests must include a smoke-test run showing the F1-max result.

---

## Pull Request Process

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-improvement
   ```

2. **Implement** your changes following the code standards above.

3. **Run the smoke test** and paste the output in the PR description.

4. **Update documentation**:
   - Add your component to the novelty map in the relevant module docstring
   - Update `LEGAL_SUM_DOCUMENTATION.md` with the new component's section
   - Update the hyperparameter table in `README.md` if you added CLI flags

5. **Submit the PR** with:
   - A clear title describing the contribution
   - The smoke-test F1-max (before/after comparison)
   - Reference to any paper or algorithm the contribution is based on

---

## Reporting Issues

When reporting a bug, include:

- Python and PyTorch version (`python --version`, `python -c "import torch; print(torch.__version__)"`)
- Full command used
- Complete error traceback from `log/*/log_train.txt`
- Dataset and split ID

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
