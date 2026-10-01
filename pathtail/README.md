# PATHTAIL

Core score constructions for finite forecast sampling. Python 3.12.

`src/pathtail/` contains the algorithms; `examples/` contains runnable entry
points; `tests/` contains unit tests. Original code is MIT licensed.

`ScorePair` stores score `U` and companion `B`. Fix the tolerance and bound
before each update, then call `MixtureEProcess.update_pair` on the same process
to accumulate evidence. `log_e_value` is the natural log of the e-value;
`first_crossing` counts updates from 1 and is `None` before crossing.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install .
```

```bash
python examples/run_binary.py --method monomial
python examples/run_binary.py --method pooled
python examples/run_volume.py
python examples/run_occupancy.py
python examples/run_degree.py --target union
```

Volume and primitive-degree inputs use integers or exact rationals such as
`"1/3"`. Each example demonstrates one construction; binary also updates evidence.

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=pathtail --cov-branch
```
