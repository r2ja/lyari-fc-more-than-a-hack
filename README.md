# Lyari FC · Frame-level ball possession

Given the organisers’ tactical tracking, Entry D labels each video frame as home possession, away possession, dead ball or outside the graded periods. It uses ten equally weighted models and a half-by-half Viterbi decoder. Inference runs on CPU using the shipped weights.

## Reproduce the submission

**Supported platforms.** Python 3.12 or 3.13 on Linux (x86-64 or aarch64), Windows (x86-64) or macOS 14+ on Apple silicon. The pinned PyTorch 2.14.1 has no wheel for Intel Macs or Python 3.11 and older; on those, use a Linux machine or container. A run needs about 3.5 GB of RAM and takes about 2.5 minutes per pass on a recent laptop CPU, so `--check` takes about 5 minutes.

**Match data.** `MTAH_DATA_DIR` must point at the root of the organisers' bundle, the folder that contains `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt` (layout in `games.py`). Only those two files are read. Alternatively, unpack the bundle into `data/` inside this checkout (gitignored) and leave `MTAH_DATA_DIR` unset.

Run every command from the repository root. `requirements-final.txt` pins the three direct dependencies; `constraints.txt` pins everything they pull in.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-final.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
$env:MTAH_DATA_DIR = "C:\path\to\mtah2027"
.\.venv\Scripts\python.exe -m lyari.final --check
.\.venv\Scripts\python.exe evaluation/validate_submission.py "evaluation/results/Lyari FC.csv"
```

On macOS or Linux (on Debian or Ubuntu, `python3.12 -m venv` needs the `python3.12-venv` package):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-final.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu
export MTAH_DATA_DIR="/path/to/mtah2027"
.venv/bin/python -m lyari.final --check
.venv/bin/python evaluation/validate_submission.py "evaluation/results/Lyari FC.csv"
```

Each run verifies all ten model digests. The `--check` command makes two independent passes, validates each output with the organisers' validator and compares their bytes. The CSV is written to `evaluation/results/Lyari FC.csv` (gitignored). Intermediates go to `runs/final` next to this checkout, which must be writable; to choose another folder outside the checkout, add `--work <folder>`. Inference runs on CPU in a single thread with PyTorch's deterministic algorithms. In our tests, environment thread settings, locale and paths containing spaces did not change the output.

Expected SHA-256 of the complete CSV:

```text
a9fd04eb2364a4de6514a3a697107885c551ffd10a1ee0f45572e8eed0bf4527
```

This hash was produced with PyTorch on macOS (Apple silicon) and on Windows (x86-64), and an independent NumPy implementation of the same networks on Linux produced the same file.

## Inspect the method

- [Short technical report](docs/technical-document.md) and its [PDF](docs/Lyari%20FC.pdf)
- [Model manifest](weights/manifest.json), [network](lyari/models/players/model.py) and [decoder](lyari/decoders/viterbi.py)
- [Open training data and attribution](DATA.md), [training instructions](docs/training.md), [training recipes](lyari/jobs/recipe.json) and [damage simulator](lyari/dataset/damage.py)
- [Settings and their sources](docs/constants-audit.md), [selection protocol](docs/prereg-2026-11-08.md), [decision ledger](docs/ledger.csv), [SkillCorner veto](docs/skillcorner-veto-2026-11-10.md) and [window-length analysis](docs/window-var-2026-11-10.md)
- [Submission checklist](docs/submission.md) and [organisers’ submission guide](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/SUBMISSION.md)

The original evaluation code and match specification (`games.py`, `evaluation/`) are from Metrica Sports and Barça Innovation Hub, unchanged, under [their licence](LICENSE-METRICA). The team's code and weights are under the [MIT licence](LICENSE). [Benchmark citation](docs/benchmark.cff).
