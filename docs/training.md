# Reproduce the training recipe

The ten checkpoints each completed 30 epochs on the same 13 matches, with no validation holdout in final training. The match identifiers, seeds and exact command-line arguments are recorded in [recipe.json](../lyari/jobs/recipe.json). Use the shipped checkpoints to reproduce the submission bytes. Training on different hardware can produce different weights.

## Prepare external inputs

Start from the official provider downloads listed in [DATA.md](../DATA.md). Under a common input directory use `sportec-idsse/` for the DFL match-information and raw observed position XML files, `metrica-sample-data/data/Sample_Game_{1,2}/` for Metrica’s original sample directories, and `soccertrack-v2/raw/<match_id>/` for each match’s tracker box metadata and data XML. Preserve provider filenames. The [importers](../lyari/dataset/providers.py) select the games in the recipe, normalise pitch direction, convert to 25 Hz and assign player slots. The [label builder](../lyari/dataset/labels.py) and [simulator](../lyari/dataset/damage.py) construct each match’s v5bf training copy.

Run these from the repository root, in the environment from the [README](../README.md):

```sh
python -m lyari.jobs.training prepare --raw /path/to/provider-files --data /path/to/new/v5bf
python -m lyari.jobs.training check --data /path/to/new/v5bf
python -m lyari.jobs.training train --data /path/to/new/v5bf --out /path/to/new/checkpoints
```

For an existing common-format clean practice master, replace `--raw /path/to/provider-files` with `--clean-master /path/to/master`. The preparation command reads the 13 training filenames only. For an already-prepared v5bf copy, start at `check`.

`check` compares each prepared match with the content hash of the v5bf copy the final models were trained on ([v5bf_hashes.json](../lyari/jobs/v5bf_hashes.json); hashes only, no data). Rebuilding all 13 matches with `prepare --clean-master` from the team's clean master reproduced those hashes, and `prepare --raw` from the Sportec and Metrica provider files reproduced the same arrays for those seven matches. The SoccerTrack raw importer has not been re-run against a fresh download.

A CUDA-enabled PyTorch build speeds up retraining. Use the same PyTorch version as [the inference environment](../requirements-final.txt). On CPU, short test trainings with this code and the same seed gave identical weights across repeat runs and matched the code that trained the final members; GPU training is generally not bit-reproducible, so a retrain gives similar, not identical, checkpoints. Training writes separate member directories with resumable checkpoints, and checks the final epoch and match membership. All three plain seeds, three control-physics seeds, three short-window-sampler seeds and the one wider member use the same loss and optimiser schedule.
