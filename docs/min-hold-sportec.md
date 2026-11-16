# How long is an interruption?

On 16 October, before the organisers’ 23 October clarification that the public window is not a tuning set, we chose a 38-frame hold to match the possession rate and mean spell length in the published five minutes. On 28 October we re-examined that value using open Sportec tracking flags and synchronised DFL events. A first analysis of seven matches, without the correction below, favoured 3.0–3.8 seconds. The correction was added after that result, while 38 frames was in use. A controlled action includes a pass, shot, ball claim or duel in which control is retained.

A short spell also offers less time for any event to be recorded. A correction estimated that opportunity using the opening part of longer controlled possessions. On the five training matches, the corrected controlled share remained at least one half from the 1.5-second bin onward. Leaving out any one of the five yielded 1.5 seconds; the match bootstrap 5th–95th percentiles were 1.0–2.0 seconds. Short bins were noisy. The evidence supports the scale, with limited precision.

The [event alignment](../lyari/analysis/min_hold_sportec.py), [opportunity correction](../lyari/analysis/min_hold_corrected.py) and [synthetic tests](../lyari/analysis/test_min_hold_sportec.py) are included for inspection. The rule is implemented in [labels.py](../lyari/dataset/labels.py).
