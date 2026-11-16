# Lyari FC · Possession from tactical tracking

More Than A Hack 2027 · Phase 1 · Entry D

Raja Ata Ul Karim, Mian Akbar Jan, Abdur Rehman Durrani, Malik Mobeen and Syed Hamza Ali

## 1. From tracking to possession

**What the algorithm sees.** The only match inputs are the organiser’s tactical EPTS metadata and positions. We map player tracks into two teams, express coordinates in metres and preserve missing observations. Physical checks mark long-frozen slots and implausible one-frame ball jumps, and can reassign a goalkeeper parked in the wrong team block. Short bounded ball gaps receive a straight-line estimate with declining confidence. Long gaps stay missing.

**How a frame is classified.** A shared network encodes each player. Mean and max pooling describe each team without depending on the order of its players. Team and ball features feed a bidirectional GRU, so the model can use play before and after an uncertain handover. It outputs probabilities for home, away and dead at each frame. Auxiliary boundary and displacement targets shape training. The shipped decoder uses the three possession probabilities.

**What counts as possession.** While the ball is live, the last controlling team can retain it through a loose ball, challenge or deflection. In the training labels, an interrupting spell shorter than 38 frames can revert to the surrounding team. A deliberate one-touch pass can establish control faster. When play stops, D lasts until the kick or throw that restarts play. The restarting team takes possession on the first live frame. Outside the two graded periods the output is X.

**Why use a sequence decoder?** A single-frame team switch can split one possession into three. We average the ten probability vectors at each frame and run Viterbi separately for each half. Switching teams costs 20 and entering or leaving D costs 32, so brief changes need evidence across frames. The published second-half kickoff defines the pre-kickoff D interval. [Organisers’ FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md) · [Settings and sources](constants-audit.md)

## 2. What the models learned from

**Thirteen labelled external matches.** All ten submitted models trained on the following matches.

| Source | Matches | Labelled play | Who produced the labels | How we used them |
| --- | --- | --- | --- | --- |
| Sportec / IDSSE, Bundesliga | 5 | 476 min | The provider, shipped with the tracking: ball status and possessing-team flags | Used directly |
| Metrica Sports sample data | 2 (games 1, 2) | 191 min | The provider's synchronised event data | Frame labels derived from the events; long source ball gaps marked as stoppages |
| SoccerTrack v2 | 6 | 563 min | The dataset authors' per-frame ball-status labels | Neutral touches carry the previous team |

That is 20.5 hours of play in total. Every label then passes through the same 38-frame hold rule above. The providers disagree most on short spells. Held out for testing during development, never trained on by the final models: Sportec J03WQQ and J03WR9, Metrica game 3, two SoccerTrack games and 20 SkillCorner A-League matches, whose team attribution comes from the provider's possession field. [Sources, attribution and licences](../DATA.md)

**Simulating a missing ball.** Clean open tracking makes possession easier than partial automatic tracking. The v5bf simulator adds ball gaps, player dropouts, identity errors and correlated position drift before training. Twenty open SkillCorner A-League games supplied ball-loss and group-dropout statistics. Published optical-tracking studies supplied position-error and player-visibility estimates. The shipped weights all use this external version of the simulator. [Damage provenance](damage-v5be.md)

**Ten final models.** Three ordinary models differ by seed. Three add relative player–ball motion and proximity features. Three sample additional windows around short possessions. The tenth doubles the standard hidden width from 128 to 256. Each trained for 30 epochs on all 13 matches. We tested on matches excluded from each training fold during development. The final fit used every practice game. All ten weights contribute equally. [Recipe and seeds](../lyari/jobs/recipe.json)

**Other assets.** Earlier experiments used nine SkillCorner 2019/20 matches with provider team attribution. We also fine-tuned a ball detector on the measured ball centres supplied for four ISSIA-CNR camera views. We tried broadcast clock alignment and audio cues. The submitted system takes only tactical tracking as input.

## 3. Evidence, limits and reproduction

**Whole-match comparison.** We scored each practice match using models trained on the other twelve. The nine-member Entry C scored 0.7826, 0.7449 and 0.7545 spellF1 on v5bfc, v7c and v6c respectively. Adding the wide member changed those means by +0.0001, +0.0023 and +0.0015, all within the uncertainty of the comparison. Three extra ordinary seeds gave a similar result to the three short-spell samplers. [Decision ledger](ledger.csv)

**What the veto established.** On 20 SkillCorner matches, D scored 0.5052 versus 0.5068 for C plus an extra plain seed. The paired delta of −0.0015 stayed above the registered rejection threshold of −0.003. This let us keep D without claiming that width improves performance. A separate practice analysis found that a single five-minute window often differs appreciably from the same match’s full score. [Selection protocol](prereg-2026-11-08.md) · [Veto report](skillcorner-veto-2026-11-10.md) · [Window study and table](window-var-2026-11-10.md)

**Where it fails.** Long ball gaps make restarts and brief opposing touches hard to time. The 38-frame training rule can suppress genuine one-touch turnovers, although the decoder can still produce shorter spells. A goalkeeper assigned to the wrong team for less than a minute can escape correction. When most player tracks disappear, the system tends to predict dead ball. Simulated tracking damage is still easier than real tracking loss.

**With more time.** We would prioritise external labels that distinguish control from deflection, learn the possession convention from more providers instead of a fixed hold, and add a video ball detector to fill the longest ball gaps.

**How we chose the training hold.** On 16 October, before the data release, we set 1.5 seconds to match the possession rate and mean spell length in the published five minutes, which the challenge text describes as the reference for the labelling convention. On 23 October the organisers clarified that this window is not a tuning set. On 28 October we re-examined the value on open Sportec data only. A first analysis of seven matches favoured 3.0–3.8 seconds. A correction for the shorter time a brief spell has to record an event, added after that result while 1.5 seconds was in use, gave 1.5 seconds on the five training matches (bootstrap 1.0–2.0 seconds). We kept 38 frames. [Open-data hold analysis](min-hold-sportec.md)

**How we chose other settings.** The decoder costs were first picked on practice copies whose simulated tracking loss was fitted to label-free aggregates of the competition tracking. Re-run on the external v5bfc copy, 20/32 was not the best cell, but the best alternative differed by less than the paired uncertainty, so we kept it. The v6c and v7c copies used in model selection also drew on statistics from the unlabelled competition tracking, which the rules allow as learning without human labels; v7c also kept stoppage-timing and ball-motion assumptions informed by PFF tracking. While building v7c we compared baseline scores on it with the organisers' published full-match baseline scores; adjusting its stoppage settings did not close the gap, and v7c was locked using label-free tracking checks only. All ten final fits used the external v5bf simulator. We enabled tracking cleanup after an automatic comparison with Smart Tagging, which the FAQ invites teams to use, and checked it on external practice data. Earlier experiments consulted the reserved open test pool, so its scores are not a fresh estimate. [Damage provenance](damage-v5be.md)

**Reproduce.** With the pinned packages and the licensed tracking folder supplied via MTAH_DATA_DIR, run `python -m lyari.final --check`. This validates the ten checkpoints, produces the possession CSV twice and requires identical bytes. The README gives the full hash and the command for validating the exact upload. [Installation and expected hash](../README.md) · [Training instructions](training.md)
