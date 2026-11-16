# External damage used for final training

The release implements **v5bf**, the external-data revision of the simulator. It converts open clean tracks into the player slots, missing ball, interruptions and position noise the learner sees. Each practice game has a reproducible random seed. Separate random streams handle ball loss, position drift, player loss and identity switches.

| Setting used in v5bf | Value | Basis |
| --- | --- | --- |
| Ball detected; probability of a gap at a stoppage | 59.6%; 0.98 | Twenty SkillCorner broadcast matches |
| Mean ball-loss lead and lag around stoppages | 0.94 s and −1.3 s under the simulator’s sign convention | SkillCorner gap edges |
| Live gap rate and typical duration | 0.8 per live minute; lognormal median 2.8 s | SkillCorner live gap runs |
| Slow-ball and goal-line loss multipliers | 1.9 and 9 | SkillCorner conditional rates |
| Ball smoothing | 0.11 s | SkillCorner tracking accelerations |
| Player and ball drift amplitudes | 0.085 m and 0.07 m | Linke, Link & Lames (2020); Mills et al. (2026) |
| Mean number of detected players | 20.35 | Midpoint of the 89–96% range reported by Crang et al. (2025), multiplied by 22 |
| Group player-loss rate and median duration | 0.041 per live minute; 4.07 s | SkillCorner simultaneous losses |
| Group-loss lognormal dispersion and size quantiles | 0.47; 0.095–0.381 | SkillCorner duration and group-size distributions |
| Group-loss dead multiplier; max objects | 1; 22 | Neutral rate for the full-pitch camera; external tracking limit |
| Camera outage and additional ghost rate | 0 | Neutral assumptions in the absence of a supported external rate |

The [SkillCorner analysis](../lyari/dataset/sc_damage_stats.py) derives the external tracking-loss statistics. Other engineering defaults and draw ranges are listed in [the simulator](../lyari/dataset/damage.py).

**Copies used for model selection.** The team compared ensemble members on v5bfc (the external simulator) and on two earlier practice copies, v6c and v7c. Statistics from unlabelled competition tracking informed both earlier copies. The v7c copy also retained stoppage timing and dead-ball motion assumptions informed by PFF tracking. All ten submitted weights trained on v5bf.

References: Linke, Link & Lames, *PLOS ONE* 15, e0230179 (2020); Mills et al., *Sports Engineering* 29, 15 (2026), using FIFA EPTS measurements; Crang et al. (2025). [Linke et al.](https://doi.org/10.1371/journal.pone.0230179) · [Crang et al.](https://arxiv.org/abs/2508.19477) · [SkillCorner](https://github.com/SkillCorner/opendata)
