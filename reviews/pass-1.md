# Reviewer pass 1: code reading only

This pass read the code as an ICML reviewer would. No baseline has run, so it can't close stage 3 of the method. Open items block any claim.

| # | Issue | State |
|---|---|---|
| 1 | No trained policy has run through the closed loop on the frozen encoder. | Open. Run `jobs.py kill`. |
| 2 | Tier budgets in `env.py` were set before the encoder existed, so they are uncalibrated. | Open. |
| 3 | swap.orig was 0 where papers report 86-98%, seen only on the old stem. | Open. Needs a targeted run. |
| 4 | Encoder shape accuracy is 87%, and colour-shape names depend on it, which caps same-colour targets. | Open. Rerun `vision.py` in full. |
| 5 | Baselines are small ports, not compared with authors' code. | Stated in `models.py`. |
| 6 | A frozen encoder matches SmolVLA; pi0 fine-tunes its own. No arm fine-tunes here. | Stated. Limits what fusion comparisons say about end-to-end training. |
| 7 | A 2D world with four task families is no replacement for LIBERO. | Stated in the README. A real-benchmark tier isn't built. |
| 8 | `langattn` has a pointing shortcut that may suit this world. | Listed as a boring explanation. |
| 9 | `claims.py`, `audit.py` and `beat.py` have only run with no rows, and thresholds are unexercised. | Open. |
