---
name: auto-research
description: Run the environment-first research method in this repo. It tests the test first, writes the win rule and predictions before results, runs design rounds, scales up, and limits every claim to the evidence.
---

# Environment-first auto research

Build a test that can reject ideas, prove that it can, then search for designs under rules written before the results. This produces evidence and a reproducible repo. It doesn't make a paper acceptable by itself.

**Debug in runs of about 7 seconds first.** Use the `tiny` tier and `check.py` until they have nothing left to find, and rerun them after any code change. A long run that finds a bug a short test could have found is a mistake, so log it in `EXPERIMENTS.md`.

Move forward only when a stage's check passes.

## 1. Try to kill the idea cheaply

Run at least 3 searches and read the closest paper in full. Write the boring explanation of the expected effect. Write the one test whose failure ends the idea, with its failure number, then run it. The current kill test is in `EXPERIMENTS.md`.

**Check:** the closest work, the one-sentence difference and a passed kill test are written down.

## 2. Keep one environment for every idea

`env.py` has tiers, baselines, per-item scores, hidden items behind `--role final`, resumable rows and tagged test runs. A new design is one class in `models.py`. Tune every baseline at least as hard as any new design, because a weak baseline makes a new design look good.

**Check:** a smoke test runs through the real job script with every score present.

## 3. Test the test with published modules, before any new design

| Check | Question | File |
|---|---|---|
| Bugs | Are held tasks in training? Does every weight get a gradient? Does the sentence change the output? | `check.py` |
| Repeat | Do two runs of one setting give identical scores? | `check.py` |
| Published results | Are they reproduced? Find the reason for each mismatch with a targeted run. | `claims.py` |
| Controls | Do `blind` and `shuffle` get no credit? Does `oracle` score high? | `audit.py` |
| Scores | Does each score move, separate modules, and agree across worlds and tiers? Set cut-offs before reading the table. | `audit.py` |
| Real benchmark | Does a lead hold on LIBERO or LangGap's tasks? | not built |

After any fix to a test, rerun every baseline and keep old rows in a separate file.

Then review everything as an ICML reviewer: bugs, hacks, tests too simple, untuned baselines, unfair comparisons, missing evaluations. Fix each issue, rerun the short tests, and review again from the start. Write each pass in `reviews/`.

**Check:** a full pass finds no new issue, and you can publish one paragraph on what the test can judge and what it can't. No new design runs before this.

## 4. Keep the win rule fixed

`beat.py` is committed before any design runs. It caps weights and compute. Count every design tried. A close result stays `no win`.

## 5. Run design rounds

1. Write the prediction, with numbers and what would drop the idea, then commit it before running.
2. Change one idea, derived from the last result.
3. Iterate at the small tier that predicts the next one, then check at that tier. Run leaders at two or more learning rates.
4. Log the result beside the prediction, including failures.
5. Explain before redesigning: switch each part off at test time and read the scores.

Before crediting a new part, add controls built only from the published parts beside it, and measure speed one run at a time.

## 6. Scale up before believing a lead

Run baselines and leaders at the largest tier you can finish. Measure rank agreement between tiers, and report which leads grew and which faded. The largest finished tier is the main result.

## 7. Score hidden items once

Use `--role final` once at the end. Say what was chosen after reading them.

## 8. Attack your own claims

| Claim | Check |
|---|---|
| "Beats X" | Name the size, training, scores won and scores lost, and say if X is your port. |
| "Best" | Is z above 2? If not, write "ties". |
| "New" | Say which parts are published. Three searches don't settle it. |
| "Adding A helps" | Does a run compare with and without A at equal size? |

List every limit, such as training runs per setting, untested sizes, compute, unread papers and missing controls. Each one is answered by a run or stated in the write-up.

## Judge ideas by the ICML reviewer form

Is it novel, relevant to machine learning, correct, and reproducible? Keep every script, row and failed prediction in the repo.
