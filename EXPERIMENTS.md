# Working log

## Question

When a vision-language robot policy ignores its instruction, which part is responsible: the fusion module, the action head, or the training data? Does the module that answers vision-language questions best also act best?

## Closest work

| Paper | What it does | What it leaves open |
|---|---|---|
| [LangGap](https://arxiv.org/abs/2603.00592) | Adds tasks that share scenes to LIBERO; pi0.5 falls from 93.8% to 21.4%, and training on them raises one case from 0% to 28.0%. | States it doesn't compare fusion modules or action heads, and can't attribute gaps to components. |
| [Encoded but Not in Control](https://arxiv.org/abs/2610.06235) | Target replacement goes to the original object 86.5-98.3% of the time, while probes decode the target. | Names same-scene competing behaviours in evaluation and training as future work; our reading of it found no tested fix. |
| [CASA](https://arxiv.org/abs/2512.19535) | Compares cross-attention and token insertion on VLM benchmarks. | No robot control. |
| [2609.23048](https://arxiv.org/abs/2609.23048) | Offline checks look fine while closed-loop success is 0/72 vs 40/72. | Not about instruction following. |

**One-sentence difference:** swap one component at a time at equal weights and on the same demonstrations, under a test where the instruction selects among k behaviours in one scene, and measure whether answering skill predicts acting skill.

Three searches don't settle that this is new. Read these papers in full before relying on it.

## Boring explanations to rule out

1. Demonstrations per instruction, not the module, set the score.
2. Visual saliency: some objects are easier to see, so "following" is really seeing.
3. A module wins because it runs more compute at equal weights. `speed.py` and rule 2 of `beat.py` cover this.
4. `langattn` wins because the world rewards pointing at one location. In a debug run it learned pointing well before `token` did, so read its lead against this first.

## Kill test: written before any baseline run

`python3 jobs.py kill --tier 1m --device cuda`. The idea is dropped if, after calibrating budgets:

| Prediction | Failure number |
|---|---|
| `coupled`: nominal.object minus swap.object is at least .4 for `token+mse`. | under .4 |
| `free` or `random`: some module has new.object at least .2 above `blind+mse`. | under .2 |
| Modules differ: the spread on swap.object across `token`, `film`, `cross`, `langattn` exceeds 2 standard errors. | no spread |
| `blind+mse` new.object stays at or below .3, and `oracle+mse` is at or above .7. | either fails |

Every outcome of the answering comparison is reportable: rank agreement between `answer.*` and `new.*` across modules is the result either way.

Designs tried so far: 0.

## Debug log

These are short runs on a heavily loaded laptop, one training run per setting. They set up the test and are not results. Rows 1-3 used the first from-scratch vision stem, which is now replaced.

| What ran | Found | Change |
|---|---|---|
| `token+mse`, `coupled`, 60k weights, 1500 steps | nominal.object 1.0, swap.object 0, off.swap .94, swap.orig 0 | Open: the papers report 86-98% to the original. Untested explanation: each layout's unique sentence acts as a lookup key, so an unseen sentence sends the action elsewhere. |
| `oracle+mse`, same budget | swap.object .8 | The reach tasks are solvable without vision or language. |
| `token`, `film`, `langattn` on `random`, 1500 steps; `token` 120k weights, 3500 steps | swap and new near 0 | From-scratch pixels can't learn naming objects at this budget. |
| Pointing at a named object in one frame, about 35 s each | from scratch `token` .03 hits after 1000 steps; `langattn` .50 after 2000; with a frozen encoder `token` .35 after 600 | Added the shared frozen encoder in `vision.py`. |
| Encoder pretraining | colour 98.5%, shape 87.3%, gripper 98.6%, lit 98.1% after 2,889 of 4,000 steps | Its run was cut by a time limit, so its learning rate didn't finish decaying. `python3 vision.py` with no limit does the whole schedule; bump TEST_VERSION if the file changes. |
| `seq` with a one-frame grip press | oracle 0 | Regression can't learn a one-frame spike, so `seq` became touch-to-light. |
| Expert on place and seq | .6 and .95 | Fixed: it now stops after placing, and counts b only when touched after a. All families are 1.0 in `check.py`. |

**Not yet run:** any trained policy on the frozen encoder through the closed loop. A 2000-step run was stopped to free the laptop.
