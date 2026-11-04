# Environment for researching whether vision-language robot policies follow their instructions

> [!IMPORTANT]
> We are using this environment to write a paper for ICML 2027. It was built in [AI Research School](https://www.skool.com/become-ai-researcher-2669). The code is under the MIT licence, so you can use the code and ideas in your own work.
>
> **This is a setup, and it has no results yet.** Bug checks pass, but no baseline has been trained through the full test on the current vision encoder. Start with the kill test below.

## Ask which part of a robot policy makes it ignore the instruction

A vision-language policy looks at a camera image, reads a sentence and outputs robot actions.

Recent papers find that these policies often ignore the sentence. [LangGap](https://arxiv.org/abs/2603.00592) reports π0.5 at 93.8% on its original tasks and 29.3% when asked for a different object in the same scene. [Encoded but Not in Control](https://arxiv.org/abs/2610.06235) finds that policies can hold the requested target in their internal state and still go to the original object.

A policy has three main parts that could cause this. The **fusion module** combines image and sentence. The **action head** turns the result into actions. The **training data** decides which sentences appear with which scenes.

LangGap states that it doesn't separate these parts.

**The question:** with equal weights and equal demonstrations, which part decides whether a policy follows its instruction? Does the module that answers questions about an image best also act best?

---

## Use one test for every design

| Part | What is here |
|---|---|
| **World** | A batched 2D tabletop in numpy. Several tasks share one scene: go to an object, go to a position, put one object on another, touch two objects in order. |
| **Fusion baselines** | Late fusion, FiLM, token insertion (π0), cross-attention (CASA), gated cross-attention (Flamingo), TokenLearner (RT-1), Perceiver Resampler, language attention (CLIPort). |
| **Action heads** | Regression, action chunks (ACT), flow matching (π0, SmolVLA), and an answer head for vision-language questions. |
| **Training data** | One task per layout (like LIBERO), several tasks per layout, a new layout per demonstration, and LangGap's small and diluted settings. |
| **Controls** | `blind` gets no sentence, `shuffle` gets another row's sentence, `oracle` gets the true goal position. |
| **Scores** | 20+ per-item scores with no combined number: trained tasks, a different target in the same scene, new layouts, unseen colour-shape pairs, unseen word order, more objects, a target that moves, offline error, and whether a linear probe can decode the target. |
| **Rules** | `claims.py` holds published results as tests, `audit.py` decides which scores can judge, and `beat.py` is the win rule, committed before any new design runs. |

Hidden test items sit behind `--role final`. A new design is one class in `models.py`.

---

## Run the checks, then the kill test

```bash
pip install -r requirements.txt
python3 check.py
python3 jobs.py kill --tier 1m --device cuda
python3 claims.py --tier 1m
python3 compare.py --tier 1m --regime free --ref token+mse
```

`check.py` takes about 10 seconds. The kill test's predictions and failure numbers are in [EXPERIMENTS.md](EXPERIMENTS.md), written before any baseline ran. To run the whole method with Claude Code, use the [auto-research skill](.claude/skills/auto-research/SKILL.md).

---

## Where it stops working

> [!WARNING]
> - **Budgets are uncalibrated.** The tiers in `env.py` were set before the shared vision encoder existed, so expect to change them after the kill test.
> - **A published result doesn't match yet.** On an earlier vision stem, asked for a different object, the policy went to the original about 0% of the time, while papers report 86-98%. The reason is untested.
> - **This is a small 2D world with small ports of each module.** They follow the papers' text and weren't compared with authors' code. A lead here needs a test on [LangGap's benchmark](https://github.com/YC11Hou/langgap) or LIBERO, and that tier isn't built.
> - **The vision encoder is frozen for every policy,** like SmolVLA's. No arm fine-tunes it, and it reads shape correctly 87% of the time.

The full list of open issues is in [reviews/pass-1.md](reviews/pass-1.md).

---

## Is this new?

| Work | Same-scene test | Compares fusion modules | Compares action heads | Compares training data | Closed-loop control |
|---|---|---|---|---|---|
| [LangGap](https://arxiv.org/abs/2603.00592) | yes | no | no | yes | yes |
| [Encoded but Not in Control](https://arxiv.org/abs/2610.06235) | target replacement | no | no | no | yes |
| [CASA](https://arxiv.org/abs/2512.19535) | no | yes | no | no | no |
| **This repo** | yes | yes | yes | yes | yes |

This table comes from three literature searches and reading these papers' summaries and excerpts, which doesn't settle that the question is new. Read them in full before relying on it.
