# another-me

A text twin of yourself: a small open model that replies the way you write.

- **Writing style** comes from post-training: a LoRA adapter on Qwen3 4B, trained on
  question and answer pairs you write yourself.
- **Facts** come from short private notes placed in the prompt.
- **Honesty**: the twin is trained to say it is a model and to decline what it cannot do.
- **Before and after**: a demo page puts the tuned model and the untouched base model
  side by side.

Everything personal (your name, answers, notes, tokens, trained adapters) stays in
gitignored files or in your own Modal account. The repository contains none of it.

```
 datasets/question_bank ──qa.py init──▶ datasets/raw/qa.txt ──qa.py sync──▶ pairs.jsonl
                                         (you answer here)                      │
                                                                         prepare.py
              profile notes + reworded questions ───────────────────────────────┤
                                                                                ▼
                                                   datasets/processed/{train,valid,test}.jsonl
                                                                                │
                                                     train.py --action train    ▼
                        Modal volume:  base model  +  adapters/<run>   (GPU, PyTorch + PEFT)
                                                                                │
                                                    train.py --action deploy    ▼
   browser ◀──▶ scripts/serve.py (localhost) ◀──▶ Twin service on Modal (GPU, sleeps when idle)
```

## What is included

| Part | Where |
|---|---|
| About 300 questions in four kinds: work, casual, things to decline, "are you real?" | `datasets/question_bank/` |
| A question sheet you fill in, and tools to turn it into training files | `scripts/qa.py`, `scripts/prepare.py` |
| Reworded versions of every question, to multiply the training examples | `datasets/question_bank/paraphrases.txt` |
| Training, held-out comparison and a model service on Modal | `another_me/train_modal.py`, `scripts/train.py` |
| A demo page with chat and side-by-side comparison, and its local server | `web/`, `scripts/serve.py` |
| A cost estimator | `scripts/budget.py` |
| Knowledge tools (browse files, or search) and a tool loop, tested but not yet used by the demo | `another_me/knowledge/`, `another_me/agent.py` |
| Tests, including checks that nothing private is tracked by git | `tests/` |

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and a [Modal](https://modal.com) account.

```bash
uv sync --extra modal
```

```bash
cp .env.example .env
```

Fill in `.env`. It is gitignored.

| Key | Needed | Notes |
|---|---|---|
| `PERSONA_NAME` | yes | The name the twin answers as. Defaults to the placeholder `Alex`. |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | yes | Create at modal.com/settings/tokens |
| `HF_TOKEN` | no | The default base model is public. Needed only for gated models. |
| `ANOTHER_ME_GPU` | no | Default `L4,A10`: tried in order, so a busy GPU type does not hold you up. |

```bash
python3 -m unittest discover -s tests
```

## Build your twin

**1. Answer the questions.** This creates `datasets/raw/qa.txt`. Write each answer after
its `A:` line, in your own words. See [datasets/README.md](datasets/README.md) for what
makes a good answer.

```bash
python3 scripts/qa.py init
```

```bash
python3 scripts/qa.py status
```

**2. Write your profile notes.** A short list of facts about you that goes into every prompt.

```bash
cp datasets/sample/profile.sample.md datasets/raw/profile.md
```

**3. Prepare the training files.** This splits off validation and test questions and adds
the reworded questions to the training split only.

```bash
python3 scripts/qa.py sync
```

```bash
python3 scripts/prepare.py
```

**4. Check Modal works, without a GPU.** The second command puts the base model
(7.5 GB) on a Modal volume.

```bash
uv run scripts/train.py --action smoke
```

```bash
uv run scripts/train.py --action download
```

**5. Dry run, then train.** The dry run takes three steps on a GPU and costs a few cents.
It catches problems such as running out of GPU memory before the real run.

```bash
uv run scripts/train.py --action train --run-name dryrun --max-steps 3 --limit 24
```

```bash
uv run scripts/train.py --action train --run-name v1
```

Validation loss is measured during training and the best adapter is the one kept. The
result is saved to `runs/v1.modal.json`.

**6. Compare with the base model** on the test questions, which were never trained on in
any wording.

```bash
uv run scripts/train.py --action compare --run-name v1
```

## Run the demo

**1. Point the service at your adapter and publish it.** Set `adapter` under `[serve]` in
`configs/default.toml` to your run name, then:

```bash
uv run scripts/train.py --action deploy
```

**2. Start the local server.**

```bash
uv run scripts/serve.py
```

**3. Open http://127.0.0.1:8000.** The page has an explainer, a chat panel, and a section
that sends one question to the tuned and the base model side by side.

To use a newer training run, repeat step 1 and restart the server. Nothing switches
automatically.

## How serving works

| Question | Answer |
|---|---|
| Where is the model? | On a Modal volume named `another-me`: the base model under `hf/`, each adapter under `adapters/<run-name>/`. Nothing is stored locally except run logs in `runs/`. |
| What runs where? | The browser talks only to `scripts/serve.py` on localhost. The server builds the prompt (name, profile notes, tone, the conversation) and calls the `Twin` service on Modal with your token. The token never reaches the browser. |
| What is the `Twin` service? | One GPU container holding the base model with the adapter attached. It loads once and then answers requests. |
| How does a new training run go live? | It does not by itself. The service loads the adapter named in `[serve] adapter` at deploy time. Change it and deploy again. |
| How does the comparison work? | Sequentially on the same GPU and the same loaded model: first the reply with the adapter, then the reply with the adapter switched off, using the same random seed. One copy of the base weights serves both, so a comparison takes about twice as long as a chat reply. |
| What does the base model see? | The same prompt as the tuned model, including the profile notes. The comparison shows the difference in writing style, not a difference in what the models were told. |
| Does chat keep history? | Only in the browser tab. The server stores nothing. |

The server is meant for one person on one machine: it listens on localhost only and sends
one model request at a time.

## Cold starts and latency

The GPU container shuts down after `[serve] scaledown_seconds` (5 minutes) without
requests, so an idle GPU is not billed. The next request has to start a container and load
the model first.

How to see what is happening:

| Where | What it shows |
|---|---|
| Top bar of the page | `model asleep`, `waking the model… 12s`, or `model ready` |
| Under a comparison | How long both replies took, and how long the wake-up took if there was one |
| Terminal running `serve.py` | One line per request: generation time, end-to-end time, whether the model was asleep |
| `http://127.0.0.1:8000/api/status` | State, last cold-start time, last reply timing, seconds until the model sleeps |
| `modal container list` | Whether a GPU container is running right now |
| `modal app logs another-me` | Output from inside the container |

The page wakes the model as soon as it opens (`[serve] warm_on_load`). For a live demo,
open the page a couple of minutes beforehand.

## Reference numbers

Measured with the default settings on an A10G GPU: base model `Qwen/Qwen3-4B-Instruct-2507`
in 16-bit, LoRA rank 16 on 0.81% of the weights, 492 training examples of about 450 tokens
(182 answers plus 310 reworded questions, with about 270 words of profile notes).

| Training | |
|---|---|
| Three epochs | 488 s |
| Throughput | 1,481 tokens/s |
| Peak GPU memory | 13.7 GiB |
| Validation loss | 5.71 before, 2.19 after the first epoch (kept), 3.03 and 3.83 after the second and third |

| Serving | |
|---|---|
| Cold start, request to ready | 22 to 25 s with a GPU available, up to 86 s when one had to be found |
| Chat reply, model awake | under 1 s for a short reply, 4 to 9 s for a paragraph |
| Comparison, both replies | 4 to 19 s |

| Held-out test questions (17) | Real answers | Tuned | Base model |
|---|---|---|---|
| Average reply length, all | 24 words | 9 words | 170 words |
| Work questions | 47 words | 15 words | 222 words |
| Casual questions | 3 words | 4 words | 124 words |

What to expect from a dataset of this size:

- **Style transfers well.** The base model writes long, enthusiastic paragraphs with emoji.
  The tuned model answers plainly and briefly, as the real answers do.
- **Facts need the notes.** Without profile notes the tuned model answers questions about
  the person vaguely or wrongly. With them it answers correctly.
- **More epochs memorise.** Validation loss was lowest within the first epoch. Lowering
  the learning rate and checking more often did not change that.
- **Limits.** Work answers come out shorter than the real ones, and the model sometimes
  answers something it should decline. More answers help more than more training.

## Cost

Modal bills per second while a container runs, and the Starter plan includes monthly free
credit. Per-second rates are in `another_me/budget.py`.

| Activity | Approximate cost |
|---|---|
| Dry run | a few cents |
| Training run above | about $0.20 |
| One wake-up of the demo, with its 5 minute idle window | about $0.10 on an L4, $0.13 on an A10 |
| Thirty minutes of continuous demo | about $0.60 to $0.70 |
| Asleep | no GPU cost |

```bash
python3 scripts/budget.py --pairs 500
```

```bash
uv run modal billing summary
```

## Design notes

- **Facts in the prompt, writing style in the weights.** A few hundred answers teach style
  well, but each fact appears once and is not reliably recalled. The profile notes carry
  the facts.
- **Reworded questions.** Each bank question has rewordings that reuse its answer, which
  roughly triples the training examples without anyone writing new answers. Held-out
  questions never appear in training in any wording.
- **Best checkpoint wins.** Training keeps the adapter with the lowest validation loss.
- **Micro-batches.** Training uses batches of 1 and accumulates gradients over 8, which
  keeps long examples within a 24 GB GPU.
- **Two tones.** `professional` and `casual` select different system prompts, so one
  adapter learns both and the caller chooses.
- **Exclusions.** Answers a model cannot honour, such as "let me check my calendar", can
  stay in the sheet but be left out of training (`datasets/raw/exclude.txt`).
- **Private by default.** Name, tokens, answers, profile notes, adapters and run logs are
  gitignored. Tests fail if any of them ends up in a tracked file.

## Roadmap

- Connect the knowledge tools to the chat, so the twin can look facts up.
- Evaluation beyond loss: blind style comparison, faithfulness to the notes, refusal rate.
- Streaming replies in the chat panel.
- A hosted demo: public API with rate limits, static page on GitHub Pages.
- Local training and chat on Apple Silicon.
- A spoken version.

## Privacy and honesty

The twin writes in a real person's style, so it must never be mistaken for them. Identity
questions in the training data teach it to say it is a model, and the page states that
every reply is machine-written.
