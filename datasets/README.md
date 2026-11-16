# Datasets

Training data is a list of **pairs**: something a person says to you, and how you reply.
The adapter learns your writing style from the replies, so you write every reply yourself.

## Layout

| Path | Tracked in git | Contents |
|---|---|---|
| `sample/pairs.sample.jsonl` | yes | Ten illustrative pairs showing the format. Not real answers. |
| `question_bank/paraphrases.txt` | yes | Reworded versions of the bank questions, as `ID \| question`. |
| `sample/profile.sample.md` | yes | Template for the private profile notes. |
| `raw/profile.md` | **no** | Short notes about you, added to every system prompt. |
| `raw/exclude.txt` | **no** | Sheet IDs to leave out of training, one per line. |
| `question_bank/*.txt` | yes | Questions to answer. File name is `<register>.<kind>.txt`. `{name}` is replaced by your `PERSONA_NAME`. |
| `raw/qa.txt` | **no** | The question sheet: every question with your answer under it. The source of truth. |
| `raw/pairs.jsonl` | **no** | Answered questions as pairs, written by `scripts/qa.py sync`. |
| `processed/{train,valid,test}.jsonl` | **no** | Chat-format files written by `scripts/prepare.py`. |

## Pair format

One JSON object per line:

```json
{"id": "pro-ans-0001", "register": "professional", "kind": "answer",
 "prompt": "What is another-me?", "reply": "It's a text twin of me...",
 "source": "interview"}
```

| Field | Values | Meaning |
|---|---|---|
| `register` | `professional`, `casual` | Who is asking. Selects the system prompt, so one adapter learns both tones. |
| `kind` | `answer` | A normal reply. |
| | `fallback` | A question the twin must not or cannot answer. The reply declines in your own words. |
| | `identity` | "Are you real?" The reply admits it is a model. |
| `prompt`, `reply` | text | The message and your reply. |
| `source` | text | Optional bookkeeping. |

`prepare.py` turns each pair into a chat example:

```json
{"messages": [
  {"role": "system", "content": "You are Alex's text twin. Reply in first person..."},
  {"role": "user", "content": "What is another-me?"},
  {"role": "assistant", "content": "It's a text twin of me..."}
]}
```

## Collecting

```bash
python3 scripts/qa.py init
```

This builds `raw/qa.txt` from the question bank, about 300 questions in six sections.
Write each answer after its `A:` line, by hand or with `scripts/qa.py set <ID>`.
Re-running `init` after editing the bank keeps existing answers.

```bash
python3 scripts/qa.py status
```

```bash
python3 scripts/qa.py sync
```

```bash
python3 scripts/prepare.py
```

## Three things that make a small dataset go further

A few hundred answers teach the writing style well, but each fact appears only once, so the model
does not reliably recall it when the question is worded differently.

1. **Profile notes** (`raw/profile.md`). A short list of facts about you, written once and
   placed in the system prompt during training and chat. The model reads the facts
   instead of having to memorise them. Copy `sample/profile.sample.md` to start.
2. **Reworded questions** (`question_bank/paraphrases.txt`). Each rewording becomes an
   extra training pair with the same answer. Only the training split is multiplied;
   validation and test keep the original wording, and a held-out question never appears
   in training in any wording.
3. **Exclusions** (`raw/exclude.txt`). Answers you gave as yourself that a model cannot
   honour, such as "let me check my calendar", teach the twin to make false promises.
   List their IDs here to keep the answer in the sheet but out of training.

## Guidelines for good pairs

- **Write as you actually write.** Do not tidy your casual replies. Typos, lowercase and
  short answers are the signal.
- **Do not paste model-written text as your reply.** The adapter would learn that style.
- **Keep facts thin in replies.** Facts belong in `knowledge/`. A reply full of dates and
  numbers teaches the adapter to produce plausible dates and numbers.
- **Vary the length.** If every reply is three sentences, every output will be too.
- **Cover fallback and identity.** Aim for roughly 10-15% of pairs, or the twin will
  answer everything confidently.
- **Quantity.** Under about 200 pairs the adapter mostly memorises. Aim for 300
  or more. Add your own questions to the bank files as you think of them.
- **Dictated answers.** Speech-to-text captures how you speak, not how you type. That is
  fine for the professional register. For casual pairs, check that the text looks like
  something you would actually type.
