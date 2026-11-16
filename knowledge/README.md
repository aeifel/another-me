# Knowledge base

Documents the twin can look facts up in. The adapter supplies the writing style; these
documents supply the content.

**Status:** the two tools below are implemented and tested (`another_me/knowledge/`,
`another_me/agent.py`) but not yet connected to the demo, which currently gets its facts
from the short profile notes in the prompt. Connecting them is on the roadmap.

| Path | Tracked in git | Contents |
|---|---|---|
| `sample/` | yes | A real document about this project and a template to copy. |
| `private/` | **no** | Your actual documents. Create this folder yourself. |

Both folders are read at run time (`[knowledge] roots` in `configs/default.toml`).
Files must be `.md` or `.txt`. `README.md` files are skipped.

## Two modes, one at a time

| Mode | Tools the model gets | Good when | Weak when |
|---|---|---|---|
| `files` | `list_files`, `read_file` | Few documents with clear titles; you want to show agentic lookup | Many or long documents: whole files fill the context |
| `search` | `search_knowledge` | Many documents; the answer is one paragraph somewhere | The query words differ from the document's words (v0 is keyword-based) |

`another_me.knowledge.make_toolbox(mode, cfg)` returns the tool set for one mode.

There is no vector database and no server. `search` builds a BM25 index in memory at
start-up, which is instant for a personal knowledge base. Embeddings and a reranker are
planned as a later step.

## Writing documents that retrieve well

- **One topic per file**, with a title on the first line. `files` mode chooses by title.
- **Use headings.** `search` splits on headings, and each chunk carries its heading.
- **Write full sentences with the subject named.** "I joined Acme in 2022" retrieves;
  "Joined 2022" under a distant heading does not.
- **State facts once.** Contradicting copies produce contradicting answers.
- **Leave out anything you would not say to a stranger**, even though `private/` is not in git.
  The model can repeat whatever it reads.
