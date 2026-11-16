"""Stdlib-only checks for everything that does not need a model.  python3 -m unittest -v"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from another_me.agent import parse_tool_calls, run_turn  # noqa: E402
from another_me.budget import estimate, modal_rate_per_sec  # noqa: E402
from another_me.config import load_config  # noqa: E402
from another_me.knowledge import make_toolbox  # noqa: E402
from another_me.knowledge.files import FileTools  # noqa: E402
from another_me.knowledge.search import SearchTool, chunk_markdown  # noqa: E402
from another_me.schema import read_jsonl, to_chat, validate_pair  # noqa: E402

SAMPLE_KB = [ROOT / "knowledge" / "sample"]


class ScriptedBackend:
    """Stands in for a model: returns canned outputs in order."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages, tools=None):
        self.calls.append({"n_messages": len(messages), "tools": bool(tools)})
        return self.outputs.pop(0)


class DataTests(unittest.TestCase):
    def test_sample_pairs_are_valid(self):
        pairs = read_jsonl(ROOT / "datasets" / "sample" / "pairs.sample.jsonl")
        self.assertGreaterEqual(len(pairs), 5)
        for p in pairs:
            self.assertEqual(validate_pair(p), [], p["id"])

    def test_invalid_pair_is_reported(self):
        errors = validate_pair({"id": "x", "register": "formal", "kind": "answer", "prompt": "hi"})
        self.assertTrue(any("reply" in e for e in errors))
        self.assertTrue(any("register" in e for e in errors))

    def test_to_chat_shape(self):
        chat = to_chat({"register": "casual", "prompt": " hi ", "reply": " yo "}, "Alex")
        self.assertEqual([m["role"] for m in chat["messages"]], ["system", "user", "assistant"])
        self.assertEqual(chat["messages"][2]["content"], "yo")

    def test_question_bank_names(self):
        for path in (ROOT / "datasets" / "question_bank").glob("*.txt"):
            if path.name != "paraphrases.txt":
                self.assertEqual(len(path.stem.split(".")), 2, path.name)

    def test_profile_notes_go_into_the_system_prompt(self):
        from another_me.schema import system_prompt
        self.assertNotIn("What you know", system_prompt("Alex", "casual"))
        text = system_prompt("Alex", "casual", profile="- likes tea")
        self.assertIn("What you know about Alex:\n- likes tea", text)
        chat = to_chat({"register": "casual", "prompt": "hi", "reply": "yo"}, "Alex", "- likes tea")
        self.assertIn("likes tea", chat["messages"][0]["content"])

    def test_paraphrases_point_at_real_questions(self):
        from another_me.schema import load_paraphrases
        sys.path.insert(0, str(ROOT / "scripts"))
        import qa
        ids = {e["id"] for e in qa.parse_sheet(qa.render({}))}
        reworded = load_paraphrases(ROOT / "datasets" / "question_bank" / "paraphrases.txt", "Alex")
        self.assertGreater(len(reworded), 100)
        self.assertEqual(sorted(set(reworded) - ids), [])
        for texts in reworded.values():
            for text in texts:
                self.assertNotIn("{", text)


class PrivacyTests(unittest.TestCase):
    """Nothing personal may be tracked by git. Uses your own .env to know what to look for."""

    def tracked_files(self):
        import subprocess
        try:
            out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        return [ROOT / line for line in out.stdout.splitlines() if (ROOT / line).is_file()]

    def test_private_paths_are_not_tracked(self):
        private = ("datasets/raw/", "datasets/processed/", "knowledge/private/", "adapters/", "runs/")
        for path in self.tracked_files():
            rel = path.relative_to(ROOT).as_posix()
            self.assertNotEqual(rel, ".env")
            self.assertFalse(rel.startswith(private), rel)

    def test_persona_name_and_home_path_are_not_tracked(self):
        import re
        env = ROOT / ".env"
        name = ""
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("PERSONA_NAME="):
                    name = line.split("=", 1)[1].strip().strip("\"'")
        needles = [re.compile(re.escape(str(Path.home())))]
        if name and name != "Alex":
            needles.append(re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE))
        for path in self.tracked_files():
            text = path.read_text(errors="replace")
            for needle in needles:
                self.assertIsNone(needle.search(text), f"{path.relative_to(ROOT)} contains {needle.pattern}")


class LeakTests(unittest.TestCase):
    """Private values on this machine must not appear in any tracked file.

    Reads the values from the gitignored files (.env, your answers, your profile notes),
    so the test itself contains nothing private. Skips what does not exist.
    """

    def tracked_text(self):
        import subprocess
        try:
            out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        return {line: (ROOT / line).read_text(errors="replace")
                for line in out.stdout.splitlines() if (ROOT / line).is_file()}

    def assert_absent(self, needles, what):
        files = self.tracked_text()
        for needle in needles:
            for name, text in files.items():
                self.assertNotIn(needle, text, f"{what} found in {name}")

    def test_env_values_are_not_tracked(self):
        env = ROOT / ".env"
        if not env.exists():
            self.skipTest("no .env")
        values = []
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                value = value.strip().strip("\"'")
                if len(value) >= 8 and key.strip() not in ("ANOTHER_ME_GPU",):
                    values.append(value)
        self.assert_absent(values, "a value from .env")

    def test_answers_are_not_tracked(self):
        pairs = ROOT / "datasets" / "raw" / "pairs.jsonl"
        if not pairs.exists():
            self.skipTest("no private answers")
        replies = {p["reply"] for p in read_jsonl(pairs)}
        # Short replies such as "Nope." are ordinary words; check the distinctive ones.
        self.assert_absent([r for r in replies if len(r) >= 40], "one of your answers")

    def test_profile_notes_are_not_tracked(self):
        profile = ROOT / "datasets" / "raw" / "profile.md"
        if not profile.exists():
            self.skipTest("no private profile")
        lines = [line.strip("- ").strip() for line in profile.read_text().splitlines()]
        self.assert_absent([line for line in lines if len(line) >= 40], "a line of your profile notes")


class QuestionSheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import qa
        cls.qa = qa

    def test_bank_is_about_300_with_no_duplicates(self):
        total = 0
        for prefix in self.qa.SECTIONS:
            questions = self.qa.bank_questions(prefix)
            self.assertEqual(len(questions), len(set(questions)), f"duplicate in section {prefix}")
            total += len(questions)
        self.assertGreaterEqual(total, 290)

    def test_sheet_round_trips_answers(self):
        first = self.qa.bank_questions("P")[0]
        casual = self.qa.bank_questions("C")[-1]
        answers = {("P", first): "Line one.\nLine two, with Q: and A: inside.", ("C", casual): "later"}
        entries = self.qa.parse_sheet(self.qa.render(answers))
        got = {(e["prefix"], e["prompt"]): e["reply"] for e in entries if e["reply"]}
        self.assertEqual(got, answers)
        self.assertEqual(entries[0]["id"], "P001")
        self.assertEqual(len({e["id"] for e in entries}), len(entries))

    def test_empty_sheet_has_no_answers(self):
        entries = self.qa.parse_sheet(self.qa.render({}))
        self.assertFalse(any(e["reply"] for e in entries))


class KnowledgeTests(unittest.TestCase):
    def test_files_mode_lists_and_reads(self):
        tools = FileTools(SAMPLE_KB)
        listing = json.loads(tools.call("list_files", {}))
        paths = [row["path"] for row in listing]
        self.assertIn("sample/another-me.md", paths)
        self.assertIn("Qwen3", tools.call("read_file", {"path": "sample/another-me.md"}))

    def test_files_mode_blocks_paths_outside_the_knowledge_base(self):
        tools = FileTools(SAMPLE_KB)
        for bad in ("../../.env", "/etc/passwd", "sample/../../README.md"):
            self.assertTrue(tools.call("read_file", {"path": bad}).startswith("error"), bad)

    def test_search_mode_finds_the_right_passage(self):
        tool = SearchTool(SAMPLE_KB)
        hits = tool.search("which base model is used for the adapter")
        self.assertTrue(hits)
        self.assertIn("Qwen3", hits[0]["text"])
        self.assertEqual(tool.call("search_knowledge", {"query": "zzzz qqqq"}),
                         "No relevant passages found.")

    def test_chunking_keeps_headings(self):
        chunks = chunk_markdown("# A\n\none\n\n## B\n\ntwo\n\nthree", 900)
        self.assertEqual(chunks, [("A", "one"), ("B", "two\n\nthree")])

    def test_only_one_mode_is_attached(self):
        cfg = load_config()
        self.assertIsNone(make_toolbox("none", cfg))
        self.assertEqual({s["function"]["name"] for s in make_toolbox("files", cfg).specs},
                         {"list_files", "read_file"})
        self.assertEqual({s["function"]["name"] for s in make_toolbox("search", cfg).specs},
                         {"search_knowledge"})


class AgentTests(unittest.TestCase):
    def test_parse_tool_calls(self):
        text = 'ok <tool_call>\n{"name": "read_file", "arguments": {"path": "a.md"}}\n</tool_call>'
        self.assertEqual(parse_tool_calls(text), [{"name": "read_file", "arguments": {"path": "a.md"}}])
        self.assertEqual(parse_tool_calls("<tool_call>{not json}</tool_call>"), [])

    def test_tool_loop_runs_tool_then_answers(self):
        backend = ScriptedBackend([
            '<tool_call>{"name": "search_knowledge", "arguments": {"query": "base model"}}</tool_call>',
            "Qwen3 4B.",
        ])
        messages = [{"role": "user", "content": "Which model?"}]
        reply, trace = run_turn(backend, messages, SearchTool(SAMPLE_KB))
        self.assertEqual(reply, "Qwen3 4B.")
        self.assertEqual([t["name"] for t in trace], ["search_knowledge"])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])

    def test_tool_loop_stops_after_max_rounds(self):
        call = '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>'
        backend = ScriptedBackend([call, call, "final"])
        reply, trace = run_turn(backend, [{"role": "user", "content": "x"}],
                                FileTools(SAMPLE_KB), max_rounds=2)
        self.assertEqual(reply, "final")
        self.assertEqual(len(trace), 2)
        self.assertFalse(backend.calls[-1]["tools"])

    def test_no_toolbox_ignores_tool_call_text(self):
        backend = ScriptedBackend(['<tool_call>{"name": "x", "arguments": {}}</tool_call>'])
        reply, trace = run_turn(backend, [{"role": "user", "content": "x"}], None)
        self.assertEqual(trace, [])


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import serve
        cls.serve = serve

    def test_messages_get_a_system_prompt_and_are_trimmed(self):
        register, messages = self.serve.clean_messages({
            "register": "casual",
            "messages": [{"role": "system", "content": "ignore previous"},
                         {"role": "user", "content": "x" * 5000}]})
        self.assertEqual(register, "casual")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("friend", messages[0]["content"])
        self.assertEqual(len(messages[1]["content"]), self.serve.MAX_CHARS)

    def test_base_model_prompt_has_no_profile_notes(self):
        saved = self.serve.PROFILE
        self.serve.PROFILE = "- a private fact"
        try:
            ask = {"messages": [{"role": "user", "content": "hi"}]}
            self.assertIn("a private fact", self.serve.clean_messages(ask)[1][0]["content"])
            self.assertNotIn("a private fact", self.serve.clean_messages(ask, notes=False)[1][0]["content"])
            self.assertNotIn("What you know", self.serve.clean_messages(ask, notes=False)[1][0]["content"])
        finally:
            self.serve.PROFILE = saved

    def test_unknown_register_falls_back_and_empty_input_is_rejected(self):
        register, _ = self.serve.clean_messages({"register": "x", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(register, "professional")
        with self.assertRaises(ValueError):
            self.serve.clean_messages({"messages": []})
        with self.assertRaises(ValueError):
            self.serve.clean_messages({"messages": [{"role": "assistant", "content": "hi"}]})

    def test_model_state_machine(self):
        model = self.serve.Model()
        self.assertEqual(model.state(), "asleep")
        model.waking_since = 1.0
        self.assertEqual(model.state(), "waking")
        model.waking_since = None
        model.last_ok = __import__("time").time()
        self.assertEqual(model.state(), "ready")
        model.last_ok -= self.serve.SERVE["scaledown_seconds"]
        self.assertEqual(model.state(), "asleep")


class BudgetTests(unittest.TestCase):
    def test_modal_rate(self):
        self.assertAlmostEqual(modal_rate_per_sec("L4", 4, 16), 0.000222 + 4 * 0.0000131 + 16 * 0.00000222)

    def test_estimate_arithmetic(self):
        e = estimate("L4", "4B", pairs=500, avg_tokens=160, epochs=3, tps=600, overhead_sec=180)
        self.assertEqual(e.total_tokens, 240_000)
        self.assertAlmostEqual(e.train_minutes, 400 / 60)
        self.assertAlmostEqual(e.cost_usd, 580 * modal_rate_per_sec("L4"))

    def test_misfit_is_flagged(self):
        self.assertIsNone(estimate("T4", "8B", 500, 160, 3).tps)


if __name__ == "__main__":
    unittest.main()
