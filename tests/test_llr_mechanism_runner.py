"""Exercise confirmation sequencing with stub commands, never physical channels."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "evaluation" / "run_llr_mechanism_controls_batch.sh"


def find_bash():
    candidates = []
    git = shutil.which("git")
    if git and os.name == "nt":
        candidates.append(Path(git).resolve().parents[1] / "bin" / "bash.exe")
    available = shutil.which("bash")
    if available:
        candidates.append(Path(available))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


BASH = find_bash()


@unittest.skipUnless(BASH, "Bash is unavailable; batch orchestration tests require Bash")
class LLRMechanismRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mechanism runner ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runner = self.root / "evaluation" / RUNNER.name
        self.runner.parent.mkdir()
        shutil.copyfile(RUNNER, self.runner)
        self.python = self.root / "stub python.sh"
        self.python.write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'if [[ "$1" == -c ]]; then\n'
            '    if [[ -n "${MECHANISM_TEST_PATH_FAIL:-}" ]]; then exit 71; fi\n'
            '    IFS=, read -r -a runs <<< "$4"\n'
            '    for run in "${runs[@]}"; do\n'
            '        for directory in training results logs; do\n'
            '            printf "%s/%s/%s\\n" "$MECHANISM_TEST_ROOT" "$directory" "$run"\n'
            '        done\n'
            '    done\n'
            '    exit 0\n'
            'fi\n'
            'printf "%s\\n" "$*" >> "$MECHANISM_TEST_ROOT/calls.txt"\n'
            'printf "stub command: %s\\n" "$*"\n'
            'if [[ -n "${MECHANISM_TEST_FAIL:-}" && " $* " == *"$MECHANISM_TEST_FAIL"* ]]; then\n'
            '    printf "intentional stage failure\\n" >&2\n'
            '    exit 37\n'
            'fi\n', encoding="utf-8", newline="\n")
        self.python.chmod(0o755)
        self.environment = dict(os.environ, PYTHON=self.python.as_posix(), MECHANISM_TEST_ROOT=self.root.as_posix())

    def invoke(self, *arguments, environment=None):
        return subprocess.run([BASH, self.runner.as_posix(), *arguments], cwd=self.root,
                              env=environment or self.environment, text=True, capture_output=True, timeout=40)

    def calls(self):
        path = self.root / "calls.txt"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def status(self, run_id="fixture"):
        contents = (self.root / "logs" / run_id / "runner_status.txt").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in contents.splitlines())

    def assert_success(self, process):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_bash_syntax_and_help(self):
        self.assert_success(subprocess.run([BASH, "-n", RUNNER.as_posix()], text=True,
                                           capture_output=True, timeout=10))
        process = self.invoke("--help")
        self.assert_success(process)
        self.assertIn("Missing training cache is an error", process.stdout)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "logs").exists())

    def test_invalid_mode_run_id_and_options_rejected(self):
        for arguments in (("full", "--run-id", "fixture"), ("smoke",),
                          ("smoke", "--run-id", "../escape"), ("smoke", "--run-id", "."),
                          ("smoke", "--run-id", "a/b"), ("smoke", "--run-id"),
                          ("smoke", "--run-id", "fixture", "--reuse-cache")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(*arguments).returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_path_resolution_failure_stops_before_outputs(self):
        environment = dict(self.environment, MECHANISM_TEST_PATH_FAIL="1")
        process = self.invoke("smoke", "--run-id", "fixture", environment=environment)
        self.assertEqual(process.returncode, 71)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "logs").exists())

    def test_smoke_orders_new_training_decode_and_summary_check(self):
        self.assert_success(self.invoke("smoke", "--run-id", "fixture"))
        calls = self.calls()
        self.assertEqual(len(calls), 5)
        self.assertTrue(calls[0].endswith("--mode smoke --preflight"))
        self.assertIn("unittest discover -s tests -p test_llr_mechanism_*.py -v", calls[1])
        self.assertIn("-m training.train_llr_mechanism_controls", calls[2])
        self.assertIn("-m evaluation.evaluate_llr_mechanism_controls", calls[3])
        self.assertTrue(calls[4].endswith("--mode smoke --check-complete"))
        self.assertFalse(any("build_paper" in call or "a1" in call.lower() for call in calls))
        self.assertEqual(self.status()["state"], "complete")
        self.assertEqual(self.status()["exit_code"], "0")
        self.assertTrue(self.status()["started_utc"].endswith("Z"))
        self.assertFalse((self.root / "logs" / "fixture" / ".runner.lock").exists())

    def test_confirmation_requires_successful_smoke_check_before_training(self):
        self.assert_success(self.invoke("confirmation", "--run-id", "fixture"))
        calls = self.calls()
        self.assertEqual(len(calls), 8)
        self.assertTrue(calls[0].endswith("--mode confirmation --preflight"))
        for call in calls[2:5]:
            self.assertIn("--run-id fixture_smoke --mode smoke", call)
        self.assertTrue(calls[4].endswith("--check-complete"))
        self.assertIn("-m training.train_llr_mechanism_controls", calls[5])
        self.assertIn("-m evaluation.evaluate_llr_mechanism_controls", calls[6])
        self.assertTrue(calls[7].endswith("--mode confirmation --check-complete"))

    def test_missing_cache_preflight_failure_never_starts_training_or_generation(self):
        environment = dict(self.environment, MECHANISM_TEST_FAIL="--preflight")
        process = self.invoke("confirmation", "--run-id", "fixture", environment=environment)
        self.assertEqual(process.returncode, 37)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.status()["stage"], "preflight")
        self.assertEqual(self.status()["state"], "failed")

    def test_failed_tests_or_smoke_stops_expensive_stages_and_records_exit(self):
        for index, (failure, count) in enumerate((("-m unittest", 2),
                                                ("--run-id case1_smoke --mode smoke --check-complete", 5))):
            run_id = f"case{index}"
            environment = dict(self.environment, MECHANISM_TEST_FAIL=failure)
            before = len(self.calls())
            with self.subTest(failure=failure):
                process = self.invoke("confirmation", "--run-id", run_id, environment=environment)
                self.assertEqual(process.returncode, 37, process.stdout + process.stderr)
                self.assertEqual(len(self.calls()) - before, count)
                self.assertEqual(self.status(run_id)["state"], "failed")
                self.assertEqual(self.status(run_id)["exit_code"], "37")
                log_dir = self.root / "logs" / run_id
                self.assertIn("intentional stage failure", (log_dir / "runner.log").read_text())
                self.assertIn("\tfailed\t", (log_dir / "stages.tsv").read_text())
                self.assertTrue((log_dir / "runner.pid").read_text().strip().isdigit())

    def test_existing_output_or_smoke_directory_refused_without_explicit_reuse(self):
        for directory, run_id in (("training", "fixture"), ("results", "fixture"), ("logs", "fixture_smoke")):
            with self.subTest(directory=directory):
                existing = self.root / directory / run_id
                existing.mkdir(parents=True)
                sentinel = existing / "preserve.txt"
                sentinel.write_text("keep")
                process = self.invoke("confirmation", "--run-id", "fixture")
                self.assertEqual(process.returncode, 2)
                self.assertIn("Output exists", process.stderr)
                self.assertEqual(sentinel.read_text(), "keep")
                sentinel.unlink()
                existing.rmdir()
        self.assertEqual(self.calls(), [])

    def test_explicit_reuse_propagates_and_appends_logs(self):
        self.assert_success(self.invoke("smoke", "--run-id", "fixture"))
        log_file = self.root / "logs" / "fixture" / "runner.log"
        original_log = log_file.read_text()
        self.assert_success(self.invoke("smoke", "--run-id", "fixture", "--reuse-complete"))
        calls = self.calls()[5:]
        self.assertTrue(calls[2].endswith("--reuse-complete"))
        self.assertTrue(calls[3].endswith("--reuse-complete"))
        self.assertTrue(calls[4].endswith("--check-complete"))
        self.assertTrue(log_file.read_text().startswith(original_log))
        self.assertGreater(log_file.stat().st_size, len(original_log))

    def test_existing_lock_refuses_concurrent_runner(self):
        lock = self.root / "logs" / "fixture" / ".runner.lock"
        lock.mkdir(parents=True)
        process = self.invoke("smoke", "--run-id", "fixture", "--reuse-complete")
        self.assertEqual(process.returncode, 2)
        self.assertIn("Runner lock exists", process.stderr)
        self.assertTrue(lock.exists())
        self.assertEqual(self.calls(), [])

    def test_tee_failure_propagates(self):
        shell_environment = self.root / "tee failure environment.sh"
        shell_environment.write_text("tee() { cat >/dev/null; return 79; }\n", encoding="utf-8", newline="\n")
        environment = dict(self.environment, BASH_ENV=shell_environment.as_posix())
        process = self.invoke("smoke", "--run-id", "fixture", environment=environment)
        self.assertEqual(process.returncode, 79)
        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual(self.status()["exit_code"], "79")


if __name__ == "__main__":
    unittest.main()
