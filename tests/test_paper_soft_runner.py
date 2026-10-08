"""Exercise batch sequencing and failure handling without PHY or model execution."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "evaluation" / "run_paper_soft_output_batch.sh"


def find_bash():
    candidates = []
    git = shutil.which("git")
    if git and os.name == "nt":
        candidates.extend((Path(git).resolve().parents[1] / "bin" / "bash.exe",
                           Path(git).resolve().parents[1] / "usr" / "bin" / "bash.exe"))
    available = shutil.which("bash")
    if available:
        candidates.append(Path(available))
    return next((str(candidate) for candidate in candidates if candidate.is_file()), None)


BASH = find_bash()


@unittest.skipUnless(BASH, "Bash is unavailable; server runner integration requires Bash")
class PaperSoftRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="paper soft runner ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runner = self.root / "evaluation" / RUNNER.name
        self.runner.parent.mkdir()
        shutil.copyfile(RUNNER, self.runner)
        self.python = self.root / "fake python.sh"
        self.python.write_text(
            '#!/usr/bin/env bash\n'
            'set -euo pipefail\n'
            'if [[ "$1" == -c ]]; then\n'
            '    if [[ -n "${SOFT_TEST_PATH_FAILURE:-}" ]]; then exit 71; fi\n'
            '    IFS=, read -r -a runs <<< "$4"\n'
            '    for run in "${runs[@]}"; do\n'
            '        for directory in cache results calibration logs; do\n'
            '            printf "%s/%s/%s\\n" "$SOFT_TEST_ROOT" "$directory" "$run"\n'
            '        done\n'
            '    done\n'
            '    exit 0\n'
            'fi\n'
            'printf "%s\\n" "$*" >> "$SOFT_TEST_ROOT/calls.txt"\n'
            'printf "fixture command: %s\\n" "$*"\n'
            'if [[ -n "${SOFT_TEST_FAIL:-}" && " $* " == *"$SOFT_TEST_FAIL"* ]]; then\n'
            '    printf "intentional stage failure\\n" >&2\n'
            '    exit 37\n'
            'fi\n',
            encoding="utf-8", newline="\n")
        self.python.chmod(0o755)
        self.environment = dict(os.environ, PYTHON=self.python.as_posix(),
                                SOFT_TEST_ROOT=self.root.as_posix())

    def invoke(self, *arguments, environment=None):
        process = subprocess.run([BASH, self.runner.as_posix(), *arguments],
                                 cwd=self.root, env=environment or self.environment,
                                 text=True, capture_output=True, timeout=40)
        return process

    def calls(self):
        path = self.root / "calls.txt"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def status(self, run_id="fixture"):
        contents = (self.root / "logs" / run_id / "runner_status.txt").read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in contents.splitlines())

    def assert_success(self, process):
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_bash_syntax_and_help(self):
        syntax = subprocess.run([BASH, "-n", RUNNER.as_posix()], text=True,
                                capture_output=True, timeout=10)
        self.assert_success(syntax)
        help_output = self.invoke("--help")
        self.assert_success(help_output)
        self.assertIn("--run-id", help_output.stdout)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "logs").exists())

    def test_invalid_mode_and_path_run_ids_stop_before_execution(self):
        for arguments in (("full", "--run-id", "fixture"),
                          ("smoke",), ("smoke", "--run-id", "../escape"),
                          ("smoke", "--run-id", "."),
                          ("smoke", "--run-id", "nested/run"),
                          ("smoke", "--run-id"),
                          ("smoke", "--run-id", "fixture", "--unknown")):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.invoke(*arguments).returncode, 2)
        self.assertEqual(self.calls(), [])

    def test_path_resolution_failure_propagates(self):
        environment = dict(self.environment, SOFT_TEST_PATH_FAILURE="1")
        process = self.invoke("smoke", "--run-id", "fixture", environment=environment)
        self.assertEqual(process.returncode, 71, process.stdout + process.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "logs").exists())

    def test_smoke_runs_only_full_codeword_smoke_stages(self):
        process = self.invoke("smoke", "--run-id", "fixture")
        self.assert_success(process)
        calls = self.calls()
        self.assertEqual(len(calls), 5)
        self.assertTrue(calls[0].endswith("--mode smoke --preflight"))
        self.assertIn("unittest discover -s tests -p test_paper_soft_*.py -v", calls[1])
        self.assertIn("evaluation.build_paper_soft_cache", calls[2])
        self.assertTrue(calls[3].endswith("--mode smoke --stage fit"))
        self.assertTrue(calls[4].endswith("--mode smoke --stage evaluate"))
        self.assertFalse(any("a1" in call.lower() or "replay" in call.lower() for call in calls))
        self.assertEqual(self.status()["state"], "complete")
        self.assertEqual(self.status()["exit_code"], "0")
        self.assertEqual(self.status()["stage"], "summary")
        self.assertFalse((self.root / "logs" / "fixture" / ".runner.lock").exists())

    def test_development_orders_smoke_before_full_batch(self):
        process = self.invoke("development", "--run-id", "fixture")
        self.assert_success(process)
        calls = self.calls()
        self.assertEqual(len(calls), 8)
        self.assertTrue(calls[0].endswith("--mode development --preflight"))
        for call in calls[2:5]:
            self.assertIn("--run-id fixture_smoke --mode smoke", call)
        self.assertIn("--run-id fixture --mode development", calls[5])
        self.assertIn("evaluation.build_paper_soft_cache", calls[5])
        self.assertTrue(calls[6].endswith("--mode development --stage fit"))
        self.assertTrue(calls[7].endswith("--mode development --stage evaluate"))
        self.assertEqual(self.status()["state"], "complete")

    def test_failed_smoke_decoder_check_prevents_full_generation(self):
        environment = dict(self.environment, SOFT_TEST_FAIL="--mode smoke --stage evaluate")
        process = self.invoke("development", "--run-id", "fixture", environment=environment)
        self.assertEqual(process.returncode, 37, process.stdout + process.stderr)
        self.assertEqual(len(self.calls()), 5)
        status = self.status()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["stage"], "smoke_evaluate")
        self.assertEqual(status["exit_code"], "37")
        log_dir = self.root / "logs" / "fixture"
        self.assertIn("intentional stage failure", (log_dir / "runner.log").read_text())
        self.assertIn("smoke_evaluate\tfailed", (log_dir / "stages.tsv").read_text())
        self.assertTrue((log_dir / "runner.pid").read_text().strip().isdigit())

    def test_preflight_or_tests_failure_stops_before_simulation(self):
        for index, failure in enumerate(("--preflight", "-m unittest")):
            with self.subTest(failure=failure):
                run_id = f"failure{index}"
                environment = dict(self.environment, SOFT_TEST_FAIL=failure)
                before = len(self.calls())
                process = self.invoke("development", "--run-id", run_id, environment=environment)
                self.assertEqual(process.returncode, 37)
                self.assertEqual(len(self.calls()) - before, index + 1)
                self.assertEqual(self.status(run_id)["state"], "failed")

    def test_refuses_existing_artifacts_and_smoke_directory_by_default(self):
        for directory, run_id in (("cache", "fixture"), ("results", "fixture"),
                                  ("calibration", "fixture"), ("logs", "fixture_smoke")):
            with self.subTest(directory=directory, run_id=run_id):
                existing = self.root / directory / run_id
                existing.mkdir(parents=True)
                sentinel = existing / "preserve.txt"
                sentinel.write_text("keep")
                process = self.invoke("development", "--run-id", "fixture")
                self.assertEqual(process.returncode, 2)
                self.assertIn("Output already exists", process.stderr)
                self.assertEqual(sentinel.read_text(), "keep")
                sentinel.unlink()
                existing.rmdir()
        self.assertEqual(self.calls(), [])

    def test_explicit_reuse_passes_both_reuse_flags_and_appends_logs(self):
        self.assert_success(self.invoke("smoke", "--run-id", "fixture"))
        log_file = self.root / "logs" / "fixture" / "runner.log"
        original_log = log_file.read_text()
        process = self.invoke("smoke", "--run-id", "fixture", "--reuse-cache")
        self.assert_success(process)
        calls = self.calls()[5:]
        self.assertTrue(calls[2].endswith("--reuse-cache"))
        self.assertTrue(calls[3].endswith("--reuse-results"))
        self.assertTrue(calls[4].endswith("--reuse-results"))
        self.assertTrue(log_file.read_text().startswith(original_log))
        self.assertGreater(log_file.stat().st_size, len(original_log))

    def test_existing_lock_refuses_concurrent_runner(self):
        lock = self.root / "logs" / "fixture" / ".runner.lock"
        lock.mkdir(parents=True)
        process = self.invoke("smoke", "--run-id", "fixture", "--reuse-cache")
        self.assertEqual(process.returncode, 2)
        self.assertIn("Runner lock already exists", process.stderr)
        self.assertTrue(lock.exists())
        self.assertEqual(self.calls(), [])

    def test_tee_failure_is_not_reported_as_success(self):
        shell_environment = self.root / "tee failure environment.sh"
        shell_environment.write_text("tee() { cat >/dev/null; return 79; }\n", encoding="utf-8", newline="\n")
        environment = dict(self.environment, BASH_ENV=shell_environment.as_posix())
        process = self.invoke("smoke", "--run-id", "fixture", environment=environment)
        self.assertEqual(process.returncode, 79, process.stdout + process.stderr)
        self.assertEqual(self.status()["state"], "failed")
        self.assertEqual(self.status()["exit_code"], "79")


if __name__ == "__main__":
    unittest.main()
