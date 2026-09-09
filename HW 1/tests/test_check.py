import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class CheckScriptTests(unittest.TestCase):
    def run_check(self, generation_exit_code):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            root = project / "HW 1"
            root.mkdir()
            (root / "tests").mkdir()
            (root / "src").mkdir()
            (root / "bin").mkdir()
            shutil.copy(Path(__file__).with_name("check.sh"), root / "tests/check.sh")
            original = 'model:\n  name: "test/original"\n'
            (root / "params.yaml").write_text(original)
            (project / "pyproject.toml").write_text('[project]\ndependencies = []\n')
            (project / "uv.lock").touch()
            for command, body in {
                "uv": "exit 0",
                "make": f"echo answer; exit {generation_exit_code}",
            }.items():
                executable = root / "bin" / command
                executable.write_text(f"#!/bin/sh\n{body}\n")
                executable.chmod(0o755)
            result = subprocess.run(
                ["bash", "tests/check.sh"],
                cwd=root,
                env={**os.environ, "PATH": f"{root / 'bin'}:{os.environ['PATH']}"},
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual((root / "params.yaml").read_text(), original)
            return result

    def test_generation_failure_cannot_pass_as_identical_output(self):
        result = self.run_check(2)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("первый запуск генерации завершился с ошибкой", result.stdout)

    def test_successful_checks_restore_original_config(self):
        result = self.run_check(0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Все проверки пройдены", result.stdout)


if __name__ == "__main__":
    unittest.main()
