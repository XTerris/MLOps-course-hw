"""End-to-end regressions: real HTTP, subprocesses and filesystem, no weights."""
import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT = Path(os.environ.get('HW8910_PROJECT', Path(__file__).resolve().parents[1])).resolve()
PYTHON = os.environ.get('HW8910_PYTHON', sys.executable)


def command(*args, ok=True):
    result = subprocess.run([PYTHON, str(PROJECT / 'lab.py'), *map(str, args)],
                            cwd=PROJECT, text=True, capture_output=True, timeout=120)
    if ok and result.returncode != 0:
        raise AssertionError(f'Command failed ({result.returncode}): {result.args}\n{result.stdout}\n{result.stderr}')
    return result


def request(port, route, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(f'http://127.0.0.1:{port}{route}', data=body,
                  headers={'Content-Type': 'application/json'})
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'workspace'
        command('--workspace', self.root, 'init')

    def run_lab(self, *args, ok=True):
        return command('--workspace', self.root, 'run', *args, ok=ok)

    def test_restart_cache_and_changed_input(self):
        fail = self.run_lab('--fail-after-prepare', ok=False)
        self.assertNotEqual(fail.returncode, 0)
        self.assertIn('PREPARED', fail.stdout)
        retry = self.run_lab()
        self.assertIn('CACHE_HIT', retry.stdout)
        with (self.root / 'input.txt').open('a') as f:
            f.write('changed input, same path\n')
        changed = self.run_lab()
        self.assertIn('PREPARED', changed.stdout)
        self.assertNotIn('CACHE_HIT', changed.stdout)

    def test_incomplete_foreign_and_refused_reports_preserve_pointer(self):
        for report in ('report-incomplete.json', 'report-A.json', 'report-reject.json'):
            with self.subTest(report=report):
                root = Path(self.temp.name) / report
                command('--workspace', root, 'init')
                accepted = root / 'accepted.json'
                original = accepted.read_bytes()
                result = command('--workspace', root, 'run', '--report', report, ok=False)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertEqual(accepted.read_bytes(), original)

    def test_repeated_release_has_one_publication(self):
        self.run_lab()
        accepted = (self.root / 'accepted.json').read_bytes()
        self.run_lab()
        self.assertEqual((self.root / 'accepted.json').read_bytes(), accepted)
        events = [json.loads(x) for x in (self.root / 'publications.jsonl').read_text().splitlines()]
        self.assertEqual(len(events), 1, events)
        self.assertEqual(events[0]['model_id'], 'fixture-B')

    def test_atomic_write_error_preserves_bytes(self):
        # Patch os.replace within a fresh Python process; no real disk exhaustion.
        program = '''import json,sys\nfrom pathlib import Path\nfrom unittest.mock import patch\nsys.path.insert(0,sys.argv[1])\nimport release\nr=Path(sys.argv[2]); m=json.loads((r/'manifest-B.json').read_text()); q=json.loads((r/'report-B.json').read_text())\ntry:\n  with patch('os.replace',side_effect=OSError('simulated disk error')): release.publish(m,q,r/'accepted.json')\nexcept OSError: sys.exit(0)\nsys.exit(3)\n'''
        before = (self.root / 'accepted.json').read_bytes()
        result = subprocess.run([PYTHON, '-c', program, str(PROJECT), str(self.root)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'accepted.json').read_bytes(), before)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_fault_after_pointer_update_recovers_once(self):
        fault = self.run_lab('--fail-after-publish', ok=False)
        self.assertNotEqual(fault.returncode, 0)
        self.assertEqual(json.loads((self.root / 'accepted.json').read_text())['model_id'], 'fixture-B')
        self.run_lab()
        self.run_lab()
        events = [json.loads(x) for x in (self.root / 'publications.jsonl').read_text().splitlines()]
        self.assertEqual(len(events), 1, events)
        self.assertEqual(events[0]['model_id'], 'fixture-B')

    def test_changed_model_file_rejected_without_overwriting_pointer(self):
        accepted = self.root / 'accepted.json'
        original = accepted.read_bytes()
        (self.root / 'model-B.txt').write_text('tampered weights')
        result = self.run_lab(ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(accepted.read_bytes(), original)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'workspace'
        command('--workspace', self.root, 'init')
        self.server = None
        self.addCleanup(self.stop_server)

    def start_server(self, warmup=0):
        with contextlib.closing(socket.socket()) as s:
            s.bind(('127.0.0.1', 0))
            self.port = s.getsockname()[1]
        self.server = subprocess.Popen([PYTHON, str(PROJECT / 'service.py'), '--workspace',
                                        str(self.root), '--port', str(self.port),
                                        '--warmup-seconds', str(warmup)], cwd=PROJECT,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.server.poll() is not None:
                self.fail(f'Server exited unexpectedly: {self.server.returncode}')
            try:
                if request(self.port, '/health')[0] == 200:
                    return
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(.05)
        self.fail('Server did not start')

    def stop_server(self):
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        self.server = None

    def wait_ready(self):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if request(self.port, '/ready')[0] == 200:
                return
            time.sleep(.05)
        self.fail('Server did not become ready')

    def test_warmup_readiness_blocks_generation(self):
        self.start_server(warmup=3)
        self.assertEqual(request(self.port, '/health')[0], 200)
        self.assertFalse(request(self.port, '/debug')[1]['warmup_finished'])
        self.assertEqual(request(self.port, '/ready')[0], 503)
        self.assertEqual(request(self.port, '/generate', {'prompt': 'test', 'max_new_tokens': 8})[0], 503)
        self.wait_ready()
        self.assertEqual(request(self.port, '/generate', {'prompt': 'test', 'max_new_tokens': 8})[0], 200)

    def test_loaded_version_stays_until_restart_and_can_roll_back(self):
        self.start_server()
        self.wait_ready()
        payload = {'prompt': 'test', 'max_new_tokens': 8}
        self.assertEqual(request(self.port, '/generate', payload)[1]['model_id'], 'fixture-A')
        command('--workspace', self.root, 'run')
        before_restart = request(self.port, '/generate', payload)[1]
        self.assertEqual(before_restart['model_id'], 'fixture-A')
        self.assertIn('fixture-A', before_restart['answer'])
        self.stop_server()
        self.start_server()
        self.wait_ready()
        loaded_b = request(self.port, '/generate', payload)[1]
        self.assertEqual(loaded_b['model_id'], 'fixture-B')
        self.assertIn('fixture-B', loaded_b['answer'])
        command('--workspace', self.root, 'run', '--manifest', 'manifest-A.json', '--report', 'report-A.json')
        self.stop_server()
        self.start_server()
        self.wait_ready()
        rolled_back = request(self.port, '/generate', payload)[1]
        self.assertEqual(rolled_back['model_id'], 'fixture-A')
        self.assertIn('fixture-A', rolled_back['answer'])

    def test_invalid_requests_never_reach_backend(self):
        self.start_server()
        self.wait_ready()
        calls = request(self.port, '/debug')[1]['calls']
        for payload in ({'prompt': '', 'max_new_tokens': 8},
                        {'prompt': 'test', 'max_new_tokens': 0},
                        {'prompt': 'test', 'max_new_tokens': 100000},
                        {'prompt': 'test', 'max_new_tokens': '8'}):
            with self.subTest(payload=payload):
                status, _ = request(self.port, '/generate', payload)
                self.assertGreaterEqual(status, 400)
                self.assertLess(status, 500)
                self.assertEqual(request(self.port, '/debug')[1]['calls'], calls)
        self.assertEqual(request(self.port, '/generate', {'prompt': 'test', 'max_new_tokens': 8})[0], 200)
        self.assertEqual(request(self.port, '/debug')[1]['calls'], calls + 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
