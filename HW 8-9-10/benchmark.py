"""Measure the same warmed-up requests with one and two concurrent clients."""
import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = 'Title: Crash when opening an empty file\nBody: The application raises IndexError instead of opening an empty document.\nChoices: bug, documentation, feature, question'


def request_one(url, mode, index, timeout):
    payload = json.dumps({'prompt': PROMPT, 'max_new_tokens': 32}).encode()
    request = urllib.request.Request(url + '/generate', payload, {'Content-Type': 'application/json'}, method='POST')
    started = time.monotonic()
    record = {'mode': mode, 'index': index}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
            record.update(status=response.status, model_id=data.get('model_id'), request_id=data.get('request_id'),
                          generation_seconds=data.get('generation_seconds'), waiting_seconds=data.get('waiting_seconds'),
                          output_tokens=data.get('output_tokens'), runtime_sha256=data.get('runtime_sha256'),
                          diagnostic_only=data.get('diagnostic_only'), answer=data.get('answer'))
    except Exception as exc:
        record.update(status=getattr(exc, 'code', None), error=f'{type(exc).__name__}: {exc}')
    record['client_seconds'] = time.monotonic() - started
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8091')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--output', type=Path, default=Path('results/benchmark.json'))
    args = parser.parse_args()
    result = {'prompt': PROMPT, 'max_new_tokens': 32, 'runs': []}
    for concurrency in (1, 2):
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(request_one, args.url, concurrency, i, args.timeout) for i in range(args.count)]
            attempts = [f.result() for f in futures]
        elapsed = time.monotonic() - started
        good = [r for r in attempts if r.get('status') == 200 and not r.get('error')]
        run = {'concurrency': concurrency, 'n': args.count, 'success': len(good),
               'errors': args.count - len(good), 'batch_seconds': elapsed,
               'successful_requests_per_second': len(good) / elapsed,
               'mean_success_seconds': statistics.mean(r['client_seconds'] for r in good) if good else None,
               'max_success_seconds': max((r['client_seconds'] for r in good), default=None),
               'attempts': attempts}
        result['runs'].append(run)
        print(json.dumps({k: v for k, v in run.items() if k != 'attempts'}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
