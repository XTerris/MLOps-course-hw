"""Строгая оценка ответов и допуск без разрешённой регрессии."""
import json
import math


def _ids(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Нужен непустой список записей')
    ids = [r.get('id') for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('ID должны быть непустыми уникальными строками')
    return ids


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повтор JSON-ключа')
        result[key] = value
    return result


def _counts(items):
    n = len(items)
    correct = sum(r['correct'] for r in items)
    valid = sum(r['valid'] for r in items)
    errors = sum(r['error'] is not None for r in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n if n else 0.0, format_rate=valid / n if n else 0.0)


def score(rows, outputs):
    ids, output_ids = _ids(rows), _ids(outputs)
    if set(ids) != set(output_ids):
        raise ValueError('Наборы ID вопросов и ответов различаются')
    by_id = {r['id']: r for r in outputs}
    items = []
    for row in rows:
        choices, slices = row.get('choices'), row.get('slices')
        if (not isinstance(choices, list) or not choices or
                any(not isinstance(c, str) for c in choices) or row.get('expected') not in choices or
                not isinstance(slices, list) or not slices or
                any(not isinstance(s, str) or not s for s in slices) or len(slices) != len(set(slices))):
            raise ValueError('Некорректные choices, expected или slices')
        output = by_id[row['id']]
        raw, error = output.get('raw', ''), output.get('error')
        parsed, valid = None, False
        if error is None and isinstance(raw, str):
            try:
                parsed = json.loads(raw, object_pairs_hook=_object,
                                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                valid = (isinstance(parsed, dict) and set(parsed) == {'answer'} and
                         isinstance(parsed['answer'], str) and parsed['answer'] in choices)
            except (ValueError, TypeError, RecursionError):
                parsed = None
        items.append(dict(id=row['id'], raw=raw, parsed=parsed, valid=bool(valid),
                          correct=bool(valid and parsed['answer'] == row['expected']),
                          error=error, slices=list(slices)))
    names = sorted({s for r in items for s in r['slices']})
    return dict(overall=_counts(items),
                slices={s: _counts([r for r in items if s in r['slices']]) for s in names},
                per_item=items)


def _validate_metrics(actual, expected):
    if not isinstance(actual, dict):
        raise ValueError('Нет метрик')
    for name in ('n', 'correct', 'valid', 'errors'):
        value = actual.get(name)
        if type(value) is not int or value < 0 or value != expected[name]:
            raise ValueError(f'Некорректный count: {name}')
    for name in ('accuracy', 'format_rate'):
        value = actual.get(name)
        if (type(value) not in (int, float) or not math.isfinite(value) or
                not math.isclose(value, expected[name], rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'Метрика не соответствует counts: {name}')


def _validate_report(report):
    items = report['per_item']
    _ids(items)
    for item in items:
        if (type(item['valid']) is not bool or type(item['correct']) is not bool or
                (item['correct'] and not item['valid']) or
                (item['error'] is not None and (item['valid'] or item['correct']))):
            raise ValueError('Несогласованный per_item')
        slices = item['slices']
        if (not isinstance(slices, list) or not slices or
                any(not isinstance(s, str) or not s for s in slices) or len(slices) != len(set(slices))):
            raise ValueError('Некорректные срезы per_item')
    _validate_metrics(report['overall'], _counts(items))
    names = {s for r in items for s in r['slices']}
    if not isinstance(report['slices'], dict) or set(report['slices']) != names:
        raise ValueError('Срезы не соответствуют per_item')
    for name in names:
        _validate_metrics(report['slices'][name], _counts([r for r in items if name in r['slices']]))
    return {r['id']: set(r['slices']) for r in items}


def quality_gate(baseline, candidate, critical_slices):
    try:
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if (not isinstance(baseline.get(key), str) or not baseline[key] or
                    baseline[key] != candidate.get(key)):
                raise ValueError(f'Несовместимый {key}')
        membership = []
        behavior_membership = []
        for report in (baseline, candidate):
            if (report.get('status') != 'complete' or not isinstance(report.get('model'), dict) or
                    not report['model'] or not report['model'].get('kind')):
                raise ValueError('Нужны полный отчёт и конкретная модель')
            membership.append(_validate_report(report))
            if report['overall']['n'] < 50 or len(report['slices']) < 3:
                raise ValueError('Нужны минимум 50 вопросов и три среза')
            if report['overall']['errors']:
                raise ValueError('Есть ошибки генерации')
            behavior_membership.append(_validate_report(report['behavioral']))
            if report['behavioral']['overall']['n'] < 3 or report['behavioral']['overall']['errors']:
                raise ValueError('Неполная behavioral-проверка или ошибка генерации')
        if membership[0] != membership[1] or behavior_membership[0] != behavior_membership[1]:
            raise ValueError('Различаются ID или состав срезов')
        if (not isinstance(critical_slices, list) or not critical_slices or
                any(not isinstance(s, str) or s not in candidate['slices'] for s in critical_slices)):
            raise ValueError('Отсутствует критический срез')
        for metric in ('accuracy', 'format_rate'):
            if candidate['overall'][metric] < baseline['overall'][metric]:
                raise ValueError(f'Регрессия общей {metric}')
        for name in critical_slices:
            if candidate['slices'][name]['accuracy'] < baseline['slices'][name]['accuracy']:
                raise ValueError(f'Регрессия критического среза {name}')
        behavior = candidate['behavioral']['overall']
        if behavior['correct'] != behavior['n']:
            raise ValueError('Не все behavioral-кейсы кандидата пройдены')
        return dict(passed=True, tolerance=0.0, critical_slices=critical_slices,
                    baseline_accuracy=baseline['overall']['accuracy'],
                    candidate_accuracy=candidate['overall']['accuracy'],
                    protocol_sha256=candidate['protocol_sha256'])
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError(f'Неполный или некорректный отчёт: {exc}') from exc
