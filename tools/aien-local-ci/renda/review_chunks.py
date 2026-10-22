"""Bounded, resumable review of complete local diff text.

No transport, model, scheduler, or inference dependencies live here. The caller
supplies one ask callback and a per-pass call budget.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

SCHEMA = 'renda-review-chunks/1'


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _identity(repository, key, base, head, diff, model, policy, chunk_chars):
    diff_digest = _digest(diff.encode('utf-8'))
    value = dict(schema=SCHEMA, repository=repository, key=key, base=base, head=head,
                 diff_digest=diff_digest, model=model, policy=policy, chunk_chars=chunk_chars)
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return _digest(encoded), diff_digest


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _make_chunks(diff, chunk_chars):
    chunks = []
    headers = list(re.finditer(r'^diff --git a/(.*?) b/(.*?)$', diff, re.M))
    header_index = 0
    file_name = headers[0].group(2) if headers else '(diff preamble)'
    for start in range(0, len(diff), chunk_chars):
        end = min(len(diff), start + chunk_chars)
        while header_index + 1 < len(headers) and headers[header_index + 1].start() <= start:
            header_index += 1
            file_name = headers[header_index].group(2)
        files = [file_name]
        for header in headers[header_index + 1:]:
            if header.start() >= end:
                break
            files.append(header.group(2))
        text = diff[start:end]
        chunks.append(dict(index=len(chunks), start=start, end=end,
                           file=', '.join(dict.fromkeys(files)),
                           chunk_digest=_digest(text.encode('utf-8')),
                           text=text, decision=None, note=''))
    return chunks


def _valid_cache(value, identity, diff_digest, chunks):
    if (not isinstance(value, dict) or value.get('schema') != SCHEMA
            or value.get('identity') != identity or value.get('diff_digest') != diff_digest
            or not isinstance(value.get('chunks'), list) or len(value['chunks']) != len(chunks)):
        return False
    for expected, actual in zip(chunks, value['chunks']):
        if (not isinstance(actual, dict)
                or any(actual.get(k) != expected[k] for k in ('index', 'start', 'end', 'file', 'chunk_digest', 'text'))
                or type(actual.get('decision')) not in (bool, type(None))
                or not isinstance(actual.get('note', ''), str)):
            return False
    return True


def _prompt(repository, key, base, head, model, policy, identity, diff_digest, chunk, count):
    return (
        'Review this complete-diff segment as a bounded part of one pinned LOCAL review. '
        'Qualification is pending; this review grants no merge authority. Inspect for concrete defects, '
        'secrets, unrelated changes, or destructive edits. Return strict JSON '
        '{"merge": true or false, "note": "one line"}.\n'
        f'Repository {repository}; queue key {key}; base {base}; head {head}; model {model}; policy {policy}.\n'
        f'Full review identity {identity}; full diff digest {diff_digest}.\n'
        f'chunk {chunk["index"] + 1}/{count}; file {chunk["file"]}; '
        f'offsets {chunk["start"]}:{chunk["end"]}; chunk digest {chunk["chunk_digest"]}.\n'
        'Segment follows exactly, including partial-line boundaries indicated by offsets:\n'
        + chunk['text']
    )


def review_diff(diff, *, repository, key, base, head, model, policy, ask, cache_path,
                budget=2, chunk_chars=12000, max_attempts=2):
    """Review text in exact character ranges; return a fail-closed partial result.

    Each invocation spends at most ``budget`` actual callback invocations. Only
    transport/malformed results are retried, up to ``max_attempts`` and the
    supplied call budget. Explicit false answers are durable refusals.
    """
    if not isinstance(diff, str):
        raise TypeError('diff must be decoded text')
    if type(budget) is not int or budget < 0:
        raise ValueError('budget must be a nonnegative integer')
    if type(chunk_chars) is not int or chunk_chars < 1:
        raise ValueError('chunk_chars must be positive')
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    identity, diff_digest = _identity(repository, key, base, head, diff, model, policy, chunk_chars)
    chunks = _make_chunks(diff, chunk_chars)
    if not chunks:
        return dict(complete=False, decision=None, note='empty diff has no reviewable chunks', actual_calls=0,
                    total_chunks=0, reviewed_chunks=0, identity=identity)
    cache_path = Path(cache_path)
    try:
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        cached = None
    if _valid_cache(cached, identity, diff_digest, chunks):
        rows = cached['chunks']
        for source, target in zip(rows, chunks):
            target['decision'], target['note'] = source['decision'], source.get('note', '')
    else:
        rows = chunks
        cached = dict(schema=SCHEMA, identity=identity, diff_digest=diff_digest, chunks=rows)
        _atomic_json(cache_path, cached)
    actual_calls = 0
    for chunk in rows:
        if any(row['decision'] is False for row in rows):
            break  # A real defect is terminal, not a reason to spend another pass.
        if chunk['decision'] is not None:
            continue
        attempts = 0
        while attempts < max_attempts and actual_calls < budget:
            attempts += 1
            actual_calls += 1
            try:
                text, returned_model = ask(_prompt(repository, key, base, head, model, policy,
                    identity, diff_digest, chunk, len(rows)))
                if returned_model != model:
                    raise ValueError('review callback returned an unexpected model identity')
                parsed = json.loads(text) if isinstance(text, str) else None
                if not isinstance(parsed, dict) or type(parsed.get('merge')) is not bool:
                    raise ValueError('review response must contain explicit boolean merge')
                chunk['decision'] = parsed['merge']
                chunk['note'] = str(parsed.get('note', ''))[:300]
                break
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'[:300]
                chunk['note'] = 'incomplete response: ' + last_error
                _atomic_json(cache_path, cached)
                if attempts < max_attempts and actual_calls < budget:
                    time.sleep(min(1.0, 0.1 * (2 ** (attempts - 1))))
        _atomic_json(cache_path, cached)
        if chunk['decision'] is False or actual_calls >= budget:
            break
    decisions = [chunk['decision'] for chunk in rows]
    complete = all(type(answer) is bool for answer in decisions)
    decision = False if any(answer is False for answer in decisions) else (True if complete else None)
    note = next((chunk['note'] for chunk in rows if chunk['decision'] is False and chunk['note']), '')
    if decision is None:
        note = next((chunk['note'] for chunk in rows if chunk['decision'] is None and chunk['note']),
                    'review incomplete; resume remaining chunks within the next pass budget')
    return dict(complete=complete, decision=decision, note=note, actual_calls=actual_calls,
                total_chunks=len(rows), reviewed_chunks=sum(type(answer) is bool for answer in decisions),
                identity=identity, diff_digest=diff_digest)
