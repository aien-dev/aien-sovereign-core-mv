"""Fail-closed GraphQL transport for publication identity reads."""
import json
import re

from local_queue import PR_FIELDS, decode_one, decode_pr, graphql_args

MAX_PAGES = 10


def _response(run, args, timeout):
    try:
        rc, out, err = run(['gh'] + args, timeout=timeout)
    except Exception as exc:
        raise ValueError(f'GitHub GraphQL transport failed: {exc}') from exc
    if rc:
        raise ValueError(f'GitHub GraphQL transport exit {rc}: {(err or out)[-400:]}')
    try:
        value = json.loads(out)
    except (TypeError, ValueError) as exc:
        raise ValueError('malformed GitHub GraphQL JSON') from exc
    if not isinstance(value, dict) or value.get('errors'):
        raise ValueError('GitHub GraphQL errors or malformed response')
    return value


def list_by_head(run, full, head_ref_name, timeout=120):
    """Return a complete, validated all-state PR snapshot for one exact head ref.

    GitHub exposes OPEN, CLOSED and MERGED as distinct GraphQL states. Every
    page is decoded and validated before callers may make a publication choice.
    """
    if (not isinstance(full, str) or not re.fullmatch(r'[^/]+/[^/]+', full)
            or not isinstance(head_ref_name, str) or not head_ref_name
            or head_ref_name.startswith('-')):
        raise ValueError('invalid publication repository or head ref')
    owner, name = full.split('/')
    query = ('query($owner:String!,$name:String!,$headRefName:String!,$cursor:String){'
        'repository(owner:$owner,name:$name){pullRequests(first:100,states:[OPEN,CLOSED,MERGED],'
        'headRefName:$headRefName,after:$cursor){pageInfo{hasNextPage endCursor} nodes{' +
        PR_FIELDS + '}}}}')
    cursor = None
    seen_cursors = set()
    prs = []
    numbers = set()
    for page_index in range(MAX_PAGES):
        args = ['api', 'graphql', '-f', 'query=' + query, '-f', 'owner=' + owner,
                '-f', 'name=' + name, '-f', 'headRefName=' + head_ref_name]
        if cursor is not None:
            args += ['-f', 'cursor=' + cursor]
        value = _response(run, args, timeout)
        try:
            connection = value['data']['repository']['pullRequests']
            nodes = connection['nodes']
            page_info = connection['pageInfo']
            more = page_info['hasNextPage']
        except (KeyError, TypeError) as exc:
            raise ValueError('malformed publication PR connection') from exc
        if not isinstance(nodes, list) or type(more) is not bool:
            raise ValueError('malformed publication PR page')
        for node in nodes:
            try:
                pr = decode_pr(node)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'malformed publication PR: {exc}') from exc
            if pr['headRefName'] != head_ref_name:
                raise ValueError('GraphQL headRefName result mismatch')
            if pr['number'] in numbers:
                raise ValueError('duplicate PR in publication lookup')
            numbers.add(pr['number'])
            prs.append(pr)
        if not more:
            return prs
        next_cursor = page_info.get('endCursor')
        if (not isinstance(next_cursor, str) or not next_cursor
                or next_cursor == cursor or next_cursor in seen_cursors):
            raise ValueError('malformed or repeated publication page cursor')
        if page_index + 1 >= MAX_PAGES:
            raise ValueError('publication PR lookup exceeds complete pagination bound')
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    raise ValueError('publication PR lookup incomplete')


def view_by_number(run, full, number, timeout=120):
    """Fetch and decode one current PR identity and exact-head check context."""
    if type(number) is not int or number <= 0:
        raise ValueError('invalid publication PR number')
    value = _response(run, graphql_args(full, number=number), timeout)
    try:
        pr = decode_one(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'malformed publication PR readback: {exc}') from exc
    if pr['number'] != number:
        raise ValueError('publication PR number readback mismatch')
    return pr
