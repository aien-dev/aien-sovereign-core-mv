"""Fail-closed GraphQL publication transport regressions."""
import json
import unittest

import publication_transport as transport

HEAD = 'a' * 40
BASE = 'b' * 40


def node(number=17, state='OPEN', head='feat/topic', oid=HEAD, base_oid=BASE):
    return dict(number=number, title='fixture', isDraft=False, headRefName=head,
        headRefOid=oid, baseRefName='main', baseRefOid=base_oid,
        mergeStateStatus='CLEAN', state=state, createdAt='2026-01-01T00:00:00Z',
        commits=dict(nodes=[dict(commit=dict(oid=oid, statusCheckRollup=None))]))


def page(nodes, more=False, cursor='cursor-1'):
    return dict(data=dict(repository=dict(pullRequests=dict(
        nodes=nodes, pageInfo=dict(hasNextPage=more, endCursor=cursor if more else None)))))


class PublicationTransport(unittest.TestCase):
    def run_pages(self, values):
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            value = values.pop(0)
            return 0, json.dumps(value), ''
        return transport.list_by_head(run, 'aien-dev/repo', 'feat/topic'), calls

    def test_lookup_uses_all_states_head_filter_and_returns_exact_pins(self):
        rows, calls = self.run_pages([page([node()])])
        self.assertEqual([(p['number'], p['headRefOid'], p['baseRefOid'])
                          for p in rows], [(17, HEAD, BASE)])
        query = next(arg[6:] for arg in calls[0] if arg.startswith('query='))
        self.assertIn('headRefName:$headRefName', query)
        self.assertIn('states:[OPEN,CLOSED,MERGED]', query)
        self.assertIn('baseRefOid', query)

    def test_lookup_completes_pagination(self):
        rows, calls = self.run_pages([page([node()], True), page([node(18, oid='c' * 40)])])
        self.assertEqual([p['number'] for p in rows], [17, 18])
        self.assertEqual(len(calls), 2)
        self.assertIn('cursor=cursor-1', calls[1])

    def test_lookup_refuses_graphql_errors_and_malformed_or_duplicate_rows(self):
        for response in (
            dict(errors=[dict(message='denied')]), page([dict(number=17)]),
            page([node(), node()]), page([node()], True, cursor=None),
        ):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    self.run_pages([response])

    def test_lookup_refuses_unfinished_page_bound(self):
        responses = [page([node(i)], True, f'cursor-{i}') for i in range(transport.MAX_PAGES)]
        with self.assertRaises(ValueError):
            self.run_pages(responses)

    def test_number_readback_uses_graphql_and_decodes_exact_base(self):
        calls = []
        def run(cmd, **kwargs):
            calls.append(cmd)
            return 0, json.dumps(dict(data=dict(repository=dict(pullRequest=node())))), ''
        result = transport.view_by_number(run, 'aien-dev/repo', 17)
        self.assertEqual((result['number'], result['headRefOid'], result['baseRefOid']), (17, HEAD, BASE))
        self.assertIn('api', calls[0])
        self.assertIn('graphql', calls[0])

    def test_nonzero_transport_and_null_readback_fail_closed(self):
        def failed(cmd, **kwargs):
            return 7, '', 'network unavailable'
        with self.assertRaises(ValueError):
            transport.list_by_head(failed, 'aien-dev/repo', 'feat/topic')
        with self.assertRaises(ValueError):
            transport.view_by_number(lambda *a, **k: (0, '{"data":{"repository":{"pullRequest":null}}}', ''),
                                     'aien-dev/repo', 17)


if __name__ == '__main__':
    unittest.main()
