import importlib.util,importlib.machinery,os,unittest,copy
from unittest.mock import patch
p=os.environ['RENDA_CANDIDATE']
spec=importlib.util.spec_from_file_location('renda',p,loader=importlib.machinery.SourceFileLoader('renda',p)); r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class Reliability(unittest.TestCase):
 def test_empty_json_refused(self):
  with patch.object(r,'run',return_value=(0,'','')):
   with self.assertRaises(RuntimeError):r.gh_json(['pr','view','1'])
 def test_null_json_refused(self):
  with patch.object(r,'run',return_value=(0,'null','')):
   with self.assertRaises(RuntimeError):r.gh_json(['pr','view','1'])
 def test_empty_list_valid(self):
  with patch.object(r,'run',return_value=(0,'[]','')):self.assertEqual(r.gh_json(['pr','list']),[])
 def test_missing_block_response_preserves_state(self):
  for val in (None,{},[],{'state':'OPEN'},{'state':'BOGUS'}):
   with self.subTest(val=val):
    st={'open':{'pr':613,'carried':[1,2]}};old=copy.deepcopy(st);res={'waiting':[]}
    with patch.object(r,'gh_json',return_value=val),patch.object(r,'run') as run,patch.object(r,'save_block') as save:
     try:r.finish_block('omega','aien-dev/omega-mv',st,{},res)
     except Exception as e:self.fail('Missing PR response crashed instead of waiting: '+repr(e))
     self.assertEqual(st,old);self.assertTrue(res['waiting']);run.assert_not_called();save.assert_not_called()
 def test_open_block_waits_not_merges(self):
  d={'state':'OPEN','headRefOid':'a'*40,'mergeStateStatus':'BLOCKED'};st={'open':{'pr':613,'carried':[1,2]}};res={'waiting':[]}
  with patch.object(r,'gh_json',return_value=d),patch.object(r,'spark_status',return_value=('pending','')),patch.object(r,'operator_verdict',return_value=None),patch.object(r,'run') as run:
   r.finish_block('omega','aien-dev/omega-mv',st,{},res);self.assertTrue(res['waiting']);run.assert_not_called()
 def test_lane_error_contained(self):
  with patch.object(r,'lane_pass',side_effect=RuntimeError('unreadable PR')):
   fn=getattr(r,'safe_lane_pass',None);self.assertIsNotNone(fn,'No lane isolation; one exception kills all lanes')
   res=fn('omega');self.assertEqual(res['repo'],'omega');self.assertTrue(res['blocked']);self.assertEqual(res['merged'],[])
 def test_successful_lane_unchanged(self):
  value={'repo':'omega','merged':[]}
  with patch.object(r,'lane_pass',return_value=value):
   fn=getattr(r,'safe_lane_pass',None);self.assertIsNotNone(fn);self.assertIs(fn('omega'),value)
if __name__=='__main__':unittest.main(verbosity=2)
