import importlib.util, tempfile, unittest, json
from pathlib import Path
spec=importlib.util.spec_from_file_location('board',Path(__file__).with_name('renda_board.py')); b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class BoardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.q=self.root/'renda-queue';self.q.mkdir()
  self.entry={'key':'aien-dev/omega-mv:ready:x@'+'a'*40,'kind':'ready','sha':'a'*40,'base':'b'*40,'ref':'x','number':None,'provenance':'synthetic test','remote_observed_at':None}
  self.save(self.q/'omega.json',{'schema':'renda-local-queue/1','repository':'aien-dev/omega-mv','entries':[self.entry]})
 def save(self,p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
 def plan(self,**kw):
  import hashlib
  d={'schema':'renda-local-plan/1','key':self.entry['key'],'commit':'a'*40,'base':'b'*40,'review':{'decision':True},'receipt':{'verified':True,'eligible':True},'plan':{'code':0},'created_at':1};d.update(kw)
  self.save(self.q/'local/omega'/(hashlib.sha256(self.entry['key'].encode()).hexdigest()+'.plan.json'),d)
 def test_cold_ready_is_actionable_review_not_merge(self):
  rows,w=b.collect(self.root);self.assertEqual(rows[0]['stage'],'Actionable');self.assertIn('review',rows[0]['next_action']);self.assertFalse(rows[0]['merge_authorized'])
 def test_pass_is_publication_pending_never_merge_authority(self):
  self.plan();r,_=b.collect(self.root);self.assertEqual(r[0]['stage'],'Awaiting publication');self.assertFalse(r[0]['merge_authorized'])
 def test_failed_receipt_overrides_review(self):
  self.plan(receipt={'verified':True,'eligible':False,'verdict':'FAIL'});r,_=b.collect(self.root);self.assertEqual(r[0]['stage'],'Blocked')
 def test_stale_plan_is_not_reused(self):
  self.plan(commit='c'*40);r,w=b.collect(self.root);self.assertEqual(r[0]['stage'],'Actionable');self.assertTrue(w)
 def test_corrupt_queue_visible(self):
  (self.q/'omega.json').write_text('{');r,w=b.collect(self.root);self.assertEqual(r,[]);self.assertTrue(w)
 def test_wrong_plan_key_ignored(self):
  self.plan(key='other');r,w=b.collect(self.root);self.assertEqual(r[0]['stage'],'Actionable');self.assertTrue(w)
 def test_partial_hash_untriaged(self):
  self.entry['sha']='abcd';self.save(self.q/'omega.json',{'schema':'renda-local-queue/1','repository':'aien-dev/omega-mv','entries':[self.entry]});r,w=b.collect(self.root);self.assertEqual(r[0]['stage'],'Untriaged')
 def test_board_read_only(self):
  before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()};b.collect(self.root);self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
 def test_duplicates_not_silent(self):
  self.save(self.q/'omega.json',{'schema':'renda-local-queue/1','repository':'aien-dev/omega-mv','entries':[self.entry,self.entry]});r,w=b.collect(self.root);self.assertEqual(len(r),1);self.assertTrue(w)
 def test_table_escapes_control_text(self):
  self.entry['ref']='x|y\nz';self.save(self.q/'omega.json',{'schema':'renda-local-queue/1','repository':'aien-dev/omega-mv','entries':[self.entry]});r,w=b.collect(self.root);s=b.render(r,w);self.assertIn('x\\|y z',s)
if __name__=='__main__':unittest.main()
