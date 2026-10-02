import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cfg=read(ROOT/'config/protocol.json');a=read(ROOT/'data/prepared/audit.json');frozen=read(ROOT/'outputs/frozen_experiment.json')
    assert a['protocol_sha256']==frozen['protocol_sha256']==sha(ROOT/'config/protocol.json')
    assert frozen['source_sha256']==sha(ROOT/'src/study.py')
    assert frozen['selection_sha256']==sha(ROOT/'outputs/selection.json')
    for filename,key in [('train_val.npz','train_val_sha256'),('test.npz','test_sha256')]:assert sha(ROOT/'data/prepared'/filename)==a[key]
    with np.load(ROOT/'data/prepared/train_val.npz') as d:
        train=set(d['train_ids'].tolist());val=set(d['val_ids'].tolist());perm=d['permutation']
    assert not train&val and len(perm)==784 and np.array_equal(np.sort(perm),np.arange(784))
    parent=np.arange(70000)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return int(i)
    for row in csv.DictReader((ROOT/'reports/data_audit/detected_near_pairs.csv').open()):
        x,y=find(int(row['combined_index_a'])),find(int(row['combined_index_b']));parent[max(x,y)]=min(x,y)
    tg={find(i) for i in train};vg={find(i) for i in val};testg={find(i) for i in range(60000,70000)}
    assert not(tg&vg or tg&testg or vg&testg)
    with np.load(ROOT/'data/prepared/test.npz') as d:labels=d['y_test']
    expected=len(cfg['models'])*len(cfg['conditions'])*len(cfg['seeds']);assert len(frozen['checkpoints'])==expected
    details=[]
    for run,checksum in frozen['checkpoints'].items():
        path=ROOT/'outputs/runs'/run;r=read(path/'test.json');assert sha(path/'model.pt')==checksum
        pred=np.load(path/'test_predictions.npy');assert pred.shape==labels.shape and np.all((pred>=0)&(pred<10))
        cm=np.bincount(labels.astype(int)*10+pred.astype(int),minlength=100).reshape(10,10)
        acc=float((pred==labels).mean());den=cm.sum(0)+cm.sum(1);f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(10,float),where=den!=0).mean())
        assert abs(acc-r['test_accuracy'])<1e-12 and abs(f1-r['test_macro_f1'])<1e-12
        assert np.array_equal(cm,r['confusion_matrix'])
        assert r['evaluated_utc']>=frozen['created_utc']
        assert r['source_sha256']==frozen['source_sha256'] and r['protocol_sha256']==frozen['protocol_sha256']
        details.append({'run':run,'accuracy':acc,'macro_f1':f1,'checkpoint_sha256':checksum})
    result={'status':'passed','runs_verified':expected,'checkpoint_hashes_verified':expected,'metrics_recomputed_from_predictions':True,'detected_duplicate_groups_disjoint':True,'permutation_bijective':True,'corrected_evaluation_after_freeze':True,'protocol_version':cfg['protocol_version'],'note':'Corrected-run freeze verified; original v1 test results had already been inspected.','runs':details}
    (ROOT/'reports/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
