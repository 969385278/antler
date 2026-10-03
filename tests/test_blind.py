"""C9：抽样、盲态、原始身份和7天复标约束；使用纯CPU合成评分。"""
import csv
from datetime import datetime, timedelta, timezone
import json

import pytest

from antler.eval import blind


@pytest.fixture
def source(tmp_path):
    records=[]
    for dim in ['O','C','E','A','N']:
        for coef in [-.2,0,.2]:
            records.append({'dim':dim,'pole':'high','coef':coef,'layer':16,'split':'calib','seed':0,
                'result':{'samples':[{'q':f'{dim}/{coef}/q{i}','a':f'合成回答{i}',
                                     'trait':i*10,'coherence':100-i*10} for i in range(10)]}})
    path=tmp_path/'source.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
    return path


def fill(path, key, missing=False):
    with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for m in blind.METRICS:r[m+'_human']=key[r['id']]['judge'][m]
    if missing:rows[0]['trait_human']=''
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=blind.FIELDS);w.writeheader();w.writerows(reversed(rows))


def test_strata_blinding_repeatable_and_no_overwrite(source,tmp_path):
    a=blind.export_first([source],tmp_path/'a',4)
    b=blind.export_first([source],tmp_path/'b',4)
    assert a['samples']==b['samples'] and a['repeat_ids']==b['repeat_ids']
    assert len(a['samples'])==100 and len(a['repeat_ids'])==20
    assert {r['n'] for r in a['strata_counts']}=={6,7}
    with (tmp_path/'a/blind.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert list(rows[0])==blind.FIELDS and all(r['trait_human']==r['coherence_human']=='' for r in rows)
    assert 'coef' not in rows[0] and 'judge' not in rows[0]
    with pytest.raises(FileExistsError):blind.export_first([source],tmp_path/'a')


def test_missing_and_constant_scores_not_validated(source,tmp_path):
    bundle=tmp_path/'b';key=blind.export_first([source],bundle)
    fill(bundle/'blind.csv',key['samples'],missing=True)
    result=blind.import_first(bundle,bundle/'blind.csv')
    assert not result['complete_human'] and not result['human_judge_threshold_met']
    assert not (bundle/'private_first_receipt.json').exists()
    for sample in key['samples'].values():sample['judge']={'trait':50,'coherence':50}
    fill(bundle/'blind.csv',key['samples'])
    result=blind.import_first(bundle,bundle/'blind.csv')
    assert result['complete_human'] and result['agreement']['trait']['spearman'] is None
    assert not result['human_judge_threshold_met']


def test_seven_day_workflow_and_receipt_preserved(source,tmp_path,monkeypatch):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    monkeypatch.setattr(blind,'utcnow',lambda:start)
    bundle=tmp_path/'b';key=blind.export_first([source],bundle)
    fill(bundle/'blind.csv',key['samples'])
    result=blind.import_first(bundle,bundle/'blind.csv')
    assert result['human_judge_threshold_met'] and not result['formal_e0_pass']
    receipt=(bundle/'private_first_receipt.json').read_bytes()
    monkeypatch.setattr(blind,'utcnow',lambda:start+timedelta(days=6,hours=23))
    blind.import_first(bundle,bundle/'blind.csv')
    assert (bundle/'private_first_receipt.json').read_bytes()==receipt
    with pytest.raises(ValueError,match='7天'):blind.export_repeat(bundle,tmp_path/'early')
    monkeypatch.setattr(blind,'utcnow',lambda:start+timedelta(days=7))
    blind.export_repeat(bundle,tmp_path/'r')
    rkey=blind.read_json(tmp_path/'r/private_repeat_key.json')
    assert len(rkey['samples'])==20 and all(i.startswith('R') for i in rkey['samples'])
    with (tmp_path/'r/repeat.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    assert all(r['trait_human']=='' and 'first_id' not in r for r in rows)
    fill(tmp_path/'r/repeat.csv',rkey['samples'])
    result=blind.import_repeat(tmp_path/'r/private_repeat_key.json',tmp_path/'r/repeat.csv')
    assert result['interval_days']==7 and result['test_retest']['trait']['spearman']==pytest.approx(1)
    assert not result['formal_e0_pass']


def test_source_and_key_tampering_rejected(source,tmp_path):
    bundle=tmp_path/'b';key=blind.export_first([source],bundle)
    key['samples']['H001']['judge']['trait']=99
    blind.write_json(bundle/'private_key.json',key)
    with pytest.raises(ValueError,match='原始来源'):blind.import_first(bundle,bundle/'blind.csv')
    source.write_text(source.read_text(encoding='utf-8')+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='哈希'):blind.import_first(bundle,bundle/'blind.csv')


def test_duplicate_private_mapping_rejected(source,tmp_path):
    bundle=tmp_path/'b';key=blind.export_first([source],bundle)
    key['samples']['H002']=key['samples']['H001']
    blind.write_json(bundle/'private_key.json',key)
    with pytest.raises(ValueError,match='重复样本'):blind.import_first(bundle,bundle/'blind.csv')


def test_csv_identity_and_receipt_immutable(source,tmp_path):
    bundle=tmp_path/'b';key=blind.export_first([source],bundle)
    fill(bundle/'blind.csv',key['samples']);blind.import_first(bundle,bundle/'blind.csv')
    for sample in key['samples'].values():sample['judge']['trait']=1
    fill(bundle/'blind.csv',key['samples'])
    with pytest.raises(ValueError,match='封存'):blind.import_first(bundle,bundle/'blind.csv')
    text=(bundle/'blind.csv').read_text(encoding='utf-8').replace('合成回答','篡改回答')
    (bundle/'blind.csv').write_text(text,encoding='utf-8')
    with pytest.raises(ValueError,match='身份变化'):blind.import_first(bundle,bundle/'blind.csv')


def test_duplicate_pool_and_missing_dimension_rejected(source,tmp_path):
    with pytest.raises(ValueError,match='重复'):blind.export_first([source,source],tmp_path/'a')
    records=[json.loads(l) for l in source.read_text(encoding='utf-8').splitlines()]
    source.write_text('\n'.join(json.dumps(r) for r in records if r['dim']!='N'),encoding='utf-8')
    with pytest.raises(ValueError,match='五维'):blind.export_first([source],tmp_path/'b')


def test_c17_adapter_keeps_extraction_separate(tmp_path):
    p=tmp_path/'c17.jsonl'
    p.write_text(json.dumps({'pole':'careless','side':'pos','source_index':2,'question':'q','answer':'a',
                            'trait_score':65,'coherence':80}),encoding='utf-8')
    pool,_=blind.read_pool([p])
    assert pool[0]['stratum']==['C','extract_pos']
    assert pool[0]['judge']=={'trait':65,'coherence':80}
    assert 'coef' not in pool[0]['condition']


def test_repeat_content_tamper_rejected(source,tmp_path,monkeypatch):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    monkeypatch.setattr(blind,'utcnow',lambda:start)
    bundle=tmp_path/'b';key=blind.export_first([source],bundle)
    fill(bundle/'blind.csv',key['samples']);blind.import_first(bundle,bundle/'blind.csv')
    monkeypatch.setattr(blind,'utcnow',lambda:start+timedelta(days=8))
    blind.export_repeat(bundle,tmp_path/'r')
    path=tmp_path/'r/private_repeat_key.json';rkey=blind.read_json(path)
    rkey['samples']['R001']['answer']='篡改';blind.write_json(path,rkey)
    with pytest.raises(ValueError,match='复标内容'):
        blind.import_repeat(path,tmp_path/'r/repeat.csv')


def test_entrypoint_records_metadata_and_never_loads_model(source,tmp_path,monkeypatch):
    import importlib
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    module=importlib.import_module('e0_blind')
    def setup(args,need_judge,need_model):
        assert not need_judge and not need_model
        return {'out_path':tmp_path,'model':'unused','api_key':'SECRET'},None,None,None
    monkeypatch.setattr(module,'setup',setup)
    config=tmp_path/'config.yaml';config.write_text('out_dir: test\n')
    args=SimpleNamespace(config=str(config),action='export',samples=[str(source)],bundle=None,csv=None,repeat_key=None,seed=0)
    out=module.run(args)
    meta=blind.read_json(out/'run_meta.json')
    assert meta['status']=='completed' and meta['data_hashes']
    assert meta['config']['api_key']=='[已隐藏]'
    assert (out/'bundle/blind.csv').exists() and not meta['formal_e0_pass']
