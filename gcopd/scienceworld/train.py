"""Prepare or execute the real ScienceWorld veRL training/validation entrypoint."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import yaml
from gcopd.common.io import ROOT, PYTHON_PATHS, digest, new_output, write_json, require_backend
from gcopd.common.resources import training_resources, validate_teacher_inference
from gcopd.common.models import check_model_config

ASSET_KEYS = ('student_model','teacher_model','train_parquet','dev_parquet','graph_catalog')

def set_nested(config, key, value):
    current = config
    for part in key.split('.')[:-1]:
        current = current.setdefault(part,{})
    current[key.split('.')[-1]] = value

def load_assets(path, check=True, require_graph=True):
    source = Path(path).expanduser().resolve()
    assets = json.loads(source.read_text())
    for key in ASSET_KEYS:
        if key not in assets:
            raise ValueError(f'Missing asset field {key}')
        target = Path(assets[key]).expanduser()
        if not target.is_absolute():
            target = source.parent / target
        assets[key] = str(target.resolve())
        if check and (key != 'graph_catalog' or require_graph) and not target.exists():
            raise ValueError(f'Missing asset {key}: configure a local existing path')
    return assets

def validate_data(assets, require_graph=True):
    import pyarrow.parquet as pq
    train = pq.read_table(assets['train_parquet'],columns=['extra_info']).to_pylist()
    dev = pq.read_table(assets['dev_parquet'],columns=['extra_info']).to_pylist()
    tids = [r['extra_info']['gamefile'] for r in train if not r['extra_info'].get('padding',False)]
    vids = [r['extra_info']['gamefile'] for r in dev]
    if len(tids)!=len(set(tids)) or len(vids)!=len(set(vids)) or set(tids)&set(vids):
        raise ValueError('Duplicate or overlapping train/development task identities')
    if len(train)%32 or len(train)!=math.ceil(len(tids)/32)*32:
        raise ValueError('Training parquet must use 32-row zero-loss padding, without duplicating tasks')
    if {tuple(x.split('|')[:2]) for x in tids}&{tuple(x.split('|')[:2]) for x in vids}:
        raise ValueError('Training/development task-variation overlap under different simplifications')
    if require_graph and 'graph_catalog' in assets:
        manifest=json.loads((Path(assets['graph_catalog'])/'MANIFEST.json').read_text())
        sources=[x['gamefile'] for x in manifest['tasks']]
        if len(sources)!=len(set(sources)) or set(sources)!=set(tids):
            raise ValueError('Graph catalog must retain exactly the entire training task set')
    return {'train_tasks':len(tids),'train_rows':len(train),'dev_tasks':len(vids),
            'updates_per_epoch':math.ceil(len(tids)/32),'train_sha256':digest(assets['train_parquet']),
            'dev_sha256':digest(assets['dev_parquet'])}

def check_tokenizers(assets):
    # This path uses native target token IDs, so identical vocabulary and merges are required.
    for filename in ['tokenizer.json','tokenizer_config.json']:
        for key in ['student_model','teacher_model']:
            if not (Path(assets[key])/filename).is_file():
                raise ValueError(f'{filename} missing in {key}')
    import tokenizers
    student=tokenizers.Tokenizer.from_file(str(Path(assets['student_model'])/'tokenizer.json'))
    teacher=tokenizers.Tokenizer.from_file(str(Path(assets['teacher_model'])/'tokenizer.json'))
    if student.get_vocab()!=teacher.get_vocab():
        raise ValueError('Student and teacher vocab/token IDs differ; this recipe requires aligned Qwen3 token IDs')
    s,t=json.loads(student.to_str()),json.loads(teacher.to_str())
    for key in ['model','normalizer','pre_tokenizer','decoder','added_tokens']:
        if s.get(key)!=t.get(key):
            raise ValueError(f'Student/teacher tokenizer {key} differs')
    return {'vocabulary_equal':True,'tokenizer_model_equal':True,'vocab_size':student.get_vocab_size()}

def prepare(a):
    require_backend()
    require_graph=a.phase=='gc' and not a.hindsight_only
    if a.hindsight_only and a.phase!='gc':raise ValueError('--hindsight-only requires --phase gc')
    if a.terminal_failure_note and (a.phase!='gc' or a.recipe!='scienceworld_1p7b' or a.hindsight_only):
        raise ValueError('--terminal-failure-note requires ScienceWorld 1.7B GC with external references')
    if a.save_freq<1:raise ValueError('--save-freq must be positive')
    assets=load_assets(a.assets,not a.skip_asset_check,require_graph=require_graph)
    facts={'updates_per_epoch':95,'train_tasks':3017,'unchecked_assets':True}
    if not a.skip_asset_check:
        facts=validate_data(assets,require_graph=require_graph)
        facts['tokenizer_alignment']=check_tokenizers(assets)
        facts['model_compatibility']={key:check_model_config(assets[key])
                                      for key in ['student_model','teacher_model']}
    steps=facts['updates_per_epoch']
    phase='gc' if a.phase=='gc' else 'opd'
    recipe=a.recipe
    config=yaml.safe_load((ROOT/'gcopd/scienceworld/configs'/f'{recipe}_{phase}.yaml').read_text())
    # These two disjoint pools match the backend's global_pool and teacher_pool.
    if a.student_nodes is None:a.student_nodes=config['trainer']['nnodes']
    if a.teacher_nodes is None:a.teacher_nodes=config['distillation']['nnodes']
    student_gpus=a.student_gpus_per_node if a.student_gpus_per_node is not None else config['trainer']['n_gpus_per_node']
    teacher_gpus=a.teacher_gpus_per_node if a.teacher_gpus_per_node is not None else config['distillation']['n_gpus_per_node']
    teacher=config['distillation']['teacher_models']['teacher_model']
    validate_teacher_inference(teacher['inference'])
    teacher_tp=a.teacher_tp if a.teacher_tp is not None else teacher['inference']['tensor_model_parallel_size']
    resources=training_resources(a.student_nodes,student_gpus,a.teacher_nodes,teacher_gpus,teacher_tp,a.total_gpus)
    if resources['student_rollout_gpus'] % config['actor_rollout_ref']['rollout']['tensor_model_parallel_size']:
        raise ValueError('Student/rollout pool must be divisible by rollout tensor parallel size')
    parent_step=0
    if phase=='gc':
        if not a.parent:
            raise ValueError('GC phase requires --parent full checkpoint (model, optimizer, scheduler and RNG)')
        parent=Path(a.parent).expanduser().resolve()
        match=re.fullmatch(r'global_step_(\d+)',parent.name)
        if not match: raise ValueError('Parent must be named global_step_N')
        parent_step=int(match.group(1))
        if not a.skip_asset_check and not (parent/'actor').is_dir():
            raise ValueError('Parent actor checkpoint directory missing')
        if not a.skip_asset_check:
            world=resources['student_rollout_gpus']
            required=[f'{kind}_world_size_{world}_rank_*.pt' for kind in ['model','optim','extra_state']]
            for glob in required:
                if len(list((parent/'actor').glob(glob)))!=world:
                    raise ValueError('Parent must contain every model/optimizer/RNG shard at the requested student world size')
    else:
        parent=None
    out=new_output(a.output)
    env={'PYTHONPATH':os.pathsep.join(map(str,PYTHON_PATHS)),
         'GCOPD_STUDENT_MODEL':assets['student_model'],'GCOPD_TEACHER_MODEL':assets['teacher_model'],
         'GCOPD_TRAIN_PARQUET':assets['train_parquet'],'GCOPD_DEV_PARQUET':assets['dev_parquet'],
         'GCOPD_AGENT_CONFIG':str(ROOT/'gcopd/scienceworld/configs/agent.yaml'),'GCOPD_RUN_DIR':str(out),
         'GCOPD_METRICS_FILE':str(out/'metrics.jsonl'),'GCOPD_EVIDENCE_CODE':str(ROOT/'gcopd/scienceworld'),
         'GCOPD_GRAPH_CATALOG':assets['graph_catalog'],'GCOPD_CHECKPOINT_OUTPUT':str(out/'checkpoints'),
         'GCOPD_PARENT_CHECKPOINT':str(parent or ''),'RAY_ADDRESS':a.ray_address,
         'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'}
    set_nested(config,'trainer.total_epochs',2 if a.phase=='opd-baseline' else 1)
    updates=steps*(2 if a.phase=='opd-baseline' else 1)
    set_nested(config,'trainer.total_training_steps',parent_step+updates)
    set_nested(config,'trainer.data_epoch_global_step_offset',parent_step)
    set_nested(config,'trainer.max_actor_ckpt_to_keep',-1)
    # Keep all checkpoints for development selection; no test-set selection.
    set_nested(config,'trainer.save_freq',a.save_freq)
    set_nested(config,'trainer.nnodes',a.student_nodes)
    set_nested(config,'trainer.n_gpus_per_node',student_gpus)
    set_nested(config,'distillation.nnodes',a.teacher_nodes)
    set_nested(config,'distillation.n_gpus_per_node',teacher_gpus)
    set_nested(config,'ray_kwargs.ray_init.runtime_env.env_vars.SW_EXTERNAL_RECORDS','0' if a.hindsight_only else '1')
    set_nested(config,'ray_kwargs.ray_init.runtime_env.env_vars.SW_TERMINAL_FAILURE_NOTE','1' if a.terminal_failure_note else '0')
    teacher['inference']['tensor_model_parallel_size']=teacher_tp
    teacher['num_replicas']=resources['scoring_teacher_replicas']
    from omegaconf import OmegaConf
    previous={k:os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        resolved=OmegaConf.to_container(OmegaConf.create(config),resolve=True)
    finally:
        for k,v in previous.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    cfg_dir=out/'config';cfg_dir.mkdir();cfg=cfg_dir/'experiment.yaml'
    cfg.write_text(yaml.safe_dump(resolved,sort_keys=False))
    env.update({k:str(v) for k,v in resolved['ray_kwargs']['ray_init']['runtime_env']['env_vars'].items()})
    command=[sys.executable,'-m','verl.trainer.main_ppo','--config-path',str(cfg_dir),'--config-name','experiment']
    write_json(out/'PREPARED_RUN.json',{'phase':a.phase,'recipe':recipe,'root':str(ROOT),'assets':assets,'data':facts,'parent':str(parent) if parent else None,
        'parent_step':parent_step,'command':command,'environment':env,'config':str(cfg),'config_sha256':digest(cfg),
        'resources':resources,'hindsight_only':a.hindsight_only,'updates_in_phase':updates,
        'executed':False,'asset_validation_skipped':a.skip_asset_check,'scope':'Prepared ScienceWorld training configuration and launch command.'})
    print(out/'PREPARED_RUN.json')

def execute(path):
    require_backend()
    spec=json.loads(Path(path).read_text())
    if spec.get('asset_validation_skipped'):raise ValueError('Refusing execution of unchecked smoke configuration; prepare again with real assets')
    if digest(spec['config'])!=spec['config_sha256']:raise ValueError('Prepared configuration changed; prepare a new run')
    config=yaml.safe_load(Path(spec['config']).read_text())
    check_model_config(config['actor_rollout_ref']['model']['path'])
    if config.get('distillation',{}).get('enabled'):
        for teacher in config['distillation']['teacher_models'].values():
            check_model_config(teacher['model_path'])
    env=dict(os.environ,**spec['environment'])
    if Path(spec['root']).resolve()!=ROOT:
        raise ValueError('Prepared run belongs to another package location; prepare it again here')
    env['PYTHONPATH']=os.pathsep.join(map(str,PYTHON_PATHS))
    return subprocess.call(spec['command'],env=env,cwd=spec['root'])

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('prepare');q.add_argument('--phase',choices=['opd','gc','opd-baseline'],required=True)
    q.add_argument('--recipe',choices=['scienceworld_1p7b','scienceworld_4b'],default='scienceworld_4b',help='ScienceWorld model configuration')
    q.add_argument('--assets',required=True);q.add_argument('--output',required=True);q.add_argument('--parent')
    q.add_argument('--ray-address',default='local');q.add_argument('--student-nodes',type=int)
    q.add_argument('--teacher-nodes',type=int)
    q.add_argument('--student-gpus-per-node',type=int);q.add_argument('--teacher-gpus-per-node',type=int)
    q.add_argument('--total-gpus',type=int,help='Allocated training GPUs; must cover both disjoint pools')
    q.add_argument('--teacher-tp',type=int);q.add_argument('--save-freq',type=int,default=10)
    q.add_argument('--terminal-failure-note',action='store_true',help='Use the recorded terminal-transition annotation of the ScienceWorld 1.7B augmented recipe')
    q.add_argument('--hindsight-only',action='store_true',help='GC continuation with the same student hindsight renderer and no external records')
    q.add_argument('--skip-asset-check',action='store_true',help='Offline preparation smoke only; execution is disabled')
    q=s.add_parser('execute');q.add_argument('prepared_run')
    a=p.parse_args()
    if a.cmd=='prepare':prepare(a)
    else:raise SystemExit(execute(a.prepared_run))

if __name__=='__main__':main()
