"""Prepare or execute ALFWorld/WebShop OPD and graph-conditioned continuation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import yaml

from gcopd.common.io import ROOT
from scripts.setup_backend import require_backend, profile_paths
from gcopd.common.resources import training_resources, validate_teacher_inference


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def training_tasks(environment,path):
    if environment=='alfworld':
        from gcopd.alfworld.tasks import load_tasks
        tasks=load_tasks(path)
        if any(t['split']!='train' for t in tasks):raise ValueError('Training requires tasks labelled train')
        return tasks
    from gcopd.webshop.collect import load_goals
    return [{'gamefile':f'goal{g}','goal':g,'task_index':i,'split':'train'} for i,g in enumerate(load_goals(path))]


def make_rows(environment,tasks):
    rows=[];total=math.ceil(len(tasks)/32)*32
    for i in range(total):
        padding=i>=len(tasks)
        extra={'gamefile':f'__padding__{i}','task_index':i,'split':'train'} if padding else dict(tasks[i])
        extra.update(index=i,padding=padding)
        rows.append({'data_source':environment,'agent_name':environment,'prompt':[{'role':'user','content':f'{environment} task '+extra['gamefile']}],
                     'reward_model':{'style':'rule','ground_truth':''},'extra_info':extra})
    return rows


def verify_models(student,teacher,environment):
    from tokenizers import Tokenizer
    for path in [student,teacher]:
        for name in ['config.json','tokenizer.json','tokenizer_config.json']:
            if not (path/name).is_file():raise ValueError(f'Missing model asset: {path/name}')
    s=Tokenizer.from_file(str(student/'tokenizer.json'));t=Tokenizer.from_file(str(teacher/'tokenizer.json'))
    student_vocab,teacher_vocab=s.get_vocab(),t.get_vocab()
    if environment=='webshop':
        if any(teacher_vocab.get(token)!=token_id for token,token_id in student_vocab.items()):
            raise ValueError('Every student token must have the same teacher token ID')
    elif student_vocab!=teacher_vocab:
        raise ValueError('Student/teacher token IDs must align')
    sj,tj=json.loads(s.to_str()),json.loads(t.to_str())
    tokenizer_core_keys=(sorted((set(sj)|set(tj))-{'added_tokens'}) if environment=='webshop'
                         else ['model','normalizer','pre_tokenizer','decoder'])
    for key in tokenizer_core_keys:
        if sj.get(key)!=tj.get(key):raise ValueError(f'Tokenizer {key} differs')
    if environment=='webshop':
        student_added={item['id']:item for item in sj.get('added_tokens',[])}
        teacher_added={item['id']:item for item in tj.get('added_tokens',[])}
        if any(teacher_added.get(token_id)!=item for token_id,item in student_added.items()):
            raise ValueError('Shared added-token definitions differ')
        extra=[item for token_id,item in teacher_added.items() if token_id not in student_added]
        if any(item.get('special') is not True or item['content'] in student_vocab for item in extra):
            raise ValueError('Only teacher-only additional special tokens may differ')
        if set(teacher_vocab)-set(student_vocab)!={item['content'] for item in extra}:
            raise ValueError('Teacher vocabulary extensions do not match its additional special tokens')
    elif sj.get('added_tokens')!=tj.get('added_tokens'):
        raise ValueError('Tokenizer added_tokens differs')
    if environment=='alfworld':
        from gcopd.common.models import check_model_config
        check_model_config(student);check_model_config(teacher)
    else:
        import transformers
        if not hasattr(transformers,'Qwen3_5ForConditionalGeneration'):
            raise ValueError('WebShop Qwen3.5 requires a compatible Transformers runtime; see README.md')
        for path in [student,teacher]:
            if json.loads((path/'config.json').read_text())['model_type']!='qwen3_5':
                raise ValueError('This WebShop recipe requires qwen3_5 model configuration')


def prepare(a):
    require_backend(ROOT,a.environment)
    tasks=training_tasks(a.environment,a.tasks);steps=math.ceil(len(tasks)/32)
    if a.save_freq<1:raise ValueError('save-freq must be positive')
    resources=training_resources(a.student_nodes,a.student_gpus,a.teacher_nodes,a.teacher_gpus,a.teacher_tp,None)
    student=Path(a.student_model).expanduser().resolve();teacher=Path(a.teacher_model).expanduser().resolve()
    if not a.skip_asset_check:verify_models(student,teacher,a.environment)
    if a.parent and a.phase!='gc':raise ValueError('--parent is only valid for GC continuation')
    if a.environment=='alfworld':
        if not a.data_root or not a.env_python:raise ValueError('ALFWorld requires --data-root and --env-python')
        if not a.skip_asset_check:
            from gcopd.alfworld.tasks import resolve_gamefile
            for task in tasks:resolve_gamefile(a.data_root,task['gamefile'])
            if not Path(a.env_python).is_file():raise ValueError('Environment Python does not exist')
    elif not a.env_url:raise ValueError('WebShop requires --env-url for its state/replay server')
    elif not a.skip_asset_check:
        from gcopd.webshop.collect import http
        health=http(a.env_url.rstrip('/')+'/health')
        if not health.get('ok') or health.get('seed')!=42 or health.get('goal_shuffle_seed')!=233 or health.get('observation_mode')!='text_rich':
            raise ValueError('WebShop server does not match the main environment protocol')
    catalog=Path(a.catalog).expanduser().resolve() if a.catalog else None
    if a.phase=='gc' and not catalog:raise ValueError('GC requires --catalog')
    if a.phase=='gc' and not a.skip_asset_check:
        if not (catalog/'READY.json').is_file():raise ValueError('Catalog build is incomplete')
        manifest=json.loads((catalog/'MANIFEST.json').read_text())
        ids=[t['gamefile'] for t in manifest['tasks']]
        if len(ids)!=len(set(ids)) or set(ids)!={t['gamefile'] for t in tasks}:
            raise ValueError('Catalog and training task identities differ')
    parent=None;offset=0
    if a.phase=='gc':
        if not a.parent:raise ValueError('GC requires a full OPD parent checkpoint')
        parent=Path(a.parent).expanduser().resolve();m=re.fullmatch(r'global_step_(\d+)',parent.name)
        if not m:raise ValueError('Parent must be named global_step_N')
        offset=int(m[1])
        if offset!=steps:raise ValueError('Use the last checkpoint of one complete OPD epoch on this task set')
        if not a.skip_asset_check:
            world=resources['student_rollout_gpus']
            for kind in ['model','optim','extra_state']:
                shards=list((parent/'actor').glob(f'{kind}_world_size_{world}_rank_*.pt'))
                if {x.name for x in shards}!={f'{kind}_world_size_{world}_rank_{i}.pt' for i in range(world)}:raise ValueError('Parent must retain every model/optimizer/RNG shard for the same student world size')
            binding=parent.parent.parent/'PREPARED_RUN.json'
            if not binding.is_file():raise ValueError('Keep the OPD PREPARED_RUN.json beside its checkpoints')
            previous=json.loads(binding.read_text())
            if previous['environment']!=a.environment or previous['phase']!='opd' or previous['task_manifest_sha256']!=digest(a.tasks):
                raise ValueError('Parent belongs to a different environment, phase or task manifest')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new or empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    data=out/'train.parquet';pq.write_table(pa.Table.from_pylist(make_rows(a.environment,tasks)),data)
    backend=profile_paths(ROOT,a.environment)[0]
    env={'PYTHONPATH':os.pathsep.join(map(str,[ROOT,backend])),
         'GCOPD_STUDENT_MODEL':str(student),'GCOPD_TEACHER_MODEL':str(teacher),'GCOPD_TRAIN_PARQUET':str(data),
         'GCOPD_AGENT_CONFIG':str(ROOT/'gcopd'/a.environment/'configs/agent.yaml'),'GCOPD_CHECKPOINT_OUTPUT':str(out/'checkpoints'),
         'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false','VERL_FILE_LOGGER_PATH':str(out/'metrics.jsonl')}
    prefix='SW' if a.environment=='alfworld' else 'WS'
    prompt_budget,model_budget=(40000,40513) if a.environment=='alfworld' else (260095,262144)
    env.update({prefix+'_RUN_DIR':str(out),prefix+'_POSTHOC_REFERENCE_MODULE':f'gcopd.{a.environment}.references.'+('render' if a.phase=='gc' else 'plain'),
                prefix+'_PHYSICAL_GRAPH_ROOT':str(catalog or ''),prefix+'_TEACHER_PROMPT_BUDGET':str(prompt_budget),prefix+'_TEACHER_MAX_MODEL_LEN':str(model_budget)})
    if a.environment=='alfworld':env.update(ALF_ENV_PYTHON=os.path.abspath(os.path.expanduser(a.env_python)),ALF_DATA_ROOT=str(Path(a.data_root).expanduser().resolve()),ALF_MAX_DECISIONS='30')
    else:env.update(WS_ENV_URL=a.env_url.rstrip('/'),WS_HISTORY='2',WS_MAX_DECISIONS='15',WS_MAX_PROMPT_TOKENS='63488',WS_MAX_RESPONSE_TOKENS='2048',WS_REPETITION_PENALTY='1.0',WS_SOURCE_RULE='binary',WS_FAILED_CONTRAST='1',WS_REFERENCE_DEDUP='0',WS_MODERN_RESPONSE_LOGITS_ONLY='1',WS_MODERN_LONG_INPUT_SAFE='1')
    config=yaml.safe_load((ROOT/'gcopd'/a.environment/'configs/train.yaml').read_text())
    config['trainer'].update(total_training_steps=offset+steps,total_epochs=1,data_epoch_global_step_offset=offset,save_freq=a.save_freq,
                             resume_mode='resume_path' if parent else 'disable',resume_from_path=str(parent) if parent else None,
                             nnodes=a.student_nodes,n_gpus_per_node=a.student_gpus)
    config['distillation'].update(nnodes=a.teacher_nodes,n_gpus_per_node=a.teacher_gpus)
    teacher_config=config['distillation']['teacher_models']['teacher_model'];teacher_config['num_replicas']=resources['scoring_teacher_replicas'];teacher_config['inference']['tensor_model_parallel_size']=a.teacher_tp
    validate_teacher_inference(teacher_config['inference'])
    rollout_tp=config['actor_rollout_ref']['rollout']['tensor_model_parallel_size']
    if resources['student_rollout_gpus']%rollout_tp:raise ValueError('Student GPUs must be divisible by rollout TP')
    config['ray_kwargs']['ray_init'].update(address=a.ray_address,runtime_env={'env_vars':env})
    from omegaconf import OmegaConf
    before=dict(os.environ)
    try:
        os.environ.update(env);resolved=OmegaConf.to_container(OmegaConf.create(config),resolve=True)
    finally:
        for key in env:
            if key in before:os.environ[key]=before[key]
            else:os.environ.pop(key,None)
    config_dir=out/'config';config_dir.mkdir();cfg=config_dir/'experiment.yaml';cfg.write_text(yaml.safe_dump(resolved,sort_keys=False))
    command=[sys.executable,'-m','verl.trainer.main_ppo','--config-path',str(config_dir),'--config-name','experiment']
    spec={'environment':a.environment,'phase':a.phase,'root':str(ROOT),'config':str(cfg),'config_sha256':digest(cfg),
          'task_manifest_sha256':digest(a.tasks),'tasks':len(tasks),'updates_in_phase':steps,'parent':str(parent) if parent else None,
          'parent_step':offset,'target_step':offset+steps,'resources':resources,'runtime_environment':env,'command':command,
          'asset_validation_skipped':a.skip_asset_check,'executed':False}
    (out/'PREPARED_RUN.json').write_text(json.dumps(spec,indent=2));print(out/'PREPARED_RUN.json')


def execute(path, expected_environment=None):
    p=Path(path).resolve();spec=json.loads(p.read_text())
    if expected_environment and spec['environment']!=expected_environment:raise ValueError('Prepared run belongs to another environment')
    if spec['asset_validation_skipped']:raise ValueError('Cannot execute an unchecked configuration; prepare with actual assets')
    if Path(spec['root'])!=ROOT or digest(spec['config'])!=spec['config_sha256']:raise ValueError('Prepared code root/configuration changed')
    require_backend(ROOT,spec['environment'])
    env=dict(os.environ,**spec['runtime_environment'])
    result=subprocess.run(spec['command'],cwd=ROOT,env=env)
    spec.update(executed=True,exit_code=result.returncode);p.write_text(json.dumps(spec,indent=2))
    if result.returncode:raise SystemExit(result.returncode)


def main(environment=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    q.add_argument('--environment',choices=['alfworld','webshop'],required=environment is None,default=environment)
    q.add_argument('--phase',choices=['opd','gc'],required=True)
    for name in ['tasks','student-model','teacher-model','output']:q.add_argument('--'+name,required=True)
    for name in ['catalog','parent','data-root','env-python','env-url']:q.add_argument('--'+name)
    for name,value in [('student-nodes',1),('student-gpus',4),('teacher-nodes',1),('teacher-gpus',4),('teacher-tp',1),('save-freq',10)]:q.add_argument('--'+name,type=int,default=value)
    q.add_argument('--ray-address',default='local');q.add_argument('--skip-asset-check',action='store_true')
    q=sub.add_parser('execute');q.add_argument('prepared_run')
    a=p.parse_args()
    if a.command=='prepare' and environment and a.environment!=environment:raise ValueError('Environment does not match this entrypoint')
    if a.command=='prepare':prepare(a)
    else:execute(a.prepared_run,expected_environment=environment)


if __name__=='__main__':main()
