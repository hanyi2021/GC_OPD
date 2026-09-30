"""Prepare native graph-free evaluation with explicit seed and identity manifests."""
import argparse
import json
import os
from pathlib import Path
import sys
import yaml
from gcopd.common.io import ROOT, PYTHON_PATHS, new_output, write_json, digest, require_backend
from gcopd.scienceworld.prepare_data import rows
from gcopd.scienceworld.train import set_nested
from gcopd.common.models import check_model_config

def recipe_configuration(recipe, model, tasks, output, ray_address):
    """Resolve the student recipe without requiring training or teacher assets."""
    from omegaconf import OmegaConf
    config=yaml.safe_load((ROOT/'gcopd/scienceworld/configs'/f'{recipe}_opd.yaml').read_text())
    env={'PYTHONPATH':os.pathsep.join(map(str,PYTHON_PATHS)),
         'GCOPD_STUDENT_MODEL':str(model),'GCOPD_TEACHER_MODEL':str(model),
         'GCOPD_TRAIN_PARQUET':str(tasks),'GCOPD_DEV_PARQUET':str(tasks),
         'GCOPD_AGENT_CONFIG':str(ROOT/'gcopd/scienceworld/configs/agent.yaml'),'GCOPD_RUN_DIR':str(output),
         'GCOPD_METRICS_FILE':str(output/'metrics.jsonl'),'GCOPD_EVIDENCE_CODE':str(ROOT/'gcopd/scienceworld'),
         'GCOPD_GRAPH_CATALOG':'','GCOPD_CHECKPOINT_OUTPUT':str(output/'unused-checkpoints'),
         'GCOPD_PARENT_CHECKPOINT':'','RAY_ADDRESS':ray_address,
         'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'}
    previous={key:os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        config=OmegaConf.to_container(OmegaConf.create(config),resolve=True)
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    env.update({key:str(value) for key,value in config['ray_kwargs']['ray_init']['runtime_env']['env_vars'].items()})
    return config,env

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source_config=p.add_mutually_exclusive_group(required=True)
    source_config.add_argument('--template',help='PREPARED_RUN.json from a training preparation')
    source_config.add_argument('--recipe',choices=['scienceworld_1p7b','scienceworld_4b'],
        help='Evaluate a released student directly, without training or teacher assets')
    p.add_argument('--model',required=True,help='Selected checkpoint actor/huggingface folder')
    p.add_argument('--tasks',required=True,help='Unpadded dev.parquet or test.parquet')
    p.add_argument('--rep',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpus',type=int,default=8);p.add_argument('--nodes',type=int,default=1)
    p.add_argument('--ray-address',default='local');p.add_argument('--skip-asset-check',action='store_true')
    a=p.parse_args()
    require_backend()
    if a.rep<0 or a.gpus<1 or a.nodes<1:raise ValueError('Invalid seed/resource allocation')
    import pyarrow as pa
    import pyarrow.parquet as pq
    source=pq.read_table(a.tasks).to_pylist()
    if any(r['extra_info'].get('padding',False) for r in source):raise ValueError('Evaluation source must be unpadded')
    ids=[r['extra_info']['gamefile'] for r in source]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('Evaluation identities must be unique')
    model=Path(a.model).expanduser().resolve()
    if not a.skip_asset_check and not (model/'config.json').is_file():raise ValueError('Model folder missing HF config.json')
    model_compatibility=None if a.skip_asset_check else check_model_config(model)
    out=new_output(a.output);world=a.gpus*a.nodes
    if a.template:
        spec=json.loads(Path(a.template).read_text())
        config=yaml.safe_load(Path(spec['config']).read_text());env=dict(spec['environment'])
    else:
        config,env=recipe_configuration(a.recipe,model,Path(a.tasks).resolve(),out,a.ray_address)
    data=rows(ids,pad_to=world,rep=a.rep);taskfile=out/'tasks.parquet';pq.write_table(pa.Table.from_pylist(data),taskfile)
    settings={'actor_rollout_ref.model.path':str(model),'data.train_files':[str(taskfile)],'data.val_files':[str(taskfile)],
        'data.train_batch_size':world,'data.val_batch_size':world,'actor_rollout_ref.actor.ppo_mini_batch_size':world,
        'trainer.val_only':True,'trainer.val_before_train':True,'trainer.resume_mode':'disable','trainer.resume_from_path':None,
        'trainer.data_epoch_global_step_offset':0,
        'trainer.nnodes':a.nodes,'trainer.n_gpus_per_node':a.gpus,'trainer.default_local_dir':str(out/'unused-checkpoints'),
        'trainer.validation_data_dir':str(out/'generations'),'distillation.enabled':False,
        'actor_rollout_ref.rollout.n':1,'actor_rollout_ref.rollout.tensor_model_parallel_size':1,
        'actor_rollout_ref.rollout.agent.num_workers':world,
        'actor_rollout_ref.rollout.gpu_memory_utilization':0.8,
        'actor_rollout_ref.rollout.val_kwargs.temperature':0.4,'actor_rollout_ref.rollout.val_kwargs.do_sample':True,
        'ray_kwargs.ray_init.address':a.ray_address}
    for k,v in settings.items():set_nested(config,k,v)
    env.update(PYTHONPATH=os.pathsep.join(map(str,PYTHON_PATHS)),SW_RUN_DIR=str(out),
        VERL_FILE_LOGGER_PATH=str(out/'metrics.jsonl'),RAY_ADDRESS=a.ray_address)
    config['ray_kwargs']['ray_init']['runtime_env']['env_vars'].update(env)
    cfgdir=out/'config';cfgdir.mkdir();cfg=cfgdir/'experiment.yaml';cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    write_json(out/'EVALUATION_TASKS.json',{'task_ids':ids,'rep':a.rep,'task_count':len(ids),'padded_rows':len(data),
        'source_sha256':digest(a.tasks),'model':str(model),'temperature':0.4,'history':5,'max_decisions':30})
    write_json(out/'PREPARED_RUN.json',{'root':str(ROOT),'command':[sys.executable,'-m','verl.trainer.main_ppo','--config-path',str(cfgdir),'--config-name','experiment'],
        'environment':env,'config':str(cfg),'config_sha256':digest(cfg),'asset_validation_skipped':a.skip_asset_check,
        'model_compatibility':model_compatibility,
        'resources':{'student_rollout_gpus':world,'scoring_teacher_gpus':0,'evaluation_total_gpus':world},
        'scope':'Prepared student evaluation configuration.'})
    print(out/'PREPARED_RUN.json')

if __name__=='__main__':main()
