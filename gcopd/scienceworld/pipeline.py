"""Run the full ScienceWorld workflow through the shipped production entrypoints."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from gcopd.common.io import ROOT, digest, new_output, write_json, task_ids, require_backend
from gcopd.common.resources import training_resources


def run_pipeline(config_path, output, execute=False):
    if execute:require_backend()
    source=Path(config_path).expanduser().resolve()
    cfg=json.loads(source.read_text())
    for key in ('train_tasks','dev_tasks','test_tasks','student_model','teacher_model'):
        p=Path(cfg[key]).expanduser()
        cfg[key]=str((source.parent/p).resolve() if not p.is_absolute() else p.resolve())
    recipe=cfg.get('recipe','scienceworld_4b')
    if recipe not in ('scienceworld_4b','scienceworld_1p7b'):raise ValueError('Unknown ScienceWorld configuration')
    resource_fields=dict(student_nodes=cfg.get('student_nodes',1),
        student_gpus_per_node=cfg.get('student_gpus_per_node',4),
        teacher_nodes=cfg.get('teacher_nodes',1),teacher_gpus_per_node=cfg.get('teacher_gpus_per_node',4),
        teacher_tp=cfg.get('teacher_tp',1),total_gpus=cfg.get('total_gpus'))
    resources=training_resources(**resource_fields)
    collection_gpus=cfg.get('collection_service_gpus')
    if type(collection_gpus) is not int or collection_gpus<1:
        raise ValueError('Declare positive collection_service_gpus for the separately managed source endpoint')
    eval_gpus=cfg.get('eval_gpus',8);eval_nodes=cfg.get('eval_nodes',1)
    if any(type(v) is not int or v<1 for v in [eval_gpus,eval_nodes]):raise ValueError('Invalid evaluation resources')
    resources.update(collection_service_gpus=collection_gpus,evaluation_student_gpus=eval_gpus*eval_nodes,
        total_if_source_service_remains_running=resources['training_total_gpus']+collection_gpus,
        source_service_lifecycle='Externally managed; stop it after collection or reserve its GPUs in addition to training.')
    steps=math.ceil(len(task_ids(cfg['train_tasks']))/32)
    out=new_output(output)
    progress={'configuration_sha256':digest(source),'recipe':recipe,'executed':execute,'stages':[],
        'scope':'ScienceWorld source collection, graph construction, training and evaluation workflow.',
        'resources':resources,'schedule':{'opd_epochs':1,'gc_epochs':1,'updates_per_epoch':steps}}
    def command(name,args):
        cmd=[sys.executable,'-m','gcopd.scienceworld.'+Path(args[0]).stem,*map(str,args[1:])]
        stage={'name':name,'command':cmd,'status':'planned'}
        progress['stages'].append(stage);write_json(out/'PIPELINE.json',progress)
        if execute:
            stage['status']='running';write_json(out/'PIPELINE.json',progress)
            code=subprocess.call(cmd,cwd=ROOT)
            stage['status']='complete' if code==0 else 'failed';stage['exit_code']=code
            write_json(out/'PIPELINE.json',progress)
            if code:raise RuntimeError('Pipeline stopped at '+name+'; inspect its output before resuming manually')
        return cmd
    data=out/'data';catalog=out/'catalog';sources=out/'sources';opd=out/'opd';gc=out/'gc'
    command('prepare_data',['prepare_data.py','--train',cfg['train_tasks'],'--dev',cfg['dev_tasks'],'--test',cfg['test_tasks'],'--output',data])
    command('collect_sources',['collect.py','--train-tasks',cfg['train_tasks'],'--endpoint',cfg['teacher_endpoint'],'--served-model',cfg.get('served_model','teacher'),'--model-path',cfg['teacher_model'],'--workers',cfg.get('collection_workers',4),'--output',sources])
    command('build_catalog',['build_graph.py','--train-tasks',cfg['train_tasks'],'--source-results',sources/'results','--workers',cfg.get('replay_workers',4),'--output',catalog])
    if cfg.get('planner_augmentation',False):
        augmented=out/'augmented_catalog'
        command('planner_augmentation',['augment_graph.py','--catalog',catalog,'--train-tasks',cfg['train_tasks'],'--output',augmented,'--workers',cfg.get('replay_workers',4)])
        catalog=augmented
    assets=out/'assets.json'
    write_json(assets,{'student_model':cfg['student_model'],'teacher_model':cfg['teacher_model'],'train_parquet':str(data/'train.parquet'),'dev_parquet':str(data/'dev.parquet'),'graph_catalog':str(catalog)})
    resource=['--ray-address',cfg.get('ray_address','local'),'--save-freq',cfg.get('save_freq',10)]
    for key,value in resource_fields.items():
        if value is not None:resource.extend(['--'+key.replace('_','-'),value])
    command('prepare_opd',['train.py','prepare','--phase','opd','--recipe',recipe,'--assets',assets,'--output',opd,*resource])
    command('train_opd',['train.py','execute',opd/'PREPARED_RUN.json'])
    parent=opd/'checkpoints'/f'global_step_{steps}'
    evidence_args=['--hindsight-only'] if cfg.get('hindsight_only',False) else []
    if recipe=='scienceworld_1p7b' and cfg.get('planner_augmentation',False) and not cfg.get('hindsight_only',False):
        evidence_args.append('--terminal-failure-note')
    command('prepare_gc',['train.py','prepare','--phase','gc','--recipe',recipe,'--assets',assets,'--parent',parent,'--output',gc,*resource,*evidence_args])
    command('train_gc',['train.py','execute',gc/'PREPARED_RUN.json'])
    # New executions select only from their own development scores.
    if execute:
        checkpoints=sorted((gc/'checkpoints').glob('global_step_*'),key=lambda x:int(x.name.rsplit('_',1)[1]))
        if not checkpoints:raise RuntimeError('No saved graph checkpoints to validate')
    else:
        checkpoints=[gc/'checkpoints'/'global_step_<each_saved_candidate>']
    summaries=[]
    for ckpt in checkpoints:
        job=out/('dev_'+ckpt.name);summary=out/(job.name+'_summary.json');summaries.append(summary)
        command('prepare_'+job.name,['evaluate.py','--template',gc/'PREPARED_RUN.json','--model',ckpt/'actor/huggingface','--tasks',data/'dev.parquet','--rep',0,'--output',job,'--gpus',eval_gpus,'--nodes',eval_nodes,'--ray-address',cfg.get('ray_address','local')])
        command('evaluate_'+job.name,['train.py','execute',job/'PREPARED_RUN.json'])
        command('summarize_'+job.name,['summarize.py',job,'--output',summary])
    selection=out/'SELECTION.json'
    command('select_gc',['select_checkpoint.py','--dev-tasks',cfg['dev_tasks'],'--summaries',*summaries,'--output',selection])
    model=json.loads(selection.read_text())['selected']['model'] if execute else str(gc/'checkpoints'/'global_step_<dev_selected>'/'actor/huggingface')
    tests=[]
    for rep in range(4):
        job=out/f'test_rep{rep}';tests.append(job)
        command('prepare_'+job.name,['evaluate.py','--template',gc/'PREPARED_RUN.json','--model',model,'--tasks',data/'test.parquet','--rep',rep,'--output',job,'--gpus',eval_gpus,'--nodes',eval_nodes,'--ray-address',cfg.get('ray_address','local')])
        command('evaluate_'+job.name,['train.py','execute',job/'PREPARED_RUN.json'])
    command('summarize_test',['summarize.py',*tests,'--output',out/'TEST_SUMMARY.json'])
    print(out/'PIPELINE.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true',help='Actually run collection, GPU training, dev selection and four-seed test')
    a=p.parse_args();run_pipeline(a.config,a.output,a.execute)
