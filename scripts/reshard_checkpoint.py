"""Pair compatible FSDP1 shards into a checkpoint with half the world size.

Model tensors retain FP32 and DTensor global metadata. Adam flat shards are
concatenated only after proving every wrapping unit has zero shard padding.
No model, GPU, process group, or training job is initialized by this script.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import re
import time
import traceback
import numpy as np
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec
import argparse
SOURCE = ROOT = DEST = None
OLD_WORLD = NEW_WORLD = 0
STEP = 0

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save_json(name, data):
    p = ROOT / name
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(p)

def load(kind, rank, root=None, world=None):
    root = SOURCE if root is None else root
    world = OLD_WORLD if world is None else world
    return torch.load(root / f'{kind}_world_size_{world}_rank_{rank}.pt', map_location='cpu', weights_only=False, mmap=True)

def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all((equal(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all((equal(x, y) for x, y in zip(a, b)))
    return a == b

def flat_layout_proof(model, optim):
    config = json.loads((SOURCE / 'huggingface/config.json').read_text())
    layers = {}
    root_names = []
    for name, tensor in model.items():
        assert isinstance(tensor, DTensor) and tensor.dtype == torch.float32
        assert len(tensor.placements) == 1 and tensor.placements[0].is_shard() and (tensor.placements[0].dim == 0)
        assert tensor.shape[0] % OLD_WORLD == 0
        m = re.match('model\\.layers\\.(\\d+)\\.', name)
        if m:
            layers.setdefault(int(m.group(1)), []).append(name)
        else:
            root_names.append(name)
    if config.get('tie_word_embeddings') and 'lm_head.weight' in root_names:
        assert 'model.embed_tokens.weight' in root_names
        assert torch.equal(model['lm_head.weight']._local_tensor, model['model.embed_tokens.weight']._local_tensor)
        root_names.remove('lm_head.weight')
    groups = [root_names] + [layers[i] for i in sorted(layers)]
    ids = optim['param_groups'][0]['params']
    assert len(optim['param_groups']) == 1 and len(groups) == len(ids) == len(optim['state'])
    proof = []
    for param_id, names in zip(ids, groups):
        true_numel = sum((model[name].numel() for name in names))
        length = optim['state'][param_id]['exp_avg'].numel()
        assert true_numel == OLD_WORLD * length, ('FlatParameter padding/order mismatch', param_id, true_numel, length)
        assert true_numel % OLD_WORLD == 0 and true_numel % NEW_WORLD == 0
        assert optim['state'][param_id]['exp_avg_sq'].numel() == length
        proof.append({'optimizer_param_id': param_id, 'parameter_names': names, 'true_flat_numel': true_numel, 'old_local_numel': length, 'new_local_numel': true_numel // NEW_WORLD, 'old_shard_padding': 0, 'new_shard_padding': 0})
    return proof

def main():
    started = time.time()
    torch.set_num_threads(4)
    ROOT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'READY.json').exists(), 'Converted checkpoint already finalized'
    if (ROOT / 'FAILED.json').exists():
        previous = ROOT / f'ATTEMPT_FAILED_{int(time.time())}.json'
        (ROOT / 'FAILED.json').replace(previous)
    config = json.loads((SOURCE / 'fsdp_config.json').read_text())
    assert config == {'FSDP_version': 1, 'world_size': OLD_WORLD}
    m0, o0 = (load('model', 0), load('optim', 0))
    proof = flat_layout_proof(m0, o0)
    save_json('FLAT_LAYOUT_PROOF.json', {'groups': proof, 'zero_padding_proven_for_all_groups': True})
    del m0, o0
    completed = []
    save_json('STATUS.json', {'status': 'converting', 'started': started, 'completed_new_ranks': []})
    for rank in range(NEW_WORLD):
        oldranks = [2 * rank, 2 * rank + 1]
        mesh = DeviceMesh('cuda', torch.arange(NEW_WORLD), mesh_dim_names=('fsdp',), _init_backend=False, _rank=rank)
        mesh._dim_group_names = ['0']
        model_pair = [load('model', r) for r in oldranks]
        assert list(model_pair[0]) == list(model_pair[1])
        converted = OrderedDict()
        for name, a in model_pair[0].items():
            b = model_pair[1][name]
            assert a.shape == b.shape and a.dtype == b.dtype and (a.placements == b.placements)
            assert a._local_tensor.shape[0] == a.shape[0] // OLD_WORLD
            local = torch.cat([a._local_tensor, b._local_tensor], dim=0).contiguous()
            assert local.shape[0] == a.shape[0] // NEW_WORLD
            spec = DTensorSpec(mesh, a.placements, a._spec.tensor_meta)
            converted[name] = DTensor(local, spec, requires_grad=a.requires_grad)
        path = DEST / f'model_world_size_{NEW_WORLD}_rank_{rank}.pt'
        if not path.exists():
            torch.save(converted, path)
        check = load('model', rank, DEST, NEW_WORLD)
        for name, a in model_pair[0].items():
            v = check[name]
            split = a._local_tensor.shape[0]
            assert v.shape == a.shape and v.dtype == a.dtype and (v.placements == a.placements)
            assert v.device_mesh.mesh.tolist() == list(range(NEW_WORLD)) and v.device_mesh.get_coordinate() == [rank]
            assert torch.equal(v._local_tensor[:split], a._local_tensor)
            assert torch.equal(v._local_tensor[split:], model_pair[1][name]._local_tensor)
        model_tensor_count = len(check)
        del converted, check, model_pair
        pair = [load('optim', r) for r in oldranks]
        assert equal(pair[0]['param_groups'], pair[1]['param_groups'])
        optimizer = {'param_groups': deepcopy(pair[0]['param_groups']), 'state': {}}
        for unit in proof:
            key = unit['optimizer_param_id']
            a, b = (pair[0]['state'][key], pair[1]['state'][key])
            assert set(a) == set(b) == {'step', 'exp_avg', 'exp_avg_sq'}
            assert equal(a['step'], b['step']) and float(a['step']) == STEP
            state = {'step': a['step'].clone()}
            for field in ['exp_avg', 'exp_avg_sq']:
                assert a[field].ndim == b[field].ndim == 1 and a[field].numel() == b[field].numel() == unit['old_local_numel']
                assert a[field].dtype == b[field].dtype == torch.float32
                state[field] = torch.cat([a[field], b[field]]).contiguous()
                assert state[field].numel() == unit['new_local_numel']
            optimizer['state'][key] = state
        opath = DEST / f'optim_world_size_{NEW_WORLD}_rank_{rank}.pt'
        if not opath.exists():
            torch.save(optimizer, opath)
        check = load('optim', rank, DEST, NEW_WORLD)
        assert equal(check['param_groups'], pair[0]['param_groups'])
        for unit in proof:
            key = unit['optimizer_param_id']
            length = unit['old_local_numel']
            assert equal(check['state'][key]['step'], pair[0]['state'][key]['step'])
            for field in ['exp_avg', 'exp_avg_sq']:
                assert torch.equal(check['state'][key][field][:length], pair[0]['state'][key][field])
                assert torch.equal(check['state'][key][field][length:], pair[1]['state'][key][field])
        del optimizer, check, pair
        extra_a, extra_b = (load('extra_state', oldranks[0]), load('extra_state', oldranks[1]))
        assert equal(extra_a['lr_scheduler'], extra_b['lr_scheduler'])
        assert extra_a['lr_scheduler']['last_epoch'] == STEP
        epath = DEST / f'extra_state_world_size_{NEW_WORLD}_rank_{rank}.pt'
        if not epath.exists():
            torch.save(extra_a, epath)
        assert equal(load('extra_state', rank, DEST, NEW_WORLD), extra_a)
        completed.append({'new_rank': rank, 'old_ranks': oldranks, 'rng_from_old_rank': oldranks[0], 'model_tensors_verified': model_tensor_count, 'optimizer_groups_verified': len(proof), 'model_sha256': sha(path), 'optimizer_sha256': sha(opath), 'extra_sha256': sha(epath)})
        save_json('STATUS.json', {'status': 'converting', 'started': started, 'completed_new_ranks': completed, 'seconds': time.time() - started})
        print(json.dumps({'completed_rank': rank, 'old_ranks': oldranks, 'seconds': time.time() - started}), flush=True)
    (DEST / 'fsdp_config.json').write_text(json.dumps({'FSDP_version': 1, 'world_size': NEW_WORLD}, indent=2) + '\n')
    import shutil
    shutil.copytree(SOURCE / 'huggingface', DEST / 'huggingface', symlinks=False)
    assert not (DEST.parent / 'data.pt').exists()
    assert not torch.cuda.is_initialized() and (not torch.distributed.is_initialized())
    result = {'status': f'cpu_bitwise_verified_ready_for_fsdp{NEW_WORLD}_load', 'source_actor': str(SOURCE), 'converted_actor': str(DEST), 'resume_from_path': str(DEST.parent), 'old_world_size': OLD_WORLD, 'new_world_size': NEW_WORLD, 'model_dtype': 'float32', 'model_all_local_values_bitwise_verified': True, 'adam_m_v_all_values_bitwise_verified': True, 'optimizer_step': STEP, 'scheduler_preserved': True, 'flat_parameter_zero_padding_verified': True, 'rng_mapping': 'new rank r inherits old rank 2*r', 'fresh_loader_no_data_pt': True, 'ranks': completed, 'script_sha256': sha(__file__), 'seconds': time.time() - started, 'cuda_initialized': False, 'distributed_initialized': False, 'runtime_load_gate': 'Load the converted model and optimizer at the target FSDP world size.'}
    save_json('READY.json', result)
    save_json('STATUS.json', result)
    print(json.dumps({'status': result['status'], 'resume_from_path': result['resume_from_path'], 'seconds': result['seconds']}), flush=True)
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,help='global_step_N directory with actor shards')
    parser.add_argument('--output',required=True,help='New conversion output root')
    parser.add_argument('--old-world',type=int,required=True)
    parser.add_argument('--new-world',type=int,required=True)
    args=parser.parse_args()
    parent=Path(args.source).expanduser().resolve()
    match=re.fullmatch(r'global_step_(\d+)',parent.name)
    if not match:raise ValueError('Expected global_step_N source')
    STEP=int(match.group(1));OLD_WORLD=args.old_world;NEW_WORLD=args.new_world
    if OLD_WORLD!=2*NEW_WORLD or NEW_WORLD<1:raise ValueError('Verified pairwise halving only')
    SOURCE=parent/'actor';ROOT=Path(args.output).expanduser().resolve()
    if ROOT.exists() and any(ROOT.iterdir()):raise ValueError('Use a new output root')
    DEST=ROOT/parent.name/'actor'
    try:main()
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True)
        save_json('FAILED.json',{'error':traceback.format_exc(),'time':time.time()})
        raise
