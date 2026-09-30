"""Convert task-ID manifests to the veRL parquet schema."""
import argparse
import math
from gcopd.common.io import task_ids, new_output, write_json, digest

def rows(ids, pad_to=1, rep=None):
    out = []
    total = math.ceil(len(ids) / pad_to) * pad_to
    for i in range(total):
        padding = i >= len(ids)
        gf = f'__padding__{i}' if padding else ids[i]
        extra = {'gamefile': gf, 'index': i, 'padding': padding}
        if rep is not None:
            extra['eval_rep'] = rep
        out.append({'data_source': 'scienceworld', 'prompt': [{'role': 'user', 'content': f'ScienceWorld task {gf}'}],
                    'reward_model': {'style': 'rule', 'ground_truth': ''}, 'extra_info': extra, 'agent_name': 'scienceworld'})
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for split in ['train','dev','test']:
        p.add_argument('--'+split, required=True, help='JSON list of gamefile IDs, in intended order')
    p.add_argument('--output', required=True)
    p.add_argument('--expected-counts', nargs=3, type=int, default=[3017,305,1661])
    a = p.parse_args()
    ids = {s: task_ids(getattr(a,s)) for s in ['train','dev','test']}
    if [len(ids[s]) for s in ids] != a.expected_counts:
        raise ValueError('Split sizes differ from expected counts; use explicit expected counts for a smoke subset')
    for left,right in [('train','dev'),('train','test'),('dev','test')]:
        if set(ids[left]) & set(ids[right]):
            raise ValueError(f'Overlapping task identities: {left}/{right}')
    import pyarrow as pa
    import pyarrow.parquet as pq
    out = new_output(a.output)
    report = {}
    for split, tasks in ids.items():
        output = out / (split+'.parquet')
        # Zero-loss padding makes exactly ceil(N/32) batches; no task repeated.
        data = rows(tasks, 32 if split == 'train' else 1)
        pq.write_table(pa.Table.from_pylist(data), output)
        report[split] = {'tasks': len(tasks), 'rows': len(data), 'sha256': digest(output),
                         'task_manifest_sha256': digest(getattr(a,split)), 'task_ids': tasks}
    write_json(out/'DATA_MANIFEST.json', report)
    print('Prepared split parquets and DATA_MANIFEST.json')

if __name__ == '__main__':
    main()
