import torch
def align_teacher(ids,lps,turn):
    a=len(turn['prompt_ids']);ta=len(turn.get('teacher_prompt_ids',turn['prompt_ids']));b=len(turn['response_ids'])
    assert ids.ndim==2 and ids.shape[1]==1 and lps.shape==ids.shape
    assert torch.equal(ids[ta-1:ta+b-1,0].long().cpu(),torch.tensor(turn['response_ids']))
    assert torch.isfinite(lps[ta-1:ta+b-1]).all()
    outids=torch.zeros((a+b,1),dtype=ids.dtype,device=ids.device);outlps=torch.zeros((a+b,1),dtype=lps.dtype,device=lps.device)
    outids[a-1:a+b-1]=ids[ta-1:ta+b-1];outlps[a-1:a+b-1]=lps[ta-1:ta+b-1]
    return outids,outlps
