#!/usr/bin/env python3
"""Generate one student decision per ALFWorld/WebShop observation JSON."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from gcopd.inference.protocol import build_prompt, parse_action, request_seed


class TransformersEngine:
    backend = 'transformers'

    def __init__(self, model, environment, device):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; pass --device cpu for CPU inference')
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        factory = AutoModelForCausalLM if environment == 'alfworld' else AutoModelForImageTextToText
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
        self.model, self.loading_info = factory.from_pretrained(
            model, dtype=dtype, attn_implementation='sdpa', output_loading_info=True)
        if self.loading_info.get('missing_keys') or self.loading_info.get('unexpected_keys'):
            raise ValueError(f'Unexpected model loading keys: {self.loading_info}')
        self.loading_info = json.loads(json.dumps(self.loading_info,
            default=lambda value: sorted(value) if isinstance(value, set) else str(value)))
        self.model.to(self.device).eval()

    def get_tokenizer(self):
        return self.tokenizer


def load_engine(model, environment, memory=0.5, backend='transformers', device='cuda'):
    if backend == 'transformers':
        return TransformersEngine(model, environment, device)
    if device != 'cuda':
        raise ValueError('The optional vLLM backend requires --device cuda')
    from vllm import LLM
    kwargs = dict(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
                  distributed_executor_backend='mp', enforce_eager=True, seed=42,
                  gpu_memory_utilization=memory, max_num_seqs=4,
                  max_num_batched_tokens=4096, enable_chunked_prefill=True,
                  enable_prefix_caching=False, disable_log_stats=True,
                  max_model_len=10752 if environment == 'alfworld' else 65536)
    if environment == 'webshop':
        kwargs['limit_mm_per_prompt'] = {'image': 0, 'video': 0}
    engine = LLM(**kwargs)
    engine.backend = 'vllm'
    return engine


def generate(engine, environment, states, temperature=0.4, max_tokens=None):
    cap = max_tokens if max_tokens is not None else (512 if environment == 'alfworld' else 2048)
    if cap < 1 or temperature < 0:
        raise ValueError('max_tokens must be positive and temperature nonnegative')
    tokenizer = engine.get_tokenizer()
    requests, parameters, metadata = [], [], []
    for state in states:
        prompt = build_prompt(environment, state)
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        limit = 10240 if environment == 'alfworld' else 63488
        if len(ids) > limit:
            raise ValueError(f'Prompt exceeds {limit} tokens; truncation is not allowed')
        seed = request_seed(environment, state)
        requests.append({'prompt_token_ids': ids})
        parameters.append({'temperature': temperature, 'top_p': 1.0, 'top_k': -1,
                           'max_tokens': cap, 'seed': seed, 'repetition_penalty': 1.0,
                           'presence_penalty': 0.0, 'frequency_penalty': 0.0})
        metadata.append({'seed': seed, 'prompt_tokens': len(ids),
                         'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
    if engine.backend == 'transformers':
        import torch
        results = []
        for request, settings, record in zip(requests, parameters, metadata):
            torch.manual_seed(settings['seed'])
            ids = torch.tensor([request['prompt_token_ids']], device=engine.device)
            kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                          do_sample=temperature > 0, max_new_tokens=cap, top_k=0,
                          top_p=1.0, repetition_penalty=1.0,
                          pad_token_id=tokenizer.pad_token_id)
            if temperature > 0:
                kwargs['temperature'] = temperature
            with torch.inference_mode():
                output = engine.model.generate(**kwargs)
            answer_ids = output[0, ids.shape[1]:].tolist()
            text = tokenizer.decode(answer_ids, skip_special_tokens=True)
            action, error = parse_action(text)
            results.append({**record, 'text': text, 'action': action, 'parse_error': error,
                            'token_ids': answer_ids,
                            'finish_reason': 'length' if len(answer_ids) == cap else 'stop'})
        return results
    from vllm import SamplingParams
    outputs = engine.generate(requests, [SamplingParams(**p) for p in parameters], use_tqdm=False)
    result = []
    for record, output in zip(metadata, outputs):
        answer = output.outputs[0]
        action, error = parse_action(answer.text)
        result.append({**record, 'text': answer.text, 'action': action, 'parse_error': error,
                       'token_ids': list(answer.token_ids), 'finish_reason': answer.finish_reason})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['transformers', 'vllm'], default='transformers')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--environment', choices=['alfworld', 'webshop'], required=True)
    parser.add_argument('--input', required=True, help='One state object or a list of state objects')
    parser.add_argument('--output', required=True)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--max-tokens', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'Choose a new output file: {out}')
    states = json.loads(Path(args.input).read_text())
    if isinstance(states, dict):
        states = [states]
    if not states:
        raise ValueError('Input state list is empty')
    engine = load_engine(args.model, args.environment, args.gpu_memory_utilization, args.backend, args.device)
    predictions = generate(engine, args.environment, states, args.temperature, args.max_tokens)
    import torch, transformers
    versions = {'python': platform.python_version(), 'executable': sys.executable,
                'torch': torch.__version__, 'transformers': transformers.__version__}
    if args.backend == 'vllm':
        import vllm
        versions['vllm'] = vllm.__version__
    record = {'environment': args.environment, 'model': str(Path(args.model).resolve()),
              'temperature': args.temperature, 'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              'runtime': versions, 'backend': args.backend,
              'loading_info': getattr(engine, 'loading_info', None),
              'predictions': predictions}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(out), 'decisions': len(predictions),
                      'valid_actions': sum(x['parse_error'] is None for x in predictions)}))

if __name__ == '__main__':
    main()
