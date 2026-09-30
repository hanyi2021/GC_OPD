"""Reject silent Qwen3 configuration changes across Transformers versions."""
import hashlib
import json
from pathlib import Path


def check_model_config(model_path):
    """Check declared Qwen3 dimensions and RoPE before loading any weights."""
    path = Path(model_path).expanduser().resolve() / 'config.json'
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    report = {'model_type': raw.get('model_type'),
              'config_sha256': hashlib.sha256(raw_bytes).hexdigest()}
    if raw.get('model_type') != 'qwen3':
        return {**report, 'scope': 'Qwen3 compatibility check not applicable'}
    import transformers
    from packaging.version import Version
    config = transformers.AutoConfig.from_pretrained(path.parent, local_files_only=True,
                                                     trust_remote_code=False)
    for field in ['hidden_size', 'intermediate_size', 'num_hidden_layers',
                  'num_attention_heads', 'num_key_value_heads', 'head_dim',
                  'vocab_size', 'max_position_embeddings', 'tie_word_embeddings',
                  'rms_norm_eps', 'use_sliding_window', 'max_window_layers']:
        if field in raw and getattr(config, field, None) != raw[field]:
            raise ValueError(f'Model config changes {field} when parsed: {path}')
    rope = raw.get('rope_parameters') or {}
    intended_theta = rope.get('rope_theta', raw.get('rope_theta'))
    intended_type = rope.get('rope_type', 'default')
    if 'rope_theta' in raw and 'rope_theta' in rope and raw['rope_theta'] != rope['rope_theta']:
        raise ValueError(f'Conflicting legacy and modern rope_theta in {path}')
    if Version(transformers.__version__) < Version('5'):
        # vLLM's Transformers-v4 compatibility path also uses these legacy fields.
        effective_theta = getattr(config, 'rope_theta', None)
        scaling = getattr(config, 'rope_scaling', None) or {}
        effective_type = scaling.get('rope_type', scaling.get('type', 'default'))
    else:
        parsed = getattr(config, 'rope_parameters', None) or {}
        effective_theta = parsed.get('rope_theta')
        effective_type = parsed.get('rope_type', 'default')
    if intended_theta is not None and effective_theta != intended_theta:
        raise ValueError(
            f'RoPE config mismatch in {path}: declared theta={intended_theta}, '
            f'but Transformers {transformers.__version__} resolves {effective_theta}. '
            'Export compatible RoPE metadata; for the supplied Qwen3 models, retain '
            'legacy rope_theta=1000000 alongside matching rope_parameters. Weights are unchanged.')
    if rope and effective_type != intended_type:
        raise ValueError(f'RoPE type changes from {intended_type} to {effective_type} in {path}; '
                         'provide compatible rope_scaling metadata for this Transformers version')
    return {**report, 'transformers_version': transformers.__version__,
            'declared_rope_theta': intended_theta, 'effective_rope_theta': effective_theta,
            'effective_rope_type': effective_type, 'scope': 'Declared Qwen3 dimensions and RoPE checked'}
