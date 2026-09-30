"""Validate requested allocations without starting Ray or touching a GPU."""


def validate_teacher_inference(inference):
    """Check the vLLM batching constraint before launching the teacher."""
    batched = inference['max_num_batched_tokens']
    context = inference['max_model_len']
    sequences = inference['max_num_seqs']
    if any(type(value) is not int or value < 1 for value in [batched, context, sequences]):
        raise ValueError('Teacher context and batching limits must be positive integers')
    if batched < sequences:
        raise ValueError('Teacher max_num_batched_tokens must cover max_num_seqs')
    if batched < context and not inference.get('enable_chunked_prefill', False):
        raise ValueError('Teacher max_num_batched_tokens below max_model_len requires chunked prefill')


def training_resources(student_nodes, student_gpus_per_node, teacher_nodes,
                       teacher_gpus_per_node, teacher_tp, total_gpus=None):
    values = dict(student_nodes=student_nodes, student_gpus_per_node=student_gpus_per_node,
                  teacher_nodes=teacher_nodes, teacher_gpus_per_node=teacher_gpus_per_node,
                  teacher_tp=teacher_tp)
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    student = student_nodes * student_gpus_per_node
    teacher = teacher_nodes * teacher_gpus_per_node
    if teacher % teacher_tp:
        raise ValueError('Independent scoring-teacher pool must be divisible by teacher TP')
    required = student + teacher
    if total_gpus is not None:
        if type(total_gpus) is not int or total_gpus < required:
            raise ValueError(f'Total allocation must cover {student} student/rollout + '
                             f'{teacher} independent scoring-teacher = {required} GPUs')
    return dict(student_rollout_gpus=student, scoring_teacher_gpus=teacher,
                training_total_gpus=required, allocated_training_gpus=total_gpus,
                scoring_teacher_replicas=teacher // teacher_tp,
                pools_are_disjoint=True,
                scope='Independent student/rollout and scoring-teacher GPU allocations.')
