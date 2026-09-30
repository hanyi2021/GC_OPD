import subprocess

def close_owned_environment(env):
    """Close only this freshly created environment, with scoped JVM fallback."""
    record = {'close_called': env is not None, 'close_error': None,
              'jvm_pid': None, 'jvm_exited': None, 'forced_termination': False}
    if env is None:
        return record
    process = getattr(getattr(env, '_gateway', None), 'java_process', None)
    if process is not None:
        record['jvm_pid'] = process.pid
    try:
        env.close()
    except Exception as exc:
        record['close_error'] = repr(exc)
    if process is not None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            record['forced_termination'] = True
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        record['jvm_exited'] = process.poll() is not None
        record['jvm_returncode'] = process.returncode
    return record
