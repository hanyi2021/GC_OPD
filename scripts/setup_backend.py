"""Prepare the pinned veRL checkout and apply the GC-OPD source patch."""
import argparse
import hashlib
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'https://github.com/verl-project/verl'
COMMIT = '483b8a009ba3a97563edee3a19887e4862b8094a'  # veRL v0.9.0
SETUP_HINT = 'Run python scripts/setup_backend.py from the package root.'


class BackendError(RuntimeError):
    pass


def _git(directory, *args, check=True):
    # Git retains the caller's network/proxy configuration; no proxy is added here.
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    try:
        result = subprocess.run(
            ['git', '-C', str(directory), '-c', 'core.autocrlf=false',
             '-c', 'core.fileMode=true', '-c', 'core.fsmonitor=false',
             '-c', 'diff.autoRefreshIndex=false', *args],
            env=env, capture_output=True, text=True,
        )
    except FileNotFoundError as exc:
        raise BackendError('Git is required to prepare and verify the veRL backend.') from exc
    if check and result.returncode:
        raise BackendError(result.stderr.strip() or result.stdout.strip() or 'Git command failed.')
    return result


def _patch_entries(patch):
    if not patch.is_file():
        raise BackendError(f'Missing method patch: {patch}')
    text = patch.read_text()
    matches = re.findall(
        r'^diff --git a/(\S+) b/(\S+)\nindex ([0-9a-f]{40})\.\.([0-9a-f]{40}) 100644$',
        text, re.MULTILINE,
    )
    if not matches or text.count('diff --git ') != len(matches):
        raise BackendError('Expected a GC-OPD patch with full Git blob IDs.')
    entries = {}
    for before_path, after_path, before, after in matches:
        if before_path != after_path or not before_path.startswith('verl/') or '..' in Path(before_path).parts:
            raise BackendError('Unexpected path in the method patch.')
        entries[before_path] = (before, after)
    if len(entries) != len(matches):
        raise BackendError('Duplicate path in the method patch.')
    return entries


def _blob_id(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def backend_state(directory, patch):
    """Return clean/patched, refusing any unrecognized tracked or untracked edits."""
    directory, patch = Path(directory), Path(patch)
    if not directory.is_dir() or not (directory / '.git').exists():
        raise BackendError(f'Backend is not a Git checkout at {directory}. {SETUP_HINT}')
    top = Path(_git(directory, 'rev-parse', '--show-toplevel').stdout.strip()).resolve()
    if top != directory.resolve():
        raise BackendError('The backend directory must be the root of its own veRL Git checkout.')
    head = _git(directory, 'rev-parse', 'HEAD').stdout.strip()
    if head != COMMIT:
        raise BackendError(f'Expected veRL commit {COMMIT}; found {head}. Existing checkout was left unchanged.')
    entries = _patch_entries(patch)
    for path, (before, _) in entries.items():
        if _git(directory, 'rev-parse', f'HEAD:{path}').stdout.strip() != before:
            raise BackendError(f'Patch base does not match the pinned commit: {path}')
    if _git(directory, 'diff', '--cached', '--quiet', check=False).returncode:
        raise BackendError('Backend has staged changes; leave them intact and use a clean checkout.')
    untracked = _git(directory, 'ls-files', '--others', '--exclude-standard', '-z').stdout
    if untracked:
        raise BackendError('Backend has unrecognized untracked files; existing files were left unchanged.')
    changed = set(filter(None, _git(directory, 'diff', '--no-ext-diff', '--no-textconv',
                                   '--name-only', '-z', 'HEAD', '--').stdout.split('\0')))
    summary = _git(directory, 'diff', '--no-ext-diff', '--no-textconv', '--summary', 'HEAD', '--').stdout
    if summary or changed - set(entries):
        raise BackendError('Backend has unrecognized modifications; existing files were left unchanged.')
    actual = {}
    for path in entries:
        source = directory / path
        if not source.is_file() or source.is_symlink():
            raise BackendError(f'Backend source is missing or replaced by a symlink: {path}')
        actual[path] = _blob_id(source)
    # With index refresh disabled, Git may list touched files whose bytes are unchanged.
    # The full blob IDs in the patch distinguish clean, patched and local edits exactly.
    for column, state in enumerate(('clean', 'patched')):
        if all(actual[path] == hashes[column] for path, hashes in entries.items()):
            return state
    raise BackendError('Backend patch files contain unrecognized edits or a partial patch; left unchanged.')


def profile_paths(root, profile='scienceworld'):
    if profile not in ('scienceworld', 'alfworld', 'webshop'):
        raise BackendError('Unknown backend profile')
    directory = root / 'external' / 'verl' / profile
    patch = root / 'patches' / f'verl-{profile}.patch'
    return directory, patch


def require_backend(root=ROOT, profile='scienceworld'):
    """Verify source before a runtime import, and prefer this checkout on sys.path."""
    root = Path(root).resolve()
    try:
        directory, patch = profile_paths(root, profile)
        state = backend_state(directory, patch)
        if state != 'patched':
            raise BackendError('The pinned veRL checkout still needs the GC-OPD patch.')
    except BackendError as exc:
        raise BackendError(f'GC-OPD {profile} backend is not ready: {exc}\nRun python scripts/setup_backend.py --profile {profile}') from exc
    backend = directory / 'verl'
    loaded = sys.modules.get('verl')
    if loaded is not None and Path(getattr(loaded, '__file__', '') or '').resolve() != backend / '__init__.py':
        raise BackendError('A different verl package is already imported; start a new process after backend setup.')
    framework = str(directory)
    if framework in sys.path:
        sys.path.remove(framework)
    sys.path.insert(0, framework)
    importlib.invalidate_caches()
    return backend


def _apply(directory, patch):
    state = backend_state(directory, patch)
    if state == 'clean':
        _git(directory, 'apply', '--check', '--whitespace=nowarn', str(patch))
        _git(directory, 'apply', '--whitespace=nowarn', str(patch))
        if backend_state(directory, patch) != 'patched':
            raise BackendError('Backend verification failed after applying the method patch.')
    return state


def setup(root=ROOT, source_checkout=None, check=False, profile='scienceworld'):
    root = Path(root).resolve()
    directory, patch = profile_paths(root, profile)
    if check:
        if backend_state(directory, patch) != 'patched':
            raise BackendError(f'Backend patch has not been applied. {SETUP_HINT}')
        return 'verified'
    if directory.exists() or directory.is_symlink():
        return 'already prepared' if _apply(directory, patch) == 'patched' else 'patch applied'
    _patch_entries(patch)
    source = UPSTREAM
    if source_checkout is not None:
        source_path = Path(source_checkout).expanduser().resolve()
        _git(source_path, 'cat-file', '-e', COMMIT + '^{commit}')
        source = str(source_path)
    # Prepare in a new directory, so a failed fetch or patch never leaves a partial backend.
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.framework-setup-', dir=directory.parent) as temporary:
        checkout = Path(temporary) / 'checkout'
        checkout.mkdir()
        _git(checkout, 'init', '--quiet')
        _git(checkout, 'remote', 'add', 'origin', UPSTREAM)
        _git(checkout, 'fetch', '--no-tags', '--depth', '1', source, COMMIT)
        _git(checkout, 'checkout', '--quiet', '--detach', COMMIT)
        _apply(checkout, patch)
        if directory.exists() or directory.is_symlink():
            raise BackendError('framework appeared during setup; it was left unchanged.')
        checkout.rename(directory)
    return 'prepared'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['scienceworld', 'alfworld', 'webshop'], default='scienceworld')
    parser.add_argument('--source-checkout', type=Path,
                        help='Fetch the pinned commit from an existing local Git checkout, without network access')
    parser.add_argument('--check', action='store_true', help='Verify the prepared backend without downloading or changing it')
    args = parser.parse_args()
    if args.check and args.source_checkout:
        parser.error('--source-checkout cannot be combined with --check')
    try:
        result = setup(source_checkout=args.source_checkout, check=args.check, profile=args.profile)
    except (BackendError, OSError) as exc:
        parser.exit(1, f'Backend setup failed: {exc}\n')
    print(f'veRL {COMMIT}: {result} at {profile_paths(ROOT, args.profile)[0] / "verl"}')


if __name__ == '__main__':
    main()
