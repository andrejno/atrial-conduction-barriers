from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import stat
import tempfile
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError(f'Invalid input path: {value}')
    return path


def _local_sources(repository):
    values = {}
    for path in sorted((repository / 'code').glob('*.py')):
        if path.is_symlink():
            raise ValueError(f'Source file is a symlink: {path}')
        values['code/' + path.name] = path.read_bytes()
    resource_root = repository / 'code/resources'
    for path in sorted(resource_root.rglob('*')):
        if not path.is_file() or any(part.startswith('.') for part in path.relative_to(resource_root).parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Resource is a symlink: {path}')
        values[path.relative_to(resource_root).as_posix()] = path.read_bytes()
    values['requirements.txt'] = (repository / 'requirements.txt').read_bytes()
    return values


def prepare_inputs(repository=None, workspace=None):
    repository = Path(repository).resolve() if repository else REPOSITORY
    destination = Path(workspace).resolve() if workspace else repository / '.research'
    manifest_path = repository / 'code/inputs/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 1:
        raise ValueError('Unknown input bundle format.')
    local = _local_sources(repository)
    expected = {item['path']: item for item in manifest['files']}
    if len(expected) != len(manifest['files']) or set(expected) & set(local):
        raise ValueError('Input paths are not unique.')
    for name in (*expected, *local):
        _relative(name)
    source_hashes = {name: _sha(value) for name, value in local.items()}
    identity = {'input_archive_sha256': manifest['sha256'], 'source_sha256': source_hashes}
    marker_name = 'repository_inputs.json'
    if destination.exists():
        marker = destination / marker_name
        if destination.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise FileExistsError(f'Workspace inputs or source code differ: {destination}. Select a new --workspace directory.')
        for name, digest in {**{name: item['sha256'] for name, item in expected.items()}, **source_hashes}.items():
            path = destination / name
            if not path.is_file() or path.is_symlink() or _sha(path.read_bytes()) != digest:
                raise FileExistsError(f'Existing workspace file differs: {path}. Select a new --workspace directory.')
        return destination
    stream = io.BytesIO()
    for chunk in manifest['chunks']:
        name = _relative(chunk['file'])
        if len(name.parts) != 1:
            raise ValueError('Input chunk must be a filename.')
        value = (manifest_path.parent / name).read_bytes()
        if len(value) != chunk['bytes'] or _sha(value) != chunk['sha256']:
            raise ValueError(f'Input chunk hash mismatch: {name}')
        stream.write(value)
    payload = stream.getvalue()
    if len(payload) != manifest['bytes'] or _sha(payload) != manifest['sha256']:
        raise ValueError('Input archive hash mismatch.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or set(archive.namelist()) != set(expected):
            raise ValueError('Input archive inventory mismatch.')
        for info in archive.infolist():
            _relative(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                raise ValueError('Input archive must contain regular files.')
        with tempfile.TemporaryDirectory(prefix='.research_stage_', dir=destination.parent) as temporary:
            staging = Path(temporary) / 'research'
            staging.mkdir()
            for name, item in expected.items():
                value = archive.read(name)
                if len(value) != item['bytes'] or _sha(value) != item['sha256']:
                    raise ValueError(f'Input file hash mismatch: {name}')
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            for name, value in local.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            (staging / marker_name).write_text(json.dumps(identity, indent=2) + '\n')
            if destination.exists():
                raise FileExistsError(destination)
            staging.rename(destination)
    return destination
