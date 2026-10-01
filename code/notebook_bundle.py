from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError(f'Invalid bundle path: {value}')
    return path


def _selected_files(root):
    paths = set(root.glob('code/**/*.py'))
    paths.update(path for path in (root / 'external_data').rglob('*') if path.is_file() and not any(part.startswith('.') for part in path.relative_to(root).parts))
    for directory in ('schemas', 'templates'):
        paths.update(path for path in (root / directory).rglob('*') if path.is_file() and not any(part.startswith('.') for part in path.relative_to(root).parts))
    for name in ('manuscript.tex', 'references.bib', 'DATA_ATTRIBUTION.md', 'study_inventory.json', 'README.md', 'svjour3.cls', 'svglov3.clo', 'spbasic.bst'):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
    for pattern in ('*DESIGN*', '*PROTOCOL*', 'requirements*.txt', 'environment*.yml'):
        paths.update(path for path in root.glob(pattern) if path.is_file())
    for directory in ('data', 'patient_extension/verification'):
        paths.update(path for path in (root / directory).rglob('*') if path.is_file() and path.suffix in {'.csv', '.json'} and not path.name.startswith('partial_'))
    patient_provenance = root / 'media_data/patient_media_provenance.json'
    patient = json.loads(patient_provenance.read_text())
    for name in patient['input_sha256']:
        relative = _safe_relative(name)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
    for provenance in sorted((root / 'media').glob('rings_*.json')):
        for case in json.loads(provenance.read_text())['source_cases']:
            for key in ('source', 'checkpoint', 'table'):
                if case.get(key):
                    path = root / _safe_relative(case[key])
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    paths.add(path)
    return sorted(paths)


def build_bundle(root=None):
    root = Path(root).resolve() if root else ROOT
    files = []
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in _selected_files(root):
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative)
            if path.is_symlink():
                raise ValueError(f'Bundle input must not be a symlink: {relative}')
            value = path.read_bytes()
            files.append({'path': relative, 'bytes': len(value), 'sha256': _digest(value)})
            info = zipfile.ZipInfo(relative, date_time=(2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        manifest = json.dumps({'format': 1, 'files': files}, indent=2, ensure_ascii=False).encode()
        info = zipfile.ZipInfo('bundle_manifest.json', date_time=(2026, 9, 30, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    payload = stream.getvalue()
    return payload, _digest(payload), files


def bootstrap_source(notebook_name='Atrial_Reconstruction.ipynb', destination_name='Atrial_Reconstruction_files'):
    return '''from pathlib import Path, PurePosixPath
import base64, hashlib, io, json, stat, tempfile, zipfile

SOURCE_ROOT = Path.cwd().resolve()
if not (SOURCE_ROOT / 'code/model.py').is_file():
    _notebook = SOURCE_ROOT / NOTEBOOK_NAME
    _bundle = json.loads(_notebook.read_text())['metadata']['research_bundle']
    _payload = base64.b64decode(_bundle['payload'], validate=True)
    _sha = hashlib.sha256(_payload).hexdigest()
    if _sha != _bundle['sha256']:
        raise ValueError('Embedded research bundle hash mismatch.')
    _destination = SOURCE_ROOT / DESTINATION_NAME
    with zipfile.ZipFile(io.BytesIO(_payload)) as _archive:
        _names = _archive.namelist()
        if len(_names) != len(set(_names)):
            raise ValueError('Duplicate embedded bundle paths.')
        for _info in _archive.infolist():
            _path = PurePosixPath(_info.filename)
            if _path.is_absolute() or '..' in _path.parts or '\\\\' in _info.filename or ':' in _info.filename or stat.S_ISLNK(_info.external_attr >> 16):
                raise ValueError('Unsafe embedded bundle path.')
        _manifest = json.loads(_archive.read('bundle_manifest.json'))
        if set(_names) != {'bundle_manifest.json', *[item['path'] for item in _manifest['files']]}:
            raise ValueError('Embedded bundle inventory mismatch.')
        if _destination.exists():
            _marker = _destination / 'bundle.sha256'
            if not _marker.is_file() or _marker.read_text().strip() != _sha:
                raise FileExistsError(f'Existing directory is not this research bundle: {_destination}')
            for _item in _manifest['files']:
                _file = _destination / _item['path']
                if not _file.is_file() or _file.is_symlink() or hashlib.sha256(_file.read_bytes()).hexdigest() != _item['sha256']:
                    raise FileExistsError(f'Existing research input differs: {_file}')
        else:
            with tempfile.TemporaryDirectory(prefix='Atrial_Reconstruction_extract_', dir=SOURCE_ROOT) as _temporary:
                _staging = Path(_temporary) / 'research'
                _staging.mkdir()
                for _item in _manifest['files']:
                    _content = _archive.read(_item['path'])
                    if len(_content) != _item['bytes'] or hashlib.sha256(_content).hexdigest() != _item['sha256']:
                        raise ValueError('Embedded research file hash mismatch.')
                    _file = _staging / _item['path']
                    _file.parent.mkdir(parents=True, exist_ok=True)
                    _file.write_bytes(_content)
                (_staging / 'bundle_manifest.json').write_bytes(_archive.read('bundle_manifest.json'))
                (_staging / 'bundle.sha256').write_text(_sha + '\\n')
                if _destination.exists():
                    raise FileExistsError(_destination)
                _staging.rename(_destination)
    SOURCE_ROOT = _destination
    del _payload, _bundle
'''.replace('NOTEBOOK_NAME', repr(notebook_name)).replace('DESTINATION_NAME', repr(destination_name))
