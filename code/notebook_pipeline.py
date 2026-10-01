from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare_workspace(source_root, destination):
    source_root, destination = Path(source_root).resolve(), Path(destination).resolve()
    if source_root == destination:
        raise ValueError('Use a separate destination from the source research directory.')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Destination must be empty: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('code', 'external_data', 'schemas', 'templates'):
        if (source_root / name).is_dir():
            shutil.copytree(source_root / name, destination / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for path in source_root.iterdir():
        if path.is_file() and path.suffix != '.ipynb' and not path.name.startswith('.'):
            shutil.copy2(path, destination / path.name)
    for name in ('data', 'figures', 'logs', 'tmp', 'patient_extension/verification'):
        (destination / name).mkdir(parents=True, exist_ok=True)
    inputs = {str(path.relative_to(destination)): _hash(path)
              for directory in ('code', 'external_data')
              for path in sorted((destination / directory).rglob('*')) if path.is_file()}
    _write(destination / 'fresh_run.json', {
        'created_utc': _utc(), 'source_root': str(source_root),
        'starting_data_files': 0, 'prior_result_checkpoints_copied': False,
        'inputs_sha256': inputs,
    })
    return destination


@dataclass(frozen=True)
class Stage:
    name: str
    commands: tuple
    dependencies: tuple = ()
    slots: int = 1


def stages():
    return (
        Stage('core', (('code/run_core.py',), ('code/run_conduction_extension.py',),
                       ('code/run_core.py', '--conduction-figure-only'),
                       ('code/notebook_figure_exports.py',))),
        Stage('algebraic_surface_verification', (('code/run_graph_certificate.py',),
              ('code/run_surface_verification.py', '--output-dir', 'patient_extension/verification'),
              ('code/test_surface_phase.py', '-v'), ('code/test_surface_fem.py', '-v'))),
        Stage('applied', (('code/run_applied.py',),)),
        Stage('support_exclusion', (('code/run_reconstruction_extension.py', '--refresh'),)),
        Stage('boundary_coverage', (('code/run_reconstruction_boundary_control.py', '--refresh'),
                                   ('code/run_reconstruction_boundary_topology_check.py',))),
        Stage('patient_base', (('code/run_zenodo_pvi_reconstruction.py', '--workers', '2'),
                               ('code/check_patient_contact_geometry.py',)), slots=2),
        Stage('factor_reuse_benchmark', (('code/benchmark_patient_factor_reuse.py',),),
              ('patient_base',)),
        Stage('patient_extension', (('code/run_patient_extension_study.py', '--part', 'all',
                                     '--workers', '2', '--force'),), slots=2),
        Stage('synthetic_summary', (('code/summarize_reconstruction_extensions.py',),),
              ('applied', 'support_exclusion', 'boundary_coverage')),
        Stage('patient_summary', (('code/summarize_patient_extension_study.py',),
                                  ('code/plot_patient_extension_study.py',),
                                  ('code/plot_patient_consolidated.py',)),
              ('patient_base', 'patient_extension')),
        Stage('forward_transfer', tuple(('code/run_reconstruction_ep.py', '--stage', item,
                                        '--workers', '2') for item in (
                'main', 'ep_refinement', 'reconstruction_refinement', 'boundary_main',
                'boundary_ep_refinement', 'boundary_ep_refinement2')),
              ('support_exclusion', 'boundary_coverage'), slots=2),
        Stage('patient_surface_ep', (('code/run_patient_surface_ep.py', '--verify', '--fields',
                '3:data/patient_extension/P3_left_pair_w3_h0_dt0p01.npz',
                '10:data/patient_extension/P3_left_pair_w10_h0_dt0p01.npz',
                '--dt', '.025', '--halved-dt', '--overwrite', '--spatial-control', '--plot'),
                ('code/run_patient_surface_ep_provenance.py',)),
              ('patient_extension',)),
    )


def _execute(root, stage, python):
    environment = dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[key] = '1'
    environment['MPLBACKEND'] = 'Agg'
    environment['PYTHONUNBUFFERED'] = '1'
    environment['PYTHONHASHSEED'] = '0'
    environment['MPLCONFIGDIR'] = str(root / 'tmp' / ('matplotlib_' + stage.name))
    started = time.perf_counter()
    records = []
    for index, arguments in enumerate(stage.commands, 1):
        path = root / 'logs' / f'{stage.name}_{index:02d}.log'
        command = [python, *arguments]
        t0 = time.perf_counter()
        with path.open('w') as output:
            process = subprocess.run(command, cwd=root, env=environment,
                                     stdout=output, stderr=subprocess.STDOUT)
        record = {'command': command, 'log': str(path.relative_to(root)),
                  'return_code': process.returncode, 'elapsed_seconds': time.perf_counter() - t0}
        records.append(record)
        if process.returncode:
            tail = path.read_text(errors='replace').splitlines()[-24:]
            raise RuntimeError(f'{stage.name}: exit {process.returncode}\n' + '\n'.join(tail))
    return {'commands': records, 'elapsed_seconds': time.perf_counter() - started,
            'finished_utc': _utc(), 'status': 'completed'}


def run_all(destination, max_workers=6, python=None, resume=False):
    root = Path(destination).resolve()
    if not (root / 'fresh_run.json').exists():
        raise ValueError('Run prepare_workspace before run_all.')
    if max_workers < 2:
        raise ValueError('At least two worker slots are required.')
    python = python or sys.executable
    status_path = root / 'execution_status.json'
    if status_path.exists() and not resume:
        raise FileExistsError('This workspace has already started. Use resume=True for this run or a fresh destination.')
    status = json.loads(status_path.read_text()) if resume and status_path.exists() else {
        'started_utc': _utc(), 'mode': 'fresh computation',
        'prior_result_checkpoints_copied': False, 'maximum_worker_slots': max_workers,
        'python': python, 'stages': {},
    }
    plan = stages()
    complete = {name for name, item in status['stages'].items() if item['status'] == 'completed'}
    pending = {item.name: item for item in plan if item.name not in complete}
    active = {}
    failure = None
    _write(status_path, status)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or active:
            used = sum(item.slots for item in active.values())
            for name, item in list(pending.items()):
                if all(dependency in complete for dependency in item.dependencies) and used + item.slots <= max_workers:
                    status['stages'][name] = {'status': 'running', 'started_utc': _utc(),
                                              'worker_slots': item.slots}
                    _write(status_path, status)
                    active[executor.submit(_execute, root, item, python)] = item
                    del pending[name]
                    used += item.slots
                    print(f'{name}: running', flush=True)
            if not active:
                raise RuntimeError(f'Unresolved stages: {list(pending)}')
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                try:
                    result = future.result()
                    status['stages'][item.name].update(result)
                    complete.add(item.name)
                    print(f'{item.name}: completed ({result["elapsed_seconds"]:.1f} s)', flush=True)
                except Exception as error:
                    status['stages'][item.name].update(status='failed', error=str(error), finished_utc=_utc())
                    failure = error
                    print(f'{item.name}: failed', flush=True)
                _write(status_path, status)
            if failure is not None:
                for future, item in list(active.items()):
                    try:
                        result = future.result()
                        status['stages'][item.name].update(result)
                    except Exception as error:
                        status['stages'][item.name].update(status='failed', error=str(error), finished_utc=_utc())
                    _write(status_path, status)
                raise failure
    status['finished_utc'] = _utc()
    status['status'] = 'completed'
    status['result_counts'] = {
        'csv': len(list((root / 'data').rglob('*.csv'))),
        'npz': len(list((root / 'data').rglob('*.npz'))),
        'figures_pdf': len(list((root / 'figures').rglob('*.pdf'))),
        'figures_png': len(list((root / 'figures').rglob('*.png'))),
    }
    status['results_sha256'] = {
        str(path.relative_to(root)): _hash(path)
        for directory in ('data', 'figures', 'patient_extension/verification')
        for path in sorted((root / directory).rglob('*')) if path.is_file()
    }
    _write(status_path, status)
    return status
