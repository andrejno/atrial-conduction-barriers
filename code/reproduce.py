from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import sys

from input_loader import prepare_inputs


REPOSITORY = Path(__file__).resolve().parents[1]


def prepare_repository(repository=None, workspace=None):
    root = prepare_inputs(repository or REPOSITORY, workspace)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    sys.path.insert(0, str(root / 'code'))
    return root


def run_research(root, workers=6, fresh=False):
    from notebook_pipeline import prepare_workspace, run_all
    root = Path(root).resolve()
    previous = root / 'latest_run.json'
    if previous.exists() and not fresh:
        work = (root / json.loads(previous.read_text())['run']).resolve()
        if not work.is_relative_to(root):
            raise ValueError('Recorded run lies outside the workspace.')
    else:
        work = root / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    status_file = work / 'execution_status.json'
    if status_file.exists():
        execution = json.loads(status_file.read_text())
        if execution.get('status') == 'completed':
            for name, expected in execution['results_sha256'].items():
                actual = hashlib.sha256((work / name).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f'Result changed: {name}. Use --fresh for a new run.')
        else:
            execution = run_all(work, max_workers=workers, resume=True)
    else:
        prepare_workspace(root, work)
        previous.write_text(json.dumps({'run': str(work.relative_to(root))}, indent=2) + '\n')
        execution = run_all(work, max_workers=workers)
    previous.write_text(json.dumps({'run': str(work.relative_to(root))}, indent=2) + '\n')
    return work, execution


def validate_research(root, work):
    from notebook_report import validate_run, export_results
    validation = validate_run(root, work)
    exports = export_results(root, work)
    return validation, exports


def generate_media(root):
    from create_ring_media import generate_all as rings
    from create_patient_media import generate_all as patients
    return {'rings': rings(), 'patients': patients()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--media', action='store_true')
    options = parser.parse_args()
    root = prepare_repository(workspace=options.workspace)
    if options.prepare_only:
        print(root)
        return
    work, execution = run_research(root, workers=options.workers, fresh=options.fresh)
    validation, exports = validate_research(root, work)
    if options.media:
        generate_media(root)
    print(validation.to_string(index=False))
    print(exports)


if __name__ == '__main__':
    main()
