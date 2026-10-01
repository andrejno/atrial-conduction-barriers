from pathlib import Path
import json
import shutil
import sys
import platform
from importlib.metadata import version
import pandas as pd


def validate_run(publication_root, run_root):
    from notebook_validation import validate
    report = validate(Path(publication_root), Path(run_root))
    return pd.DataFrame(report)


def export_results(publication_root, run_root):
    root, run = Path(publication_root), Path(run_root)
    destination = root / 'results'
    destination.mkdir(exist_ok=True)
    for subdirectory in ('data', 'patient_extension/verification'):
        for path in sorted((run / subdirectory).rglob('*.csv')):
            target = destination / 'tables' / path.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for path in sorted((run / 'figures').rglob('*')):
        if path.is_file() and path.suffix in ('.pdf', '.png'):
            target = destination / 'figures' / path.relative_to(run / 'figures')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    packages = ('numpy', 'scipy', 'pandas', 'matplotlib', 'pillow', 'nbformat', 'nbclient', 'ipykernel')
    environment = {
        'python': sys.version,
        'platform': platform.platform(),
        'packages': {name: version(name) for name in packages},
        'run': str(run.relative_to(root)),
        'table_source': 'manuscript.tex',
        'computed_table_data': 'results/tables',
        'prior_result_checkpoints_copied': False,
    }
    (destination / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    shutil.copy2(run / 'execution_status.json', destination / 'execution_status.json')
    for filename in ('notebook_validation.json', 'notebook_validation_summary.csv', 'validation_independent.json'):
        if (run / filename).exists():
            shutil.copy2(run / filename, destination / filename)
    return destination
