from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = (
    'scope', 'study', 'experiment', 'acquisition', 'quantity', 'regime', 'backend', 'patient_id',
    'mask', 'mask_name', 'label', 'geometry', 'method', 'metric', 'endpoint',
    'contrast', 'comparison', 'comparator', 'baseline', 'sample', 'pv_label',
    'metric_support', 'evaluation_width_mm', 'mesh_level', 'level', 'N',
    'blackout_arc_width_mm', 'blackout_angle_deg', 'width_mm', 'angle_deg',
    'gap_width_mm', 'gap_diffusivity_fraction', 'gap_angle_deg', 'matched_case',
    'case_id', 'direction', 'pacing_direction', 'face_average', 'configuration',
    'n', 'n_radial', 'n_angular', 'refinement_level', 'reconstruction_n', 'ep_n', 'dx_mm', 'dt', 'dt_ms', 'pseudo_dt',
    'reconstruction_dt', 'ep_dt_ms', 'admm_tolerance', 'subcells_per_axis', 'bin',
)


def _excluded_column(name):
    low = name.lower()
    return (any(word in low for word in ('seconds', 'elapsed', 'wall_time', 'cpu_time', 'sha256'))
            or low.endswith(('_path', '_file', '_json'))
            or low in {'checkpoint', 'reused_identical_reference_from'})


def _key(value):
    if pd.isna(value):
        return '<NA>'
    if isinstance(value, (float, np.floating)):
        return f'{value:.12g}'
    return str(value)


def compare_csv(reference, computed, rtol=5e-8, atol=1e-10):
    first, second = pd.read_csv(reference), pd.read_csv(computed)
    original_counts = {'reference': len(first), 'computed': len(second)}
    excluded_rows = {}
    for label_column in ('metric', 'endpoint'):
        if label_column not in first or label_column not in second:
            continue
        labels = set(first[label_column].dropna().astype(str)) | set(second[label_column].dropna().astype(str))
        timing_labels = sorted(label for label in labels if any(token in label.lower()
            for token in ('cpu_seconds', 'wall_seconds', 'elapsed_seconds', 'solver_seconds',
                          'assembly_seconds', 'setup_seconds', 'advance_seconds')))
        if timing_labels:
            excluded_rows[label_column] = {
                'metrics': timing_labels,
                'reference_rows': int(first[label_column].isin(timing_labels).sum()),
                'computed_rows': int(second[label_column].isin(timing_labels).sum()),
            }
            first = first.loc[~first[label_column].isin(timing_labels)].copy()
            second = second.loc[~second[label_column].isin(timing_labels)].copy()
    common = [name for name in first if name in second and not _excluded_column(name)]
    missing = [name for name in first if name not in second and not _excluded_column(name)]
    row = {'rows_reference': len(first), 'rows_computed': len(second),
           'original_row_counts': original_counts, 'excluded_timing_rows': excluded_rows,
           'columns_compared': common, 'excluded_columns': [c for c in first if _excluded_column(c)],
           'new_columns': [c for c in second if c not in first],
           'missing_columns': missing, 'mismatches': []}
    if len(first) != len(second) or missing:
        row['passed'] = False
        return row
    keys = [name for name in KEY_COLUMNS if name in common]
    if not keys and len(first) > 1:
        row.update(passed=False, alignment_error='No declared row keys')
        return row
    index1 = first[keys].map(_key).agg('|'.join, axis=1) if keys else pd.Series(['row'])
    index2 = second[keys].map(_key).agg('|'.join, axis=1) if keys else pd.Series(['row'])
    if index1.duplicated().any() or index2.duplicated().any():
        if Path(reference).name == 'reconstruction_extension_field_sensitivity.csv' and Path(reference).parent.name == 'reconstruction_boundary_control':
            index1 = index1 + '|occurrence=' + index1.groupby(index1, sort=False).cumcount().astype(str)
            index2 = index2 + '|occurrence=' + index2.groupby(index2, sort=False).cumcount().astype(str)
            row['alignment_note'] = 'The source table omits blackout width; repeated keys follow the declared width-2 then width-6 generation order.'
        else:
            row.update(passed=False, alignment_error='Declared row keys are not unique', keys=keys)
            return row
    if set(index1) != set(index2):
        row.update(passed=False, alignment_error='Different row keys', keys=keys,
                   missing_keys=sorted(set(index1)-set(index2)), new_keys=sorted(set(index2)-set(index1)))
        return row
    first.index, second.index = index1, index2
    first, second = first.sort_index(), second.sort_index()
    row['keys'] = keys
    maximum_difference = 0.0
    for column in common:
        a, b = first[column], second[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            equal = np.isclose(av, bv, rtol=rtol, atol=atol, equal_nan=True)
            finite = np.isfinite(av) & np.isfinite(bv)
            difference = float(np.max(np.abs(av[finite]-bv[finite]))) if finite.any() else 0.0
            maximum_difference = max(maximum_difference, difference)
            if not equal.all():
                failures = np.flatnonzero(~equal)
                row['mismatches'].append({'column': column, 'count': len(failures),
                    'maximum_absolute_difference': difference,
                    'examples': [{'key': str(first.index[i]), 'reference': _json_value(av[i]),
                                  'computed': _json_value(bv[i])} for i in failures[:5]]})
        else:
            equal = a.map(_key).to_numpy() == b.map(_key).to_numpy()
            if not equal.all():
                failures = np.flatnonzero(~equal)
                row['mismatches'].append({'column': column, 'count': len(failures),
                    'examples': [{'key': str(first.index[i]), 'reference': str(a.iloc[i]),
                                  'computed': str(b.iloc[i])} for i in failures[:5]]})
    row['maximum_absolute_numeric_difference'] = maximum_difference
    row['passed'] = not row['mismatches']
    return row


def _json_value(value):
    return float(value) if np.isfinite(value) else str(value)


def compare_results(publication_root, run_root):
    publication_root, run_root = Path(publication_root), Path(run_root)
    paths = list((publication_root / 'data').rglob('*.csv'))
    paths += list((publication_root / 'patient_extension/verification').glob('*.csv'))
    results = []
    for path in sorted(paths):
        relative = path.relative_to(publication_root)
        if path.name.startswith('partial_'):
            continue
        target = run_root / relative
        if not target.exists():
            results.append({'path': str(relative), 'passed': False, 'error': 'Missing recomputed CSV'})
            continue
        try:
            result = compare_csv(path, target)
        except Exception as error:
            result = {'passed': False, 'error': f'{type(error).__name__}: {error}'}
        results.append({'path': str(relative), **result})
    return results


def _independent_checks(run_root):
    root = Path(run_root).resolve()
    sys.path.insert(0, str(root / 'code'))
    import validate_outputs as outputs
    import validate_revision as revision
    import validate_zenodo_pvi_outputs as patients
    counts, records = {}, []
    current = ''
    original_require = outputs.require
    def count_require(condition, message):
        counts[current] = counts.get(current, 0) + 1
        original_require(condition, message)
    outputs.require = count_require
    patient_require = patients._require
    def count_patient(condition, message):
        counts[current] = counts.get(current, 0) + 1
        patient_require(condition, message)
    patients._require = count_patient
    functions = [
        ('exact_solutions', outputs.validate_exact_solutions),
        ('graph_limit', outputs.validate_graph_limit),
        ('conduction_calibration', outputs.validate_calibration),
        ('original_synthetic', outputs.validate_geometry_reconstruction_outputs),
        ('pvi_capacity_ep', outputs.validate_pvi_outputs),
        ('patient_base', lambda: patients.validate(root / 'data', root / 'figures')),
    ]
    for name, function in functions:
        current = name
        started = time.perf_counter()
        try:
            function()
            item = {'name': name, 'passed': True}
        except Exception as error:
            item = {'name': name, 'passed': False, 'error': f'{type(error).__name__}: {error}'}
        records.append({**item, 'checks': counts.get(name, 0),
                        'seconds': time.perf_counter()-started})
    for name, function in revision.SECTIONS.items():
        revision.CURRENT = name
        started = time.perf_counter()
        try:
            function()
            item = {'name': name, 'passed': True}
        except Exception as error:
            item = {'name': name, 'passed': False, 'error': f'{type(error).__name__}: {error}'}
        records.append({**item, 'checks': revision.COUNTS.get(name, 0),
                        'seconds': time.perf_counter()-started})
    result = {'checks': records, 'notes': revision.NOTES}
    (root / 'validation_independent.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def validate(publication_root, run_root, strict=True):
    publication_root, run_root = Path(publication_root).resolve(), Path(run_root).resolve()
    execution = json.loads((run_root / 'execution_status.json').read_text())
    if execution.get('status') != 'completed':
        raise RuntimeError('Fresh computation has not completed.')
    environment = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[name] = '1'
    command = [sys.executable, str(Path(__file__).resolve()), '--independent', str(run_root)]
    log = run_root / 'logs' / 'independent_validation.log'
    with log.open('w') as output:
        process = subprocess.run(command, cwd=run_root, env=environment,
                                 stdout=output, stderr=subprocess.STDOUT)
    independent_path = run_root / 'validation_independent.json'
    independent = json.loads(independent_path.read_text()) if process.returncode == 0 and independent_path.exists() else {
        'checks': [{'name': 'validation_process', 'passed': False, 'checks': 0,
                    'error': f'exit {process.returncode}; see {log}'}], 'notes': []}
    comparisons = compare_results(publication_root, run_root)
    inventory_path = publication_root / 'study_inventory.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        figure_paths = sorted(Path(item['path']) for item in inventory['archive_figures'])
    else:
        figure_paths = sorted(path.relative_to(publication_root)
                              for path in (publication_root / 'figures').rglob('*.pdf'))
    if not figure_paths:
        raise FileNotFoundError('Figure inventory is empty; provide study_inventory.json or the original figures directory.')
    figures = [{'path': str(path), 'passed': (run_root/path).exists() and (run_root/path).stat().st_size > 1000}
               for path in figure_paths]
    source_manifest = json.loads((run_root / 'fresh_run.json').read_text())
    freshness = source_manifest.get('starting_data_files') == 0 and source_manifest.get('prior_result_checkpoints_copied') is False
    rows = [
        {'Check': 'Fresh computation', 'Count': len(execution['stages']), 'Passed': freshness and all(x['status']=='completed' for x in execution['stages'].values())},
        {'Check': 'Independent numerical checks', 'Count': sum(x['checks'] for x in independent['checks']), 'Passed': all(x['passed'] for x in independent['checks'])},
        {'Check': 'Scientific CSV comparison', 'Count': len(comparisons), 'Passed': all(x['passed'] for x in comparisons)},
        {'Check': 'Figure outputs', 'Count': len(figures), 'Passed': all(x['passed'] for x in figures)},
    ]
    report = {'summary': rows, 'independent': independent, 'csv_comparison': comparisons,
              'figures': figures, 'comparison_tolerances': {'rtol': 5e-8, 'atol': 1e-10},
              'comparison_exclusions': ['execution timing columns and metric/endpoint rows', 'file hashes and paths', 'serialized metadata',
                                        'original incomplete partial_ CSVs'],
              'passed': all(x['Passed'] for x in rows)}
    report_path = run_root / 'notebook_validation.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    summary = pd.DataFrame(rows)
    summary.to_csv(run_root / 'notebook_validation_summary.csv', index=False)
    summary.attrs['report_path'] = str(report_path)
    if strict and not report['passed']:
        failures = [x['name'] for x in independent['checks'] if not x['passed']]
        failures += [x['path'] for x in comparisons if not x['passed']]
        failures += [x['path'] for x in figures if not x['passed']]
        raise RuntimeError(f'Validation failed: {failures}. Details: {report_path}')
    return summary


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--independent':
        _independent_checks(sys.argv[2])
    else:
        raise SystemExit('Usage: notebook_validation.py --independent RUN_DIRECTORY')
