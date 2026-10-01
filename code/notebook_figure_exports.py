from pathlib import Path
import os
import tempfile
from unittest.mock import patch

import pandas as pd
import run_core


def main():
    destination = run_core.FIG
    temporary_root = run_core.ROOT / 'tmp'
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
        with patch.object(run_core, 'FIG', Path(folder)), \
             patch.object(run_core, 'make_exact_figure'), \
             patch.object(run_core, 'make_conduction_figure'):
            run_core.make_core_figures(None, None, pd.read_csv(run_core.DATA / 'graph_limit.csv'), None)
        for suffix in ('pdf', 'png'):
            source = Path(folder) / f'fig2_graph_limit.{suffix}'
            if source.stat().st_size < 1000:
                raise RuntimeError(f'Incomplete figure export: {source.name}')
            os.replace(source, destination / source.name)
    print('Graph-limit figure exported.')


if __name__ == '__main__':
    main()
