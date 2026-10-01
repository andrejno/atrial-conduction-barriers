from pathlib import Path
import csv
import html
import json
import re


ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    'tab:graph-certificate': 'Algebraic error bound',
    'tab:configuration': 'Numerical configurations',
    'tab:phase-spatial': 'Phase-field spatial refinement',
    'tab:exact': 'Exact-solution verification',
    'tab:graph-limit': 'Smooth-to-graph convergence',
    'tab:sparse-size': 'Nested acquisition studies',
    'tab:sparse-refinement': 'Reconstruction refinement',
    'tab:sparse-cost': 'Reconstruction cost',
    'tab:phase-diagram': 'Bidirectional EP response',
    'tab:matched-refinement': 'Near-capacity-matched refinement',
    'tab:patient-reconstruction': 'Patient reconstruction',
    'tab:patient-missingness': 'Patient observation coverage',
    'tab:patient-numerics': 'Patient numerical sensitivity',
    'tab:reconstruction-functional': 'Complete-ring reconstruction and EP',
    'tab:surface-verification': 'Surface-operator verification',
    'tab:patient-surface-ep': 'Patient surface EP',
    'tab:calibration': 'EP conduction-scale characterisation',
    'tab:sparse': 'Synthetic reconstruction endpoints',
    'tab:patient-reconstruction-full': 'Patient reconstruction endpoints',
}
CSS = '''<style>
.manuscript-table {font-family:Arial,Helvetica,sans-serif;color:#17202a;background:white;margin:1em 0 1.5em;overflow-x:auto;}
.manuscript-table h3 {font-size:16px;font-weight:600;margin:0 0 12px;}
.manuscript-table table {border-collapse:collapse;font-size:13px;line-height:1.55;width:auto;max-width:100%;border-top:2px solid #34495e;border-bottom:2px solid #34495e;}
.manuscript-table th {text-align:right;font-weight:600;border-bottom:1px solid #aab7c4;background:#f5f7fa;}
.manuscript-table td,.manuscript-table th {padding:7px 13px;vertical-align:top;white-space:nowrap;}
.manuscript-table td:first-child,.manuscript-table th:first-child {text-align:left;}
.manuscript-table td {text-align:right;}
.manuscript-table tr.rule td {border-top:1px solid #ccd4dc;}
.manuscript-table tr.space td {padding-top:15px;}
.manuscript-table tr.group td {text-align:left;font-weight:600;background:#f5f7fa;}
.manuscript-table.configuration table {width:100%;}
.manuscript-table.configuration td,.manuscript-table.configuration th {white-space:normal;text-align:left;min-width:145px;}
.manuscript-table.configuration td:first-child,.manuscript-table.configuration th:first-child {min-width:110px;}
.manuscript-table.surface td {white-space:normal;text-align:left;}
</style>'''


def _group(text, start):
    while start < len(text) and text[start].isspace():
        start += 1
    if text[start] != '{':
        raise ValueError(text[start:start + 50])
    depth = 1
    end = start + 1
    while end < len(text) and depth:
        if text[end] == '{' and text[end - 1] != '\\':
            depth += 1
        elif text[end] == '}' and text[end - 1] != '\\':
            depth -= 1
        end += 1
    if depth:
        raise ValueError('Unclosed TeX group')
    return text[start + 1:end - 1], end


def _command(text, command):
    match = re.search(re.escape('\\' + command) + r'\b', text)
    return _group(text, match.end())[0] if match else ''


def _split(text, delimiter):
    pieces = []
    start = 0
    depth = 0
    i = 0
    while i < len(text):
        if text[i] in '{}' and (i == 0 or text[i - 1] != '\\'):
            depth += 1 if text[i] == '{' else -1
        if depth == 0 and text.startswith(delimiter, i) and (i == 0 or text[i - 1] != '\\'):
            pieces.append(text[start:i])
            i += len(delimiter)
            start = i
        else:
            i += 1
    pieces.append(text[start:])
    return pieces


def _references(source):
    refs = {}
    labels = re.findall(r'\\label\{(tab:[^}]+)\}', source)
    refs.update({key: str(i) for i, key in enumerate(labels, 1)})
    appendix = source.split('\\appendix', 1)[1]
    for i, section in enumerate(re.split(r'\\section\{', appendix)[1:]):
        match = re.search(r'\\label\{(app:[^}]+)\}', section)
        if match:
            refs[match.group(1)] = chr(65 + i)
    return refs


def _normalise(text, refs):
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    text = text.replace(r'\LK', r'\mathcal L_{K}')
    text = text.replace(r'\newline', '<LINEBREAK>')
    while r'\shortstack' in text:
        start = text.index(r'\shortstack')
        content, end = _group(text, start + len(r'\shortstack'))
        text = text[:start] + content.replace(r'\\', '<LINEBREAK>') + text[end:]
    text = re.sub(r'\\(?:c|C)ref\{([^}]+)\}', lambda m: 'Table ' + refs.get(m.group(1), m.group(1)), text)
    text = re.sub(r'\\ref\{([^}]+)\}', lambda m: refs.get(m.group(1), m.group(1)), text)
    text = text.replace('``', '“').replace("''", '”')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _cell_html(text, refs):
    text = _normalise(text, refs)
    parts = re.split(r'(\$[^$]*\$)', text)
    out = []
    for part in parts:
        if part.startswith('$') and part.endswith('$'):
            out.append('<span class="math">\\(' + html.escape(part[1:-1]) + '\\)</span>')
        else:
            part = part.replace(r'\%', '%').replace(r'\&', '&').replace('~', '\u00a0')
            part = part.replace('---', '—').replace('--', '–')
            out.append(html.escape(part).replace('&lt;LINEBREAK&gt;', '<br>'))
    return ''.join(out)


def _cell_plain(text, refs):
    text = _normalise(text, refs).replace('<LINEBREAK>', ' ')
    text = text.replace('$', '').replace('~', ' ')
    text = re.sub(r'([\d.]+)\\times\s*10\^\{([+-]?\d+)\}', r'\1e\2', text)
    text = re.sub(r'\\operatorname\{([^}]+)\}', r'\1', text)
    replacements = {
        r'\mathcal L_{K}': 'L_K', r'\mathrm c': 'c', r'\rm EP': 'EP',
        r'\Delta': 'Δ', r'\delta': 'δ', r'\tau': 'τ', r'\mu': 'μ', r'\nu': 'ν',
        r'\eta': 'η', r'\chi': 'χ', r'\beta': 'β', r'\max': 'max',
        r'\times': '×', r'\circ': '°', r'\ast': '*', r'\%': '%',
        r'\&': '&', r'\,': ' ',
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    text = re.sub(r'([_^])\{([^{}]+)\}', r'\1\2', text)
    text = text.replace('^°', '°').replace('^*', '*')
    text = text.replace('---', '—').replace('--', '–')
    text = text.replace('{', '').replace('}', '')
    if '\\' in text:
        raise ValueError('Unconverted TeX in CSV cell: ' + text)
    return re.sub(r'\s+', ' ', text).strip()


def _rows(table):
    start = table.index(r'\begin{tabular}') + len(r'\begin{tabular}')
    _, start = _group(table, start)
    body = table[start:table.index(r'\end{tabular}', start)]
    rows = []
    for raw in _split(body, r'\\'):
        classes = []
        if r'\midrule' in raw:
            classes.append('rule')
        if r'\addlinespace' in raw:
            classes.append('space')
        raw = re.sub(r'\\(?:toprule|midrule|bottomrule|addlinespace)\b', '', raw)
        raw = re.sub(r'^\s*\[[^\]]+\]', '', raw).strip()
        if not raw:
            continue
        cells = []
        for cell in _split(raw, '&'):
            cell = cell.strip()
            span = 1
            if cell.startswith(r'\multicolumn'):
                count, pos = _group(cell, len(r'\multicolumn'))
                _, pos = _group(cell, pos)
                content, pos = _group(cell, pos)
                if cell[pos:].strip():
                    raise ValueError(cell)
                span = int(count)
                cell = content
                classes.append('group')
            cells.append((cell, span))
        rows.append((cells, classes))
    ncols = sum(span for _, span in rows[0][0])
    if any(sum(span for _, span in cells) != ncols for cells, _ in rows):
        raise ValueError('Inconsistent table width')
    return rows


def generate_tables(manuscript=None, output_dir=None):
    manuscript = Path(manuscript) if manuscript else ROOT / 'manuscript.tex'
    output_dir = Path(output_dir) if output_dir else ROOT / 'notebook_tables'
    output_dir.mkdir(parents=True, exist_ok=True)
    source = manuscript.read_text()
    refs = _references(source)
    tables = re.findall(r'\\begin\{table\}(?:\[[^\]]*\])?[\s\S]*?\\end\{table\}', source)
    labels = [_command(table, 'label') for table in tables]
    if labels != list(TITLES):
        raise ValueError('Manuscript table inventory differs from the expected 19 tables')
    records = []
    for number, table in enumerate(tables, 1):
        label = _command(table, 'label')
        title = TITLES[label]
        rows = _rows(table)
        css_class = ' configuration' if label == 'tab:configuration' else (' surface' if label == 'tab:surface-verification' else '')
        parts = [CSS, f'<div class="manuscript-table{css_class}"><h3>Table {number}. {html.escape(title)}</h3><table>']
        for row_index, (cells, classes) in enumerate(rows):
            tag = 'th' if row_index == 0 else 'td'
            if row_index == 0:
                parts.append('<thead>')
            elif row_index == 1:
                parts.append('<tbody>')
            parts.append('<tr class="' + ' '.join(classes) + '">')
            for value, span in cells:
                span_text = f' colspan="{span}"' if span > 1 else ''
                parts.append(f'<{tag}{span_text}>' + _cell_html(value, refs) + f'</{tag}>')
            parts.append('</tr>')
            if row_index == 0:
                parts.append('</thead>')
        parts.append('</tbody></table></div>')
        markup = ''.join(parts)
        stem = f'table_{number:02d}_' + label.split(':', 1)[1].replace('-', '_')
        texpath = output_dir / (stem + '.tex')
        htmlpath = output_dir / (stem + '.html')
        csvpath = output_dir / (stem + '.csv')
        texpath.write_text(table + '\n')
        htmlpath.write_text('<!doctype html><html><head><meta charset="utf-8"><title>' + html.escape(title) + '</title><script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]]}};</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script></head><body>' + markup + '</body></html>')
        group_column = any('group' in classes for _, classes in rows)
        with csvpath.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow((['Group'] if group_column else []) + [_cell_plain(value, refs) for value, _ in rows[0][0]])
            group = ''
            for cells, classes in rows[1:]:
                if 'group' in classes:
                    group = _cell_plain(cells[0][0], refs)
                    continue
                writer.writerow(([group] if group_column else []) + [_cell_plain(value, refs) for value, _ in cells])
        records.append({
            'number': number, 'label': label, 'shorttitle': title,
            'html': markup, 'csvpath': str(csvpath), 'texpath': str(texpath),
            'htmlpath': str(htmlpath), 'caption_tex': _command(table, 'caption'),
            'rows': len(rows) - 1, 'columns': sum(span for _, span in rows[0][0]),
        })
    metadata = [{key: (Path(value).name if key in {'csvpath', 'texpath', 'htmlpath'} else value) for key, value in record.items() if key != 'html'} for record in records]
    (output_dir / 'tables.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n')
    return records


if __name__ == '__main__':
    generate_tables()
