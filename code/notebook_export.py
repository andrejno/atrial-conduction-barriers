from pathlib import Path
from html.parser import HTMLParser
import base64
import html
import json
import mimetypes
import re


STYLE = '''
:root {color-scheme:light;}
* {box-sizing:border-box;}
body {margin:0;background:#fff;color:#17202a;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.65;}
main {max-width:1248px;margin:38px auto 80px;padding:0 24px;}
h1 {font-size:30px;line-height:1.25;max-width:1100px;font-weight:600;margin:0 0 30px;}
h2 {font-size:23px;line-height:1.35;margin:42px 0 18px;font-weight:600;}
h3 {font-size:18px;margin:28px 0 14px;font-weight:600;}
p {margin:12px 0;}
a {color:#225c91;}
.markdown-cell,.code-cell {margin:16px 0;}
.code-cell {break-inside:avoid-page;}
details {margin:12px 0 14px;border:1px solid #dbe2e8;border-radius:5px;background:#f8fafc;}
summary {cursor:pointer;padding:7px 12px;font-size:12px;color:#5d6b7b;user-select:none;}
pre {white-space:pre-wrap;overflow-wrap:anywhere;overflow-x:auto;margin:0;padding:14px 16px;font-size:12px;line-height:1.55;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;}
code {font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.9em;}
.output {margin:12px 0;max-width:100%;overflow-x:auto;}
.output img,.output svg,.markdown-cell img {display:block;max-width:100%;height:auto;margin:0 auto;}
.output video,.markdown-cell video {display:block;width:100%;height:auto;max-width:100%;margin:0 auto;background:#080c12;}
.output pre {padding:10px 0;color:#334155;}
.output.error pre {color:#9f2323;}
.math-block {overflow-x:auto;text-align:center;margin:22px 0;}
table {max-width:100%;}
@media (max-width:650px) {main {padding:0 14px;margin-top:24px;}h1 {font-size:24px;}h2 {font-size:20px;}}
@media print {main {max-width:none;margin:0;padding:0;}details {display:none;}h2 {break-after:avoid-page;}.output {overflow:visible;}}
'''


def _text(value):
    return ''.join(value) if isinstance(value, list) else str(value or '')


def _data_url(path):
    mime = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def _resource(value, base):
    if not value or re.match(r'^(?:[a-z][a-z0-9+.-]*:|//|#)', value, re.I):
        return value
    path = (base / value).resolve()
    return _data_url(path) if path.is_file() else value


class _EmbedHTML(HTMLParser):
    def __init__(self, base):
        super().__init__(convert_charrefs=False)
        self.base = base
        self.fragments = []

    def _tag(self, tag, attrs, closed=False):
        values = []
        for key, value in attrs:
            if key == 'src' and tag in {'img', 'video', 'audio', 'source'}:
                value = _resource(value, self.base)
            if key in {'href', 'xlink:href'} and tag == 'image':
                value = _resource(value, self.base)
            values.append(key if value is None else key + '="' + html.escape(value, quote=True) + '"')
        self.fragments.append('<' + tag + (' ' if values else '') + ' '.join(values) + (' />' if closed else '>'))

    def handle_starttag(self, tag, attrs):
        self._tag(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._tag(tag, attrs, True)

    def handle_endtag(self, tag):
        self.fragments.append('</' + tag + '>')

    def handle_data(self, data):
        self.fragments.append(data)

    def handle_entityref(self, name):
        self.fragments.append('&' + name + ';')

    def handle_charref(self, name):
        self.fragments.append('&#' + name + ';')

    def handle_comment(self, data):
        self.fragments.append('<!--' + data + '-->')

    def handle_decl(self, decl):
        self.fragments.append('<!' + decl + '>')

    def handle_pi(self, data):
        self.fragments.append('<?' + data + '>')


def _embed_html(value, base):
    parser = _EmbedHTML(base)
    parser.feed(value)
    parser.close()
    return ''.join(parser.fragments)


def _inline(source, base, attachments=None):
    replacements = []

    def protect(value):
        index = len(replacements)
        replacements.append(value)
        return f'\x00{index}\x00'

    source = re.sub(r'(?<!\\)\$(?!\$)(.+?)(?<!\\)\$', lambda m: protect('\\(' + html.escape(m.group(1)) + '\\)'), source)
    source = re.sub(r'\\\((.*?)\\\)', lambda m: protect('\\(' + html.escape(m.group(1)) + '\\)'), source)
    source = re.sub(r'`([^`]+)`', lambda m: protect('<code>' + html.escape(m.group(1)) + '</code>'), source)

    def picture(match):
        alt, url = match.groups()
        if url.startswith('attachment:') and attachments:
            attachment = attachments.get(url.split(':', 1)[1], {})
            for mime in ('image/png', 'image/jpeg', 'image/gif', 'image/svg+xml'):
                if mime in attachment:
                    value = _text(attachment[mime])
                    url = 'data:' + mime + ';base64,' + (base64.b64encode(value.encode()).decode() if mime == 'image/svg+xml' else value)
                    break
        else:
            url = _resource(url, base)
        return protect('<img alt="' + html.escape(alt, quote=True) + '" src="' + html.escape(url, quote=True) + '">')

    source = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', picture, source)
    source = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: protect('<a href="' + html.escape(m.group(2), quote=True) + '">' + html.escape(m.group(1)) + '</a>'), source)
    source = html.escape(source)
    source = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', source)
    source = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', source)
    for index, value in enumerate(replacements):
        source = source.replace(f'\x00{index}\x00', value)
    return source


def _markdown(source, base, attachments=None):
    pieces = []
    parts = re.split(r'(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|```[\s\S]*?```)', source)
    for part in parts:
        if not part.strip():
            continue
        if part.startswith('$$') and part.endswith('$$'):
            pieces.append('<div class="math-block">\\[' + html.escape(part[2:-2].strip()) + '\\]</div>')
            continue
        if part.startswith(r'\[') and part.endswith(r'\]'):
            pieces.append('<div class="math-block">' + html.escape(part) + '</div>')
            continue
        if part.startswith('```'):
            pieces.append('<pre><code>' + html.escape(part[3:-3].split('\n', 1)[-1]) + '</code></pre>')
            continue
        for block in re.split(r'\n\s*\n', part.strip()):
            lines = block.splitlines()
            if all(re.match(r'^\s*[-*]\s+', line) for line in lines):
                pieces.append('<ul>' + ''.join('<li>' + _inline(re.sub(r'^\s*[-*]\s+', '', line), base, attachments) + '</li>' for line in lines) + '</ul>')
                continue
            if all(re.match(r'^\s*\d+\.\s+', line) for line in lines):
                pieces.append('<ol>' + ''.join('<li>' + _inline(re.sub(r'^\s*\d+\.\s+', '', line), base, attachments) + '</li>' for line in lines) + '</ol>')
                continue
            for line in lines:
                match = re.match(r'^(#{1,6})\s+(.+)$', line)
                if match:
                    level = len(match.group(1))
                    pieces.append(f'<h{level}>' + _inline(match.group(2), base, attachments) + f'</h{level}>')
                elif line.lstrip().startswith(('<div', '<video', '<img', '<table', '<style')):
                    pieces.append(_embed_html(line, base))
                else:
                    pieces.append('<p>' + _inline(line, base, attachments) + '</p>')
    return ''.join(pieces)


def _output(output, base):
    kind = output.get('output_type')
    if kind == 'stream':
        return '<pre>' + html.escape(_text(output.get('text'))) + '</pre>'
    if kind == 'error':
        text = '\n'.join(output.get('traceback', [])) or output.get('ename', '') + ': ' + output.get('evalue', '')
        text = re.sub(r'\x1b\[[0-9;]*m', '', text)
        return '<pre>' + html.escape(text) + '</pre>'
    data = output.get('data', {})
    if 'text/html' in data:
        return _embed_html(_text(data['text/html']), base)
    if 'image/svg+xml' in data:
        return _embed_html(_text(data['image/svg+xml']), base)
    for mime in ('image/png', 'image/jpeg', 'image/jpg', 'image/gif'):
        if mime in data:
            return '<img alt="" src="data:' + mime + ';base64,' + _text(data[mime]).replace('\n', '') + '">'
    if 'text/latex' in data:
        value = _text(data['text/latex'])
        if value.strip().startswith(('$', r'\(', r'\[')):
            return '<div class="math-block">' + html.escape(value) + '</div>'
        return '<div class="math-block">\\[' + html.escape(value) + '\\]</div>'
    if 'application/json' in data:
        return '<pre>' + html.escape(json.dumps(data['application/json'], indent=2, ensure_ascii=False)) + '</pre>'
    if 'text/plain' in data:
        return '<pre>' + html.escape(_text(data['text/plain'])) + '</pre>'
    return ''


def export_html(notebook_path, output_path):
    notebook_path = Path(notebook_path).resolve()
    output_path = Path(output_path).resolve()
    notebook = json.loads(notebook_path.read_text())
    title = notebook.get('metadata', {}).get('title') or notebook_path.stem.replace('_', ' ')
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            match = re.search(r'^#\s+(.+)$', _text(cell.get('source')), re.M)
            if match:
                title = match.group(1)
                break
    content = []
    for cell in notebook.get('cells', []):
        kind = cell.get('cell_type')
        source = _text(cell.get('source'))
        if kind == 'markdown':
            content.append('<section class="markdown-cell">' + _markdown(source, notebook_path.parent, cell.get('attachments')) + '</section>')
        elif kind == 'code':
            content.append('<section class="code-cell">')
            if source.strip():
                content.append('<details><summary>Code</summary><pre><code>' + html.escape(source) + '</code></pre></details>')
            for output in cell.get('outputs', []):
                rendered = _output(output, notebook_path.parent)
                if rendered:
                    css = ' error' if output.get('output_type') == 'error' else ''
                    content.append('<div class="output' + css + '">' + rendered + '</div>')
            content.append('</section>')
        elif kind == 'raw' and source.strip():
            content.append('<pre>' + html.escape(source) + '</pre>')
    mathjax = {'tex': {'inlineMath': [[r'\(', r'\)'], ['$', '$']], 'displayMath': [[r'\[', r'\]'], ['$$', '$$']], 'processEscapes': True}, 'options': {'skipHtmlTags': ['script', 'noscript', 'style', 'textarea', 'pre', 'code']}}
    document = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + html.escape(title) + '</title><style>' + STYLE + '</style><script>window.MathJax=' + json.dumps(mathjax) + ';</script><script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script></head><body><main>' + ''.join(content) + '</main></body></html>'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding='utf-8')
    return output_path
