from pathlib import Path
import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser(description='Set the repository and GitHub Pages links.')
    parser.add_argument('repository', help='OWNER/REPOSITORY or its https://github.com URL')
    args = parser.parse_args()
    name = args.repository.strip().removeprefix('https://github.com/').removesuffix('.git').strip('/')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+', name):
        parser.error('Use OWNER/REPOSITORY or https://github.com/OWNER/REPOSITORY.')
    owner, repository = name.split('/')
    root = Path(__file__).resolve().parents[1]
    url = f'https://{owner}.github.io/' + ('' if repository.lower() == f'{owner.lower()}.github.io' else repository + '/')
    readme = root / 'README.md'
    text = readme.read_text()
    text, count = re.subn(r'\[(?:Website preview|Website)\]\([^\n)]*\)', f'[Website]({url})', text, count=1)
    if count != 1:
        raise ValueError('The website link was not found in README.md.')
    readme.write_text(text)
    (root / 'docs/site-config.js').write_text('window.RESEARCH_REPOSITORY = ' + json.dumps('https://github.com/' + name) + ';\n')
    index = root / 'docs/index.html'
    text = re.sub(r'<meta property="og:(?:url|image)"[^>]*>\n?', '', index.read_text())
    tags = f'<meta property="og:url" content="{url}">\n<meta property="og:image" content="{url}images/social-preview.png">\n'
    text = text.replace('</head>', tags + '</head>')
    index.write_text(text)
    print(url)


if __name__ == '__main__':
    main()
