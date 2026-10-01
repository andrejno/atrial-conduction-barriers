"""Write a complete content manifest after compilation and validation."""
from pathlib import Path
import hashlib,json,platform
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'SHA256SUMS.txt','revision_manifest.json'}
def main():
    records={}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.name in EXCLUDED or any(part.startswith('.') or part in {'__pycache__','tmp'} for part in p.relative_to(ROOT).parts):
            continue
        if p.suffix in {'.aux','.log','.fdb_latexmk','.fls','.out','.toc','.synctex','.pyc','.blg'}:
            continue
        b=p.read_bytes(); records[p.relative_to(ROOT).as_posix()]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    manifest={'revision':'Journal of Scientific Computing, 6 September 2026: shortened main text and consolidated patient figures','publication_experiments':[1,2,3,4,5,6,7],'supporting_experiment_in_appendix':3,'exploratory_appendix':'P3 reconstruction-to-EP','presentation_record':'data/presentation_revision_validation.json','python':platform.python_version(),'original_metadata_scope':'data/run_metadata.json retains the original six-study schema; this manifest covers the revision.','records':records}
    (ROOT/'data/revision_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=[f'{v["sha256"]}  {k}' for k,v in records.items()]
    b=(ROOT/'data/revision_manifest.json').read_bytes()
    lines.append(hashlib.sha256(b).hexdigest()+'  data/revision_manifest.json')
    (ROOT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
    print(f'Manifest written for {len(records)} files')
if __name__=='__main__':main()
