from pathlib import Path
from datetime import datetime, timezone
import os, sys, traceback
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

path=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'notebooks'/'Atrial_Reconstruction.ipynb'
root=path.parent
os.chdir(root)
nb=nbformat.read(path,as_version=4)
shell=InteractiveShell.instance()
nb.metadata['execution']={'engine':'IPython in-process','started':datetime.now(timezone.utc).isoformat()}
for i,cell in enumerate(nb.cells):
    if cell.cell_type!='code':
        continue
    print(datetime.now().isoformat(timespec='seconds'),'cell',i,flush=True)
    started=datetime.now(timezone.utc).isoformat()
    with capture_output(stdout=True,stderr=True,display=True) as captured:
        result=shell.run_cell(cell.source,store_history=True)
    cell.execution_count=result.execution_count
    cell.outputs=[]
    if captured.stdout:
        cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:
        cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for out in captured.outputs:
        cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
    cell.metadata['execution']={'started':started,'completed':datetime.now(timezone.utc).isoformat()}
    error=result.error_before_exec or result.error_in_exec
    if error:
        cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exception(error)))
        nbformat.write(nb,path)
        raise error
    nbformat.write(nb,path)
nb.metadata['execution']['completed']=datetime.now(timezone.utc).isoformat()
nbformat.validate(nb)
nbformat.write(nb,path)
print('Notebook saved:',path,flush=True)
