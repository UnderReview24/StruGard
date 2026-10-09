"""Export reports, complete tabular evidence and source; keep data/weights remote."""
import _bootstrap
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from src.continual.v2_trainer import VARIANTS
from src.utils.logging import save_json


if __name__=='__main__':
    root=Path.cwd();runs=root/'outputs/v2_5k';summary=runs/'summary'
    queue=json.loads((runs/'queue_status.json').read_text())
    if queue['status']!='complete' or queue['completed']!=VARIANTS:
        raise ValueError('The seven-run queue is not complete')
    for name in VARIANTS:
        if json.loads((runs/name/'audit.json').read_text())['status']!='passed':
            raise ValueError(f'Unverified experiment {name}')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    (runs/'final_source_commit.txt').write_text(commit+'\n')
    (runs/'git_history.txt').write_text(subprocess.check_output(['git','log','--oneline','-n','20'],text=True))
    files={}
    for path in sorted(runs.rglob('*')):
        if path.is_file() and path.name!='file_checksums.json' and path.suffix in {'.csv','.json','.yaml','.md','.txt','.png'}:
            files[str(path.relative_to(runs))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(summary/'file_checksums.json',files)
    output=root/'outputs/SP_Crystal_V2_5k_results.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(summary.iterdir()):
            if path.is_file():z.write(path,'report/'+path.name)
        for path in sorted(runs.rglob('*')):
            if path.is_file() and summary not in path.parents and path.suffix in {'.csv','.json','.yaml','.md','.txt'}:
                z.write(path,'evidence/'+str(path.relative_to(runs)))
        for name in ['v2_all_tests.log','v2_unit_tests.log','v2_prepare.log','v2_suite.log','environment.json','environment.txt','environment_packages.json']:
            path=root/'outputs'/name
            if path.exists():z.write(path,'evidence/'+name)
        z.write(root/'outputs/pilot/rae/final_results.csv','evidence/archived_old_rae/final_results.csv')
        for item in subprocess.check_output(['git','ls-files'],text=True).splitlines():
            path=root/item
            if path.is_file():z.write(path,'source/'+item)
    print('EXPORT',output,output.stat().st_size,flush=True)
    print('SHA256',hashlib.sha256(output.read_bytes()).hexdigest(),flush=True)
