"""Package audited results and reproducibility sources; retain weights remotely."""
import _bootstrap
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import save_json


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=load_config(args.config);root=Path(cfg['sgcr']['output_root'])
    assert json.loads((root/'audit.json').read_text())['status']=='passed'
    assert json.loads((root/'progress.json').read_text())['status']=='complete'
    stop=json.loads((root/'training_stop_verification.json').read_text())
    assert stop['training_stopped'] and stop['training_source_unchanged']
    weights={str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(root.rglob('*.pt'))}
    save_json(root/'remote_weights_manifest.json',weights)
    (root/'WEIGHTS_LOCATION.txt').write_text('Model weights remain on the server. Exact paths, sizes and SHA-256 hashes are in remote_weights_manifest.json.\n')
    files={p.relative_to(root).as_posix():p for p in root.rglob('*')
        if p.is_file() and p.suffix not in ('.pt','.npz','.pid','.zip') and '__pycache__' not in p.parts
        and p.name not in ('packaging.log','package_verification.json')}
    for folder in ('src','scripts','configs','tests'):
        for p in Path(folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.yaml','.md') and '__pycache__' not in p.parts:
                files['source/'+p.as_posix()]=p
    for name in ['README.md','SGCR_IMPLEMENTATION.md','SGCR_10K_IMPLEMENTATION.md','requirements.txt','third_party/ALIGNN_COMMIT.txt']:
        p=Path(name)
        if p.is_file():files['source/'+name]=p
    assignment=Path(cfg['domains']['assignments'])
    for p in [assignment,Path(cfg['domains']['splits']),assignment.with_suffix('.summary.csv')]:
        files[p.as_posix()]=p
    manifest={name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for name,p in sorted(files.items())}
    destination=root.parent/'SGCR_10k_4domains_memory500_results.zip'
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,p in sorted(files.items()):archive.write(p,name)
        archive.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name,item in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==item['sha256']
    summary=dict(archive=str(destination),bytes=destination.stat().st_size,sha256=sha256(destination),
        verified_files=len(manifest),weights_included=False,remote_weight_files=len(weights))
    save_json(root/'package_verification.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
