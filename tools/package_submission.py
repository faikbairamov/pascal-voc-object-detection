"""Build a portable submission ZIP including the selected inference weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRS = {'configs','splits','voc_detection','notebooks','tests','reports','models','tools','output'}
ALLOWED_FILES = {'README.md','SUBMISSION.md','AGENTS.md','pyproject.toml','requirements.txt','requirements-serving.txt',
                 'Dockerfile','.dockerignore','.gitignore','.gitattributes'}


def main():
    if not (ROOT / 'models/best.pt').is_file():
        raise SystemExit('Export models/best.pt before packaging.')
    paths = []
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if relative.parts[0] not in ALLOWED_DIRS and str(relative) not in ALLOWED_FILES:
            continue
        if not path.is_file() or '__pycache__' in relative.parts or '.ipynb_checkpoints' in relative.parts:
            continue
        if relative.parts[0] == 'models' and relative.name not in {'best.pt','best.metadata.json','README.md'}:
            continue
        if path.suffix in {'.pyc','.pth','.pt'} and str(relative) != 'models/best.pt':
            continue
        paths.append(path)
    manifest = {'format': 1, 'model': 'models/best.pt', 'files': {}}
    for path in sorted(paths):
        manifest['files'][path.relative_to(ROOT).as_posix()] = {
            'bytes': path.stat().st_size,
            'sha256': hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}
    destination = ROOT / 'dist/pascal-voc-submission.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, 'pascal-voc-object-detection/' + path.relative_to(ROOT).as_posix())
        archive.writestr('pascal-voc-object-detection/MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed.')
    digest = hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()
    destination.with_suffix('.zip.sha256').write_text(f'{digest}  {destination.name}\n')
    print(json.dumps({'path': str(destination), 'bytes':destination.stat().st_size,
                      'sha256':digest,'files':len(paths)},indent=2))


if __name__ == '__main__': main()
