"""Check package completeness, locked bytes and Python syntax."""
from pathlib import Path
import ast
import hashlib
import json


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
    failures = []
    for item in manifest['files']:
        path = root/item['path']
        if not path.is_file():
            failures.append('missing: ' + item['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append('hash mismatch: ' + item['path'])
        elif path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=item['path'])
    if failures:
        raise SystemExit('\n'.join(failures))
    print(json.dumps({'status': 'PASS', 'files_checked': len(manifest['files']),
                      'scope': 'file integrity and Python syntax; not statistical validation'}))


if __name__ == '__main__':
    main()
