"""Reproducible ingestion for the pilot; refuses silently changed inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

REPO = 'jameslyman7/NBA'
REF = '20ecc62ad43b9f48fa63fc5460525c316fbbeb86'
FILES = {
    'team_boxscores.csv': ('team', '3fa950bb72769ab242d2e91ba0decba293c7c330'),
    'boxscores.csv': ('player', '255554b4f55de1f290dcff68fbe19dc220614c19'),
    'combined_injuries.csv': ('injury_snapshot', '793f7eb685846d50ca1bb7e8481dc0e631a9f7f9'),
}

def git_blob_sha(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()

def pilot(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (role, expected) in FILES.items():
        url = f'https://raw.githubusercontent.com/{REPO}/{REF}/data/exports/{name}'
        response = requests.get(url, timeout=(15, 60)); response.raise_for_status()
        data = response.content
        actual = git_blob_sha(data)
        if actual != expected:
            raise ValueError(f'{name}: expected git blob {expected}, got {actual}; verify commit and source before accepting')
        (out/name).write_bytes(data)
        manifest.append({'file':name, 'role':role, 'url':url, 'git_blob_sha':actual,
                         'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data),
                         'note':'Current injury snapshot; not a historical availability archive' if role=='injury_snapshot' else 'Published box scores'})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='data/raw/pilot'); args=p.parse_args(); pilot(args.out)
