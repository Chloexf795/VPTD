#!/usr/bin/env python3
"""Download official Qwen weights from ModelScope and verify repository checksums."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--revision', default='master')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import requests
    from modelscope import snapshot_download
    response = requests.get(f'https://modelscope.cn/api/v1/models/{args.model}/repo/files',
                            params={'Recursive': 'true', 'Revision': args.revision}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    if metadata.get('Code') != 200:
        raise RuntimeError(f'ModelScope metadata failed: {metadata.get("Message")}')
    files = [item for item in metadata['Data']['Files'] if item['Type'] == 'blob']
    if not files or not any(item['Path'].endswith('.safetensors') for item in files):
        raise RuntimeError('Official repository has no safetensors weights')
    print(f'Downloading {len(files)} official files ({sum(i["Size"] for i in files):,} bytes)', flush=True)
    snapshot_download(args.model, revision=args.revision, local_dir=str(args.output_dir), max_workers=4)
    verified = []
    for item in files:
        path = (args.output_dir / item['Path']).resolve()
        if not path.is_relative_to(args.output_dir.resolve()):
            raise ValueError('Repository file escapes model directory')
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                checksum.update(chunk)
        if path.stat().st_size != item['Size'] or checksum.hexdigest() != item['Sha256']:
            raise ValueError(f'Official file checksum mismatch: {item["Path"]}')
        verified.append({'path': item['Path'], 'sha256': checksum.hexdigest(),
                         'revision': item['Revision'], 'size': item['Size']})
    manifest = {'provider': 'ModelScope', 'model': args.model, 'requested_revision': args.revision,
                'files': verified}
    (args.output_dir / 'vptd_snapshot.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Verified {len(verified)} model files against official SHA-256 checksums', flush=True)


if __name__ == '__main__':
    main()
