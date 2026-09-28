#!/usr/bin/env python3
"""Fetch only the SWiG images referenced by local data, using HTTP Range requests.

Image source: https://github.com/allenai/swig#1-swig-dataset
Remote ZIP API: https://github.com/gtsystem/python-remotezip#usage
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import zlib
from zipfile import ZipFile, ZipInfo

OFFICIAL_ARCHIVE = 'https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip'


def required_images(paths):
    names = set()
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()] if path.suffix == '.jsonl' else json.load(stream)
        for row in rows:
            for image in row.get('images', []):
                name = Path(image).name
                if name in {'', '.', '..'}:
                    raise ValueError(f'Invalid image name in {path}')
                names.add(name)
    if not names:
        raise ValueError('Inputs contain no images')
    return names


def select_members(archive, required):
    found = {}
    for info in archive.infolist():
        name = Path(info.filename).name
        if info.is_dir() or name not in required:
            continue
        if name in found:
            raise ValueError(f'Ambiguous archive image basename: {name}')
        if info.file_size > 25 * 1024 * 1024:
            raise ValueError(f'Unexpectedly large image: {info.filename}')
        found[name] = info
    missing = required - found.keys()
    if missing:
        raise ValueError(f'{len(missing)} requested images are absent from archive: {sorted(missing)[:10]}')
    return sorted(found.values(), key=lambda info: info.header_offset)


def fetch_member(archive, info, destination, read_payload=None):
    from PIL import Image
    path = destination / Path(info.filename).name
    payload = path.read_bytes() if path.is_file() else b''
    cached = len(payload) == info.file_size and zlib.crc32(payload) == info.CRC
    if not cached:
        payload = read_payload(info) if read_payload else archive.read(info.filename)
    if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
        raise ValueError(f'Archive checksum failed: {info.filename}')
    with Image.open(BytesIO(payload)) as image:
        image.verify()
    if not cached:
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_bytes(payload)
        temporary.replace(path)
    return {'name': path.name, 'sha256': hashlib.sha256(payload).hexdigest(),
            'bytes': len(payload), 'cached': cached}


def decode_member(payload, info):
    """Decode a single ZIP entry fetched by its central-directory byte range."""
    header = struct.unpack('<4s5H3I2H', payload[:30])
    if header[0] != b'PK\x03\x04' or header[2] & 1 or header[3] != info.compress_type:
        raise ValueError(f'Unsupported or inconsistent ZIP header: {info.filename}')
    offset = 30 + header[-2] + header[-1]
    compressed = payload[offset:offset + info.compress_size]
    if len(compressed) != info.compress_size:
        raise ValueError(f'Truncated ZIP member: {info.filename}')
    if info.compress_type == 0:
        return compressed
    if info.compress_type == 8:
        return zlib.decompress(compressed, -15)
    raise ValueError(f'Unsupported ZIP compression: {info.compress_type}')


def http_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])))
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/swig/images_512'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official ZIP')
    parser.add_argument('--index', type=Path, help='Reusable ZIP metadata cache; created if absent')
    parser.add_argument('--index-only', action='store_true', help='Prepare the index without fetching images')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error('--workers must be between 1 and 16')
    if args.index_only and not args.index:
        parser.error('--index-only requires --index')
    required = required_images(args.data)
    archive_id = str(args.archive.resolve()) if args.archive else OFFICIAL_ARCHIVE

    def open_archive():
        if args.archive:
            return ZipFile(args.archive)
        from remotezip import RemoteZip
        return RemoteZip(OFFICIAL_ARCHIVE, timeout=90)

    if args.index and args.index.exists():
        catalog = json.loads(args.index.read_text())
        if catalog['archive'] != archive_id or set(catalog['required']) != required:
            raise ValueError('Index does not match requested archive/images')
        members, ends = [], {}
        for entry in catalog['members']:
            info = ZipInfo(entry['filename'])
            for field in ('header_offset', 'compress_size', 'file_size', 'CRC', 'compress_type'):
                setattr(info, field, entry[field])
            members.append(info)
            ends[info.header_offset] = entry['end']
    else:
        with open_archive() as archive:
            members = select_members(archive, required)
            offsets = sorted(info.header_offset for info in archive.infolist()) + [archive.start_dir]
            ends = dict(zip(offsets, offsets[1:]))
        if args.index:
            catalog = {'archive': archive_id, 'required': sorted(required), 'members': []}
            for info in members:
                entry = {field: getattr(info, field) for field in
                         ('filename', 'header_offset', 'compress_size', 'file_size', 'CRC', 'compress_type')}
                entry['end'] = ends[info.header_offset]
                catalog['members'].append(entry)
            args.index.parent.mkdir(parents=True, exist_ok=True)
            args.index.write_text(json.dumps(catalog) + '\n')
    if len(members) != len(required) or {Path(i.filename).name for i in members} != required:
        raise ValueError('Index member list is incomplete or duplicated')
    print(json.dumps({'requested_images': len(members),
                      'compressed_bytes': sum(i.compress_size for i in members)}), flush=True)
    if args.index_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = [members[i::args.workers] for i in range(args.workers)]

    def worker(items):
        results = []
        # The large central directory is read only once, shared by all workers.
        with ZipFile(args.archive) if args.archive else http_session() as resource:
            def read_payload(info):
                start, end = info.header_offset, ends[info.header_offset] - 1
                if end - start > 26 * 1024 * 1024:
                    raise ValueError(f'Unexpectedly large ZIP byte range: {info.filename}')
                response = resource.get(OFFICIAL_ARCHIVE, headers={'Range': f'bytes={start}-{end}'}, timeout=90)
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != end - start + 1:
                    raise ValueError('Archive server returned an invalid HTTP byte range')
                return decode_member(response.content, info)

            for index, info in enumerate(items, 1):
                results.append(fetch_member(resource, info, args.output_dir,
                                            None if args.archive else read_payload))
                if index % 100 == 0:
                    print(f'Worker progress: {index}/{len(items)}', flush=True)
        return results

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker, chunk) for chunk in chunks if chunk]):
            results.extend(future.result())
    report = {'archive': str(args.archive) if args.archive else OFFICIAL_ARCHIVE,
              'images': len(results), 'downloaded': sum(not r['cached'] for r in results),
              'files': sorted(results, key=lambda r: r['name'])}
    manifest = args.output_dir.parent / 'download_manifest.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(results)} images. Manifest: {manifest}', flush=True)


if __name__ == '__main__':
    main()
