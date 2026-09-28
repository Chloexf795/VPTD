#!/usr/bin/env python3
"""Fetch and verify the pinned official Vision-OPD repository; never overwrite edits."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_checkout(path: Path, lock: dict):
    commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if commit != lock['commit'] or dirty:
        raise ValueError(f'Framework checkout differs from lock or has tracked edits: {path}')


def main():
    lock = json.loads((ROOT / 'configs/framework.lock.json').read_text())
    destination = ROOT / 'external/Vision-OPD'
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'remote', 'add', 'origin', lock['repository']], check=True)
        subprocess.run(['git', '-C', str(destination), 'fetch', '--depth', '1', 'origin', lock['commit']], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
    verify_checkout(destination, lock)
    print(f'Verified {lock["name"]} {lock["commit"]}\n{destination}')


if __name__ == '__main__':
    main()
