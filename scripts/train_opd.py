#!/usr/bin/env python3
"""Launch the locked upstream verl trainer with ACE/SWiG configuration and preflight."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from setup_framework import verify_checkout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vptd_eae.datasets import require_unambiguous_inputs


def resume_compatible(previous: dict, current: dict) -> bool:
    """Keep training fixed; allow budget/storage changes and the equivalent AdamW kernel."""
    if previous == current:
        return True
    key = 'trainer.total_training_steps='
    storage_keys = ('trainer.save_freq=', 'trainer.max_actor_ckpt_to_keep=')
    def training_signature(launch):
        command = launch['command']
        low_memory_adam = {'actor_rollout_ref.actor.optim.optimizer="LowMemoryAdamW"',
                           'actor_rollout_ref.actor.optim.optimizer_impl="vptd_eae.optim"'}
        if low_memory_adam.issubset(command):
            command = [arg for arg in command if arg not in low_memory_adam]
        steps = [int(arg.removeprefix(key)) for arg in command if arg.startswith(key)]
        return {**launch, 'command': [arg for arg in command if not arg.startswith((key, *storage_keys))]}, steps
    old, old_steps = training_signature(previous)
    new, new_steps = training_signature(current)
    return old == new and len(old_steps) == len(new_steps) == 1 and new_steps[0] >= old_steps[0]


def build_overrides(args) -> list[str]:
    static = args.mode == 'ace-static-opd'
    model = str(Path(args.model).resolve()) if Path(args.model).exists() else args.model
    config = {
        'data.train_files': [str(args.data.resolve())], 'data.val_files': [],
        'data.train_batch_size': args.batch_size, 'data.max_prompt_length': args.max_prompt_tokens,
        'data.max_response_length': args.max_response_tokens, 'data.filter_overlong_prompts': False,
        'data.truncation': 'error', 'data.trust_remote_code': False,
        'data.return_multi_modal_inputs': True, 'data.dataloader_num_workers': args.dataloader_workers,
        'actor_rollout_ref.model.path': model, 'actor_rollout_ref.model.trust_remote_code': False,
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.ppo_mini_batch_size': args.batch_size,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.use_dynamic_bsz': True,
        'actor_rollout_ref.actor.ppo_max_token_len_per_gpu': args.max_prompt_tokens + args.max_response_tokens,
        'actor_rollout_ref.actor.optim.lr': args.lr,
        'actor_rollout_ref.actor.optim.lr_warmup_steps': 10,
        'actor_rollout_ref.actor.fsdp_config.param_offload': not args.keep_actor_on_gpu,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.actor.policy_loss.loss_mode': 'vopd',
        'actor_rollout_ref.actor.calculate_entropy': False,
        'actor_rollout_ref.actor.self_distillation.teacher_always_on': True,
        'actor_rollout_ref.actor.self_distillation.teacher_prompt_mode': None if static else 'answer_hint',
        'actor_rollout_ref.actor.self_distillation.teacher_image_key': 'bbox_images' if static else None,
        'actor_rollout_ref.actor.self_distillation.teacher_model_source': 'legacy',
        'actor_rollout_ref.actor.self_distillation.teacher_regularization': 'ema',
        'actor_rollout_ref.actor.self_distillation.teacher_update_rate': 0.0,
        'actor_rollout_ref.actor.self_distillation.alpha': 0.5,
        'actor_rollout_ref.actor.self_distillation.distillation_topk': 100,
        'actor_rollout_ref.actor.self_distillation.max_reprompt_len': args.max_teacher_tokens,
        'actor_rollout_ref.actor.self_distillation.include_environment_feedback': False,
        'actor_rollout_ref.actor.self_distillation.fallback_to_policy_loss_on_missing_teacher': False,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        # The teacher is frozen. Preserve the checkpoint's BF16 precision instead
        # of inheriting the colocated actor's FP32 master-weight setting.
        'actor_rollout_ref.ref.fsdp_config.model_dtype': 'bf16',
        'actor_rollout_ref.rollout.n': args.rollouts,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': args.tensor_parallel,
        'actor_rollout_ref.rollout.gpu_memory_utilization': args.rollout_memory_utilization,
        'actor_rollout_ref.rollout.max_model_len': args.max_prompt_tokens + args.max_response_tokens,
        # The pinned server overwrites config.max_model_len from the HF config.
        # engine_kwargs is applied last when constructing the actual vLLM engine.
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len': args.max_prompt_tokens + args.max_response_tokens,
        'actor_rollout_ref.rollout.max_num_seqs': max(8, min(64, args.batch_size * args.rollouts)),
        'actor_rollout_ref.rollout.enforce_eager': args.eager_rollout,
        'actor_rollout_ref.rollout.max_num_batched_tokens': args.max_prompt_tokens + args.max_response_tokens,
        'actor_rollout_ref.rollout.response_length': args.max_response_tokens,
        'actor_rollout_ref.rollout.agent.num_workers': 4,
        'critic.model.path': model, 'reward_model.enable': False, 'reward_model.use_reward_loop': False,
        'trainer.project_name': 'VPTD-EAE', 'trainer.group_name': args.mode,
        'trainer.experiment_name': args.output_dir.name, 'trainer.logger': ['console', 'tensorboard'],
        'trainer.n_gpus_per_node': args.gpus, 'trainer.nnodes': 1,
        'trainer.total_epochs': args.epochs, 'trainer.save_freq': args.save_freq,
        'trainer.test_freq': -1, 'trainer.val_before_train': False,
        'trainer.default_local_dir': str(args.output_dir.resolve()),
        'trainer.rollout_data_dir': str((args.output_dir / 'rollouts').resolve()),
    }
    if args.steps:
        config['trainer.total_training_steps'] = args.steps
    if args.cpus:
        config['ray_kwargs.ray_init.num_cpus'] = args.cpus
    if args.max_checkpoints is not None:
        config['trainer.max_actor_ckpt_to_keep'] = args.max_checkpoints
    if args.low_memory_adam:
        config['actor_rollout_ref.actor.optim.optimizer'] = 'LowMemoryAdamW'
        config['actor_rollout_ref.actor.optim.optimizer_impl'] = 'vptd_eae.optim'
    return [f'{key}={json.dumps(value)}' for key, value in config.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['ace-opsd', 'swig-opsd', 'ace-static-opd'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--cpus', type=int, help='Ray CPU budget, bounded by the rented allocation')
    parser.add_argument('--eager-rollout', action='store_true', help='Skip rollout graph compilation for short smoke runs')
    parser.add_argument('--keep-actor-on-gpu', action='store_true', help='Trade GPU memory for lower host RAM use')
    parser.add_argument('--low-memory-adam', action='store_true', help='Use single-tensor PyTorch AdamW, preserving resumed moments')
    parser.add_argument('--rollout-memory-utilization', type=float, default=0.6)
    parser.add_argument('--tensor-parallel', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--rollouts', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--max-checkpoints', type=int, default=1, help='Retain this many new checkpoints (default: 1); reserve space for one extra save')
    parser.add_argument('--dataloader-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--max-prompt-tokens', type=int, default=8192)
    parser.add_argument('--max-response-tokens', type=int, default=1024)
    parser.add_argument('--max-teacher-tokens', type=int, default=16384)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--check-config', action='store_true', help='Compose actual upstream Hydra config without loading models')
    parser.add_argument('--resume', action='store_true', help='Resume unchanged training; may extend steps, change checkpoint storage or use low-memory AdamW')
    parser.add_argument('--allow-ambiguous-ace', action='store_true', help='Explicit legacy reproduction only, including old unaudited Parquet')
    args = parser.parse_args()
    positive = [args.gpus, args.tensor_parallel, args.batch_size, args.rollouts, args.epochs,
                args.save_freq, args.lr, args.max_prompt_tokens, args.max_response_tokens, args.max_teacher_tokens]
    if min(positive) <= 0 or args.steps is not None and args.steps <= 0 or args.cpus is not None and args.cpus <= 0:
        parser.error('Resource, step and token settings must be positive')
    if args.gpus % args.tensor_parallel or args.batch_size * args.rollouts % args.gpus:
        parser.error('GPU count must be divisible by tensor parallel size and divide batch_size * rollouts')
    if not 0 < args.rollout_memory_utilization < 1:
        parser.error('Rollout memory utilization must be between zero and one')
    if args.dataloader_workers < 0 or args.max_checkpoints is not None and args.max_checkpoints <= 0:
        parser.error('Dataloader workers must be nonnegative; checkpoint retention must be positive')
    lock = json.loads((ROOT / 'configs/framework.lock.json').read_text())
    framework = ROOT / 'external/Vision-OPD'
    verify_checkout(framework, lock)
    manifest = json.loads(args.data.with_suffix('.manifest.json').read_text())
    if args.mode.startswith('ace-') and 'input_audit' not in manifest and not args.allow_ambiguous_ace:
        raise ValueError('ACE Parquet lacks input audit; re-export with the current exporter before training')
    if 'input_audit' in manifest:
        require_unambiguous_inputs(manifest['input_audit'], allow_ambiguous_ace=args.allow_ambiguous_ace)
    if manifest['mode'] != args.mode or manifest['parquet_sha256'] != hashlib.sha256(args.data.read_bytes()).hexdigest():
        raise ValueError('Parquet mode or checksum differs from export manifest')
    import pyarrow.parquet as pq
    rows = pq.read_table(args.data).to_pylist()
    if len(rows) < args.batch_size:
        raise ValueError('Training data has fewer rows than batch size')
    for row in rows:
        if row['extra_info']['split'] != 'train':
            raise ValueError('Found non-training row')
        for item in row['images'] + row.get('bbox_images', []):
            from PIL import Image
            with Image.open(item['path']) as image:
                image.verify()
    for path, expected in manifest['image_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Training image changed after export: {path}')
    overrides = build_overrides(args)
    command = [sys.executable, '-m', 'verl.trainer.main_ppo', '--config-name',
               'vopd' if args.mode == 'ace-static-opd' else 'opsd', *overrides]
    if args.check_config:
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(framework / 'verl/trainer/config'), version_base=None):
            config = compose(config_name='vopd' if args.mode == 'ace-static-opd' else 'opsd', overrides=overrides)
        print(f'Upstream Hydra configuration passed: {config.actor_rollout_ref.model.path}')
        return
    if args.dry_run:
        print(f'cd {shlex.quote(str(framework))}\n{shlex.join(command)}')
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise RuntimeError(f'Training needs {args.gpus} CUDA GPU(s); available={torch.cuda.device_count()}')
    for package, expected in lock['runtime_versions'].items():
        actual = importlib.metadata.version(package).split('+')[0]
        if actual != expected:
            raise RuntimeError(f'{package}={actual}; pinned training runtime requires {expected}. See docs/RUN_EXPERIMENTS.md')
    launch = {'framework': lock, 'data': manifest, 'command': command}
    launch_path = args.output_dir / 'launch.json'
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        history_path = args.output_dir / 'resume_history.jsonl'
        if history_path.exists():
            previous = json.loads(history_path.read_text().splitlines()[-1])['launch']
        if not args.resume or not resume_compatible(previous, launch):
            raise ValueError('Resume must preserve training settings; only step extension, checkpoint storage and low-memory AdamW may change')
        tracker = args.output_dir / 'latest_checkpointed_iteration.txt'
        completed_step = int(tracker.read_text()) if tracker.exists() else 0
        if args.steps is not None and completed_step >= args.steps:
            print(f'Checkpoint already reached step {completed_step}; no further training requested')
            return
        with history_path.open('a') as handle:
            handle.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(),
                                     'checkpoint_step': completed_step, 'launch': launch}) + '\n')
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Nonempty output directory without launch.json; select a new directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not launch_path.exists():
        launch_path.write_text(json.dumps(launch, indent=2) + '\n')
    with (args.output_dir / 'input_policy_history.jsonl').open('a') as stream:
        stream.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(),
                                 'ambiguity_override': args.allow_ambiguous_ace,
                                 'parquet_sha256': manifest['parquet_sha256'],
                                 'input_audit_present': 'input_audit' in manifest})+'\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT), str(framework))), PYTHONUNBUFFERED='1', VLLM_USE_V1='1')
    subprocess.run(command, cwd=framework, env=env, check=True)


if __name__ == '__main__':
    main()
