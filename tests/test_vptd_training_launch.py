import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('train_opd', SCRIPTS / 'train_opd.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
sys.path.pop(0)


class TrainingLaunchTests(unittest.TestCase):
    def test_old_unaudited_ace_parquet_is_blocked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.parquet'
            path.with_suffix('.manifest.json').write_text(json.dumps({'mode':'ace-opsd'}))
            argv=['train_opd.py','--mode','ace-opsd','--data',str(path),'--output-dir',str(Path(directory)/'run')]
            with patch.object(sys,'argv',argv),patch.object(training,'verify_checkout'):
                with self.assertRaisesRegex(ValueError,'lacks input audit'):
                    training.main()

    def test_resume_allows_only_extending_the_step_budget(self):
        original = {'framework': {'commit': 'locked'}, 'data': {'sha': 'same'},
                    'command': ['python', '-m', 'trainer', 'model=qwen', 'trainer.total_training_steps=2']}
        extended = copy.deepcopy(original)
        extended['command'][-1] = 'trainer.total_training_steps=3'
        self.assertTrue(training.resume_compatible(original, original))
        self.assertTrue(training.resume_compatible(original, extended))
        self.assertFalse(training.resume_compatible(extended, original))
        for field in ['framework', 'data']:
            changed = copy.deepcopy(extended)
            changed[field] = {'changed': True}
            self.assertFalse(training.resume_compatible(original, changed))
        extended['command'][3] = 'model=other'
        self.assertFalse(training.resume_compatible(original, extended))

    def test_resume_can_change_checkpoint_storage_policy(self):
        original = {'framework': {}, 'data': {}, 'command': [
            'python', 'trainer.total_training_steps=3', 'trainer.save_freq=2', 'actor.optim.lr=2e-6']}
        extended = copy.deepcopy(original)
        extended['command'][1:3] = ['trainer.total_training_steps=50', 'trainer.save_freq=25']
        extended['command'].append('trainer.max_actor_ckpt_to_keep=1')
        self.assertTrue(training.resume_compatible(original, extended))
        extended['command'][3] = 'actor.optim.lr=1e-5'
        self.assertFalse(training.resume_compatible(original, extended))

    def test_resume_allows_adam_kernel_but_not_a_different_optimizer(self):
        original = {'framework': {}, 'data': {}, 'command': ['python', 'trainer.total_training_steps=50']}
        current = copy.deepcopy(original)
        current['command'] += ['actor_rollout_ref.actor.optim.optimizer="LowMemoryAdamW"',
                               'actor_rollout_ref.actor.optim.optimizer_impl="vptd_eae.optim"']
        self.assertTrue(training.resume_compatible(original, current))
        current['command'][-2] = 'actor_rollout_ref.actor.optim.optimizer="SGD"'
        self.assertFalse(training.resume_compatible(original, current))


if __name__ == '__main__':
    unittest.main()
