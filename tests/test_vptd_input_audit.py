import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

from vptd_eae.datasets import audit_inputs, require_unambiguous_inputs, normalize_ace


class InputAuditTests(unittest.TestCase):
    def row(self, sid='a', entity='Alice'):
        return {'sample_id': sid, 'dataset': 'ace', 'protocol': 'surface',
                'prompt': 'Alice attacked Bob.', 'images': [], 'trigger': 'attacked',
                'arguments': [{'entity': entity, 'role': 'Attacker'}]}

    def test_same_visible_input_with_different_gold_is_blocked(self):
        rows = [self.row(), self.row('b', 'Bob')]
        report = audit_inputs(rows)
        self.assertEqual(report['conflicting_groups'], 1)
        self.assertEqual(report['conflicting_rows'], 2)
        with self.assertRaisesRegex(ValueError, 'Ambiguous inputs'):
            require_unambiguous_inputs(report)
        self.assertEqual(rows[0]['arguments'][0]['entity'], 'Alice')

    def test_gold_order_and_duplicate_pairs_do_not_create_false_conflict(self):
        first = self.row()
        first['arguments'].append({'entity': 'Bob', 'role': 'Target'})
        second = copy.deepcopy(first)
        second['sample_id'] = 'b'
        second['arguments'] = list(reversed(first['arguments'])) + first['arguments']
        self.assertEqual(audit_inputs([first, second])['conflicting_groups'], 0)

    def test_images_and_visible_trigger_distinguish_inputs_but_hidden_ids_do_not(self):
        a, b = self.row(), self.row('b', 'Bob')
        b['prompt'] += ' Target trigger is the second attacked.'
        self.assertEqual(audit_inputs([a,b])['conflicting_groups'], 0)
        b['prompt'] = a['prompt']
        b['event_id'] = 'different-hidden-id'
        self.assertEqual(audit_inputs([a,b])['conflicting_groups'], 1)
        b['images'] = ['other-image.jpg']
        self.assertEqual(audit_inputs([a,b])['conflicting_groups'], 0)

    def test_missing_trigger_is_not_silently_allowed(self):
        a = self.row()
        a['trigger'] = ''
        report = audit_inputs([a])
        self.assertEqual(report['missing_ace_trigger_rows'], 1)
        with self.assertRaises(ValueError):
            require_unambiguous_inputs(report)
        require_unambiguous_inputs(report, allow_ambiguous_ace=True)

    def test_repeated_trigger_requires_location_even_when_labels_happen_to_agree(self):
        row=self.row()
        row['sentence']='Alice attacked Bob and attacked Carol.'
        report=audit_inputs([row])
        self.assertEqual(report['unlocated_ace_trigger_rows'],1)
        self.assertFalse(report['ready'])

    def test_ace_override_cannot_allow_conflicting_swig_labels(self):
        a,b=self.row(),self.row('b','Bob')
        a['dataset']=b['dataset']='swig'
        with self.assertRaises(ValueError):
            require_unambiguous_inputs(audit_inputs([a,b]),allow_ambiguous_ace=True)

    def test_processed_trigger_positions_are_visible_and_provenance_retained(self):
        raw = [{'doc_id':'d', 'sent_id':'s', 'sentence':'They attacked and attacked.',
                'entity_mentions':[{'text':'They','entity_type':'PER'}],
                'event_mentions':[
                    {'id':'e1','event_type':'Conflict:Attack','trigger':{'text':'attacked','start':1,'end':2},
                     'arguments':[{'text':'They','role':'Attacker','start':0,'end':1}]},
                    {'id':'e2','event_type':'Conflict:Attack','trigger':{'text':'attacked','start':3,'end':4},
                     'arguments':[]}]}]
        rows = normalize_ace(raw, 'dev')
        self.assertNotEqual(rows[0]['prompt'], rows[1]['prompt'])
        self.assertEqual(rows[0]['source_annotations']['trigger']['start'], 1)
        self.assertEqual(rows[0]['source_annotations']['event_id'], 'e1')
        self.assertEqual(rows[0]['source_annotations']['arguments'][0]['start'], 0)
        changed = copy.deepcopy(raw)
        changed[0]['event_mentions'][0]['arguments'][0]['text'] = 'SECRET_GOLD'
        self.assertEqual(normalize_ace(changed, 'dev')[0]['prompt'], rows[0]['prompt'])
        self.assertEqual(audit_inputs(rows)['conflicting_groups'], 0)
        self.assertEqual(rows[0]['protocol'], 'ace-processed-surface-eae-given-trigger-v2')

    def test_inference_and_export_block_ambiguous_data_before_backend_or_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            a, b = self.row(), self.row('b', 'Bob')
            for row in (a,b):
                row.update(split='train', event_type='CONFLICT||ATTACK', visual_prototypes=[])
            data = folder/'train.jsonl'
            data.write_text('\n'.join(json.dumps(r) for r in (a,b)))
            for script,args in [('run_qwen_benchmark.py',['--output-dir',str(folder/'run'),'--dry-run']),
                                ('export_opd_data.py',['--mode','ace-opsd','--output',str(folder/'out.parquet')])]:
                spec = importlib.util.spec_from_file_location('guard_cli', root/'scripts'/script)
                cli = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cli)
                with patch.object(sys,'argv',[script,'--data',str(data),*args]):
                    with self.assertRaisesRegex(ValueError,'Ambiguous inputs'):
                        cli.main()
            self.assertFalse((folder/'out.parquet').exists())
            self.assertFalse((folder/'run').exists())


if __name__ == '__main__':
    unittest.main()
