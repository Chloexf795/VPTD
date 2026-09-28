import json
import tempfile
import unittest
from pathlib import Path

from vptd_eae.datasets import normalize_ace, normalize_swig, audit_splits
from vptd_eae.benchmark import parse_answer, score_predictions, prediction_identity, match_boxes
from vptd_eae.opd import build_opd_records


def ace_row(entity='Rebels'):
    return {
        'instruction': '**Event Type**: Conflict||Attack\n**Argument Roles**: Attacker, Target\n'
                       '**Input Sentence**: Rebels attacked the base.',
        'input': '', 'output': json.dumps([{'role': 'Attacker', 'entity': entity}]),
    }


class IntegrationTests(unittest.TestCase):
    def test_ace_gold_is_not_in_prompt_and_identity_ignores_gold(self):
        first = normalize_ace([ace_row()], 'dev')[0]
        changed = normalize_ace([ace_row('SECRET_GOLD')], 'dev')[0]
        self.assertEqual(first['prompt'], changed['prompt'])
        self.assertNotIn('SECRET_GOLD', changed['prompt'])
        self.assertEqual(first['sample_id'], changed['sample_id'])
        self.assertNotEqual(prediction_identity(first), prediction_identity(changed))

    def test_swig_formats_share_canonical_grounding_task(self):
        common = {'instruction': 'Event Type:\nContact||Meet\n', 'images': ['/old/a.jpg']}
        a = {**common, 'output': '{"objects":[{"category":"Participant","bbox":[0,0,100,100]}]}'}
        b = {**common, 'output': '[{"role":"Entity","bboxes":[[0,0,100,100]]}]'}
        x = normalize_swig([a], 'train', Path('/new'))[0]
        y = normalize_swig([b], 'dev', Path('/new'))[0]
        self.assertEqual(x['arguments'], y['arguments'])
        self.assertEqual(x['images'], ['/new/a.jpg'])
        self.assertNotIn('[0, 0, 100, 100]', x['prompt'])

    def test_duplicate_cross_split_prompts_are_reported(self):
        report = audit_splits({'train': normalize_ace([ace_row()], 'train'),
                               'dev': normalize_ace([ace_row()], 'dev')})
        self.assertEqual(report['overlap_pairs']['train:dev'], 1)

    def test_parse_errors_are_distinct_from_empty_prediction(self):
        self.assertEqual(parse_answer('```json\n[]\n```'), [])
        with self.assertRaises(ValueError):
            parse_answer('I cannot answer')
        with self.assertRaises(ValueError):
            parse_answer('{"error":"failed"}')

    def test_scoring_requires_exact_sample_set_and_identity(self):
        records = normalize_ace([ace_row()], 'dev')
        row = records[0]
        pred = {'sample_id': row['sample_id'], 'input_sha256': prediction_identity(row),
                'output': '[{"role":"Attacker","entity":"Rebels"}]', 'status': 'ok'}
        self.assertEqual(score_predictions(records, [pred])['metrics']['argument_classification']['f1'], 1)
        for bad in [[], [pred, pred], [{**pred, 'input_sha256': 'changed'}], [{**pred, 'status': 'error'}]]:
            with self.assertRaises(ValueError):
                score_predictions(records, bad)

    def test_invalid_output_is_counted_as_a_miss(self):
        rows = normalize_ace([ace_row()], 'dev')
        pred = {'sample_id': rows[0]['sample_id'], 'input_sha256': prediction_identity(rows[0]),
                'status': 'ok', 'output': 'not JSON'}
        report = score_predictions(rows, [pred])
        self.assertEqual(report['invalid_outputs'], 1)
        self.assertEqual(report['metrics']['argument_classification']['recall'], 0)

    def test_static_opd_text_student_and_image_teacher(self):
        rows = normalize_ace([ace_row()], 'train')
        with tempfile.TemporaryDirectory() as folder:
            from PIL import Image
            image = Path(folder) / 'p.png'
            Image.new('RGB', (8, 8)).save(image)
            rows[0]['visual_prototypes'] = [{'image': str(image)}]
            converted = build_opd_records(rows, 'ace-static-opd')
        self.assertEqual(converted[0]['images'], [])
        self.assertNotIn('<image>', converted[0]['prompt'][0]['content'])
        self.assertIn('<image>', converted[0]['teacher_prompt'][0]['content'])
        self.assertNotIn('"Attacker"', converted[0]['teacher_prompt'][0]['content'])

    def test_training_refuses_dev_and_missing_images(self):
        with self.assertRaises(ValueError):
            build_opd_records(normalize_ace([ace_row()], 'dev'), 'ace-opsd')
        rows = normalize_ace([ace_row()], 'train')
        rows[0]['visual_prototypes'] = [{'image': '/missing/a.jpg'}]
        with self.assertRaises(FileNotFoundError):
            build_opd_records(rows, 'ace-static-opd')

    def test_native_ace_string_trigger_is_preserved(self):
        rows = [{'sentence': 'Rebels attacked.', 'event_mentions': [
            {'event_type': 'Conflict:Attack', 'trigger': 'attacked', 'arguments': [
                {'role': 'Attacker', 'text': 'Rebels'}]}]}]
        converted = normalize_ace(rows, 'train')
        self.assertEqual(converted[0]['trigger'], 'attacked')

    def test_ace_empty_role_schema_is_repaired_without_using_gold(self):
        raw = {'instruction': '**Event Type**: Transaction||Transfer|Money\n**Argument Roles**: \n'
                              '**Input Sentence**: Alice paid Bob.', 'output': '[]'}
        row = normalize_ace([raw], 'dev')[0]
        self.assertIn('**Argument Roles**: Giver, Recipient, Money, Beneficiary, Place', row['prompt'])
        self.assertEqual(row['arguments'], [])

    def test_ace_meet_participant_alias_matches_existing_evaluator(self):
        raw = {'instruction': '**Event Type**: Contact||Meet\n**Argument Roles**: Participant, Place\n'
                              '**Input Sentence**: Alice met Bob.',
               'output': '[{"role":"Entity","entity":"Alice"}]'}
        rows = normalize_ace([raw], 'dev')
        self.assertIn('**Argument Roles**: Entity, Place', rows[0]['prompt'])
        pred = {'sample_id': rows[0]['sample_id'], 'input_sha256': prediction_identity(rows[0]),
                'status': 'ok', 'output': '[{"role":"Participant","entity":"Alice"}]'}
        self.assertEqual(score_predictions(rows, [pred])['metrics']['argument_classification']['f1'], 1)

    def test_grounding_credit_is_one_to_one_and_role_sensitive(self):
        gold = [{'role': 'Attacker', 'bbox': [0, 0, 100, 100]}]
        predicted = [{'role': 'Target', 'bbox': [0, 0, 100, 100]},
                     {'role': 'Target', 'bbox': [1, 1, 99, 99]}]
        self.assertEqual(match_boxes(gold, predicted, False), (1, 2, 1))
        self.assertEqual(match_boxes(gold, predicted, True), (0, 2, 1))


if __name__ == '__main__':
    unittest.main()
