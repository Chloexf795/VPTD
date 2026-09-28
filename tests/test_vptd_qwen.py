"""Exercise the actual OpenAI client at its HTTP boundary; no GPU or network needed."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from vptd_eae.datasets import normalize_ace
from vptd_eae.qwen import QwenAPI
from vptd_eae.benchmark import prediction_identity, score_predictions

ROOT = Path(__file__).resolve().parents[1]


class QwenIntegrationTests(unittest.TestCase):
    def test_real_client_and_cli_resume_then_score(self):
        spec = importlib.util.spec_from_file_location('benchmark_cli', ROOT / 'scripts/run_qwen_benchmark.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(payload['model'], 'qwen-test')
            self.assertNotIn('ground_truth', payload)
            return httpx.Response(200, json={
                'id': 'test', 'object': 'chat.completion', 'created': 1, 'model': 'qwen-test',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': '[{"role":"Attacker","entity":"Rebels"}]'}}],
            })

        def api(**kwargs):
            backend = QwenAPI(**kwargs)
            backend.client = OpenAI(api_key='test', base_url='http://test.invalid/v1',
                                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
            return backend

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            raw = [{'instruction': '**Argument Roles**: Attacker, Target\n**Event Mentions**: [{"entity":"Rebels"}]\n'
                    f'**Event Type**: Conflict||Attack\n**Input Sentence**: Rebels attacked base {i}.',
                    'output': '[{"role":"Attacker","entity":"Rebels"}]'} for i in range(4)]
            rows = normalize_ace(raw, 'dev')
            data = folder / 'dev.jsonl'
            data.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
            output = folder / 'run'
            base = ['run_qwen_benchmark.py', '--data', str(data), '--output-dir', str(output),
                    '--backend', 'api', '--base-url', 'http://test.invalid/v1', '--model', 'qwen-test', '--workers', '2',
                    '--allow-ambiguous-ace']
            with patch.object(cli, 'QwenAPI', api), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', base + ['--limit', '2']):
                    cli.main()
                self.assertFalse((output / 'metrics.json').exists())
                with patch.object(sys, 'argv', base + ['--resume']):
                    cli.main()
                metrics = json.loads((output / 'metrics.json').read_text())
                self.assertEqual(metrics['samples'], 4)
                self.assertEqual(metrics['metrics']['argument_classification']['f1'], 1.0)
                self.assertEqual(len(requests), 4)
                with patch.object(sys, 'argv', base + ['--resume']):
                    cli.main()
                self.assertEqual(len(requests), 4)
                with patch.object(sys, 'argv', base + ['--resume', '--max-tokens', '2048']):
                    with self.assertRaisesRegex(ValueError, 'identical settings'):
                        cli.main()
                for option in [['--output-policy','schema-v1'],['--repetition-penalty','1.05']]:
                    with patch.object(sys,'argv',base+['--resume',*option]):
                        with self.assertRaisesRegex(ValueError,'identical settings'):
                            cli.main()
                self.assertEqual(len(requests),4)
                # A completed historical run can still be reproduced explicitly;
                # it must not regenerate predictions or change historical settings.
                historical=json.loads((output/'run.json').read_text())
                for key in ('allow_ambiguous_ace','output_policy','repetition_penalty'):
                    historical['config'].pop(key)
                historical.pop('input_audit')
                (output/'run.json').write_text(json.dumps(historical))
                with patch.object(sys,'argv',base+['--resume']):
                    cli.main()
                self.assertEqual(len(requests),4)
                self.assertTrue((output/'resume_input_audit.json').exists())

    def test_truncated_response_is_retained_and_scored_as_invalid(self):
        def respond(request):
            return httpx.Response(200, json={'id': 't', 'object': 'chat.completion', 'created': 1, 'model': 'test',
                 'choices': [{'index': 0, 'finish_reason': 'length',
                              'message': {'role': 'assistant', 'content': '[{"role":"Attacker","entity":"Rebels"}]'}}]})
        backend = QwenAPI('test', 'http://test.invalid/v1', max_tokens=1, timeout=2)
        backend.client = OpenAI(api_key='test', base_url='http://test.invalid/v1',
                                http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        row = normalize_ace([{'instruction': '**Event Type**: Conflict||Attack\n**Input Sentence**: Rebels attacked.',
                              'output': '[{"role":"Attacker","entity":"Rebels"}]'}], 'dev')[0]
        generated = backend.generate(row)
        self.assertEqual(generated['finish_reason'], 'length')
        report = score_predictions([row], [{'sample_id': row['sample_id'], 'status': 'ok',
                                   'input_sha256': prediction_identity(row), **generated}])
        self.assertEqual(report['samples'], 1)
        self.assertEqual(report['truncated_outputs'], 1)
        self.assertEqual(report['invalid_outputs'], 1)
        self.assertEqual(report['metrics']['argument_classification']['gold'], 1)
        self.assertEqual(report['metrics']['argument_classification']['tp'], 0)


if __name__ == '__main__':
    unittest.main()
