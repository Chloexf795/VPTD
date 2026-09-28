import copy
import json
import unittest

import httpx
from openai import OpenAI
from vptd_eae.qwen import QwenAPI, output_schema


class OutputConstraintTests(unittest.TestCase):
    def row(self):
        return {'dataset':'ace','event_type':'CONFLICT||ATTACK','images':[],
                'prompt':'**Argument Roles**: Attacker, Target\n**Event Mentions**: [{"entity":"Alice"},{"entity":"Bob"}]',
                'arguments':[{'role':'Attacker','entity':'SECRET_GOLD'}]}

    def test_schema_uses_only_input_candidates_and_roles(self):
        row = self.row()
        a = output_schema(row)
        changed = copy.deepcopy(row)
        changed['arguments'] = []
        self.assertEqual(a, output_schema(changed))
        self.assertNotIn('SECRET_GOLD',json.dumps(a))
        self.assertEqual(a['items']['properties']['entity']['enum'],['Alice','Bob'])
        self.assertEqual(a['items']['properties']['role']['enum'],['Attacker','Target'])
        self.assertFalse(a['items']['additionalProperties'])

    def test_swig_coordinates_have_four_bounded_values_without_gold_count_cap(self):
        row = {'dataset':'swig','prompt':'Allowed roles: Agent, Vehicle','arguments':[]}
        schema = output_schema(row)
        bbox = schema['items']['properties']['bbox']
        self.assertEqual(bbox['minItems'],4)
        self.assertEqual(bbox['maxItems'],4)
        self.assertEqual(bbox['items']['maximum'],1000)
        self.assertNotIn('maxItems',schema)

    def test_missing_candidates_cannot_be_filled_from_gold(self):
        row=self.row()
        row['prompt']='Allowed roles: Attacker, Target'
        with self.assertRaisesRegex(ValueError,'candidate'):
            output_schema(row)

    def test_actual_http_payload_and_raw_output_are_preserved(self):
        payloads=[]
        raw='[{"role":"Attacker","entity":"Alice"},{"role":"Attacker","entity":"Alice"}]'
        def respond(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200,json={'id':'t','object':'chat.completion','created':1,'model':'test',
                'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':raw}}]})
        model=QwenAPI('test','http://test.invalid/v1',max_tokens=1024,timeout=2,
                      output_policy='schema-v1',repetition_penalty=1.05)
        model.client=OpenAI(api_key='test',base_url='http://test.invalid/v1',
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        result=model.generate(self.row())
        self.assertEqual(payloads[0]['structured_outputs']['json'],output_schema(self.row()))
        self.assertEqual(payloads[0]['repetition_penalty'],1.05)
        self.assertEqual(result['output'],raw)


if __name__=='__main__':
    unittest.main()
