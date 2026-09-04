import unittest

from vptd_eae.data import attach_event_type_prototypes


class PrototypeDataTests(unittest.TestCase):
    def test_swig_is_attached_as_prototype_not_ace_instance_label(self):
        ace = [
            {
                "doc_id": "d1",
                "sent_id": "s1",
                "sentence": "Rebels attacked the base.",
                "event_type": "Conflict||Attack",
                "trigger": "attacked",
                "arguments": [
                    {"role": "Attacker", "entity": "Rebels"},
                    {"role": "Target", "entity": "the base"},
                ],
            }
        ]
        swig = [
            {
                "image": "attack.jpg",
                "event_type": "Conflict||Attack",
                "verb": "attacking",
                "bounding_boxes": {"Attacker": [[0, 0, 100, 100]], "Target": [[200, 0, 300, 100]]},
            }
        ]
        records, stats = attach_event_type_prototypes(ace, swig, split="train")
        record = list(records)[0]
        self.assertEqual(stats.paired_events, 1)
        self.assertFalse(record["alignment"]["instance_aligned"])
        self.assertEqual(record["arguments"], ace[0]["arguments"])
        self.assertNotIn("bounding_boxes", record["arguments"][0])


if __name__ == "__main__":
    unittest.main()
