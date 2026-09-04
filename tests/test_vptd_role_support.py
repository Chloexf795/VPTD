import unittest

from vptd_eae.role_support import NONE_ROLE, ROLE_TO_INDEX, build_role_mask, roles_for_event


class RoleSupportTests(unittest.TestCase):
    def test_attack_support_uses_local_schema_and_none(self):
        roles = roles_for_event("Conflict:Attack")
        self.assertIn("Attacker", roles)
        self.assertIn("Target", roles)
        self.assertIn(NONE_ROLE, roles)
        mask = build_role_mask(["Conflict:Attack"])
        self.assertTrue(mask[0, ROLE_TO_INDEX["Attacker"]])
        self.assertTrue(mask[0, ROLE_TO_INDEX[NONE_ROLE]])
        self.assertFalse(mask[0, ROLE_TO_INDEX["Giver"]])


if __name__ == "__main__":
    unittest.main()
