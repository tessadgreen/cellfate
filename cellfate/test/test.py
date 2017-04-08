from unittest import TestCase

import cell-fate-decision

class Test(TestCase):
    def test_is_string(self):
        s = cell-fate-decision.test.test_print()
        self.assertTrue(isinstance(s, basestring))
