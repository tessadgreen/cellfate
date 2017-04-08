from unittest import TestCase

import cellfate

class Test(TestCase):
    def test_is_string(self):
        s = cellfate.test.test_print()
        self.assertTrue(isinstance(s, basestring))
