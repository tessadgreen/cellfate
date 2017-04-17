from unittest import TestCase

import cellfate
from cellfate import io
from cellfate import model 

class Test(TestCase):
    def test_is_string(self):
        s = cellfate.test_print()
        self.assertTrue(isinstance(s, str))

    def test_io(self):
    	x = io.read('test-data','R','G',5,10)
    	assert isinstance(x.cellwidth,int)
    	assert x.bin_num==10

    def test(self):
    	assert True
