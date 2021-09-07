# encoding: utf-8
#
# main.py


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from pytest import mark


class Tester(object):

    def test_truth(self):
        assert 1 + 1 == 2


def test_foo():
    assert True
