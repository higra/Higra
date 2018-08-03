############################################################################
# Copyright ESIEE Paris (2018)                                             #
#                                                                          #
# Contributor(s) : Benjamin Perret                                         #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import unittest
import higra as hg


class Dummy:
    def __init__(self, val):
        self.hash = val

    def __hash__(self):
        return self.hash

    def __str__(self):
        return str(self.hash)


@hg.data_provider("attr1")
def provider1(obj, crash=False):
    if crash:
        raise Exception("Should not have been called")
    return 1


@hg.data_consumer("attr0")
def consumer0(obj, attr0):
    return attr0


@hg.data_consumer("attr1")
def consumer1(obj, attr1):
    return attr1


@hg.data_consumer(attr1="attr1")
def consumer2(obj, attr1):
    return attr1


@hg.data_consumer("attr1", attr2="dep.attr2")
def consumer3(obj, attr1, attr2):
    return attr1 + attr2


@hg.data_provider("attr2")
@hg.data_consumer("attr1")
def provider_consumer(obj, attr1, crash=False):
    if crash:
        raise Exception("Should not have been called")
    return attr1 + 1


class TestDataCache(unittest.TestCase):

    def test_provider(self):
        obj1 = Dummy(1)
        self.assertTrue(provider1(obj1, False) == 1)
        self.assertTrue(provider1(obj1, True) == 1)
        self.assertRaises(Exception, provider1, obj1, True, force_recompute=True)
        hg.clear_all_attributes()

    def test_consumer(self):
        obj3 = Dummy(3)
        self.assertRaises(Exception, consumer0, obj3)
        self.assertTrue(consumer0(obj3, 1) == 1)
        self.assertTrue(consumer0(obj3, attr0=1) == 1)
        hg.set_attribute(obj3, "attr0", 1)
        self.assertTrue(consumer0(obj3) == 1)
        hg.clear_all_attributes()

        obj4 = Dummy(4)
        self.assertTrue(consumer1(obj4) == 1)
        self.assertTrue(hg.get_attribute(obj4, "attr1") == 1)
        hg.clear_all_attributes()

        obj5 = Dummy(5)
        self.assertTrue(consumer2(obj5) == 1)
        self.assertTrue(hg.get_attribute(obj5, "attr1") == 1)
        hg.clear_all_attributes()

    def test_providerconsumer(self):
        obj2 = Dummy(2)
        self.assertTrue(provider_consumer(obj2, crash=False) == 2)
        self.assertTrue(provider_consumer(obj2, crash=True) == 2)
        self.assertRaises(Exception, provider_consumer, obj2, crash=True, force_recompute=True)
        hg.clear_all_attributes()

    def test_consumer_repath(self):
        obj6 = Dummy(6)
        self.assertTrue(consumer2(obj6, attr1="attr2") == 2)
        self.assertTrue(consumer2(obj6, "attr2") == 2)
        self.assertTrue(hg.get_attribute(obj6, "attr2") == 2)
        hg.clear_all_attributes()

    def test_consumer_compoundPath(self):
        obj7 = Dummy(7)
        obj8 = Dummy(8)
        self.assertRaises(Exception, consumer3, obj7)
        hg.set_attribute(obj7, "dep", obj8)
        self.assertTrue(consumer3(obj7) == 3)
        self.assertTrue(hg.get_attribute(obj8, "attr2") == 2)
        self.assertTrue(hg.get_attribute(obj8, "attr1") == 1)
        hg.clear_all_attributes()


if __name__ == '__main__':
    unittest.main()
