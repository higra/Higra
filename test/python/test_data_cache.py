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


class MyException(Exception):
    def __init__(self, message=""):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


crash_provider1 = False


@hg.data_provider("attr1")
def provider1(obj, a=None):
    global crash_provider1
    if crash_provider1:
        raise Exception("Should not have been called")
    return 1


@hg.argument_helper("attr1")
def accept_everything(dummy, attr1):
    return attr1


@hg.argument_helper(hg.CptGridGraph)
def accept_RegularGraph2d(graph, shape):
    if type(graph) != hg.RegularGraph2d:
        raise MyException("kaput")
    return 2


class TestDataCache(unittest.TestCase):

    def test_provider_caching_and_force_recompute(self):
        global crash_provider1
        obj1 = Dummy(1)
        crash_provider1 = False
        self.assertTrue(provider1(obj1, 1) == 1)
        crash_provider1 = True
        self.assertTrue(provider1(obj1, 1) == 1)
        self.assertRaises(Exception, provider1, obj1, 2)
        self.assertRaises(Exception, provider1, obj1, force_recompute=True)
        hg.clear_all_attributes()

    def test_provider_caching_global_setting(self):
        global crash_provider1
        obj1 = Dummy(1)
        crash_provider1 = False
        hg.set_provider_caching(False)
        self.assertTrue(provider1(obj1, 1) == 1)
        crash_provider1 = True
        self.assertRaises(Exception, provider1, obj1, 1)
        hg.set_provider_caching(True)
        crash_provider1 = False
        self.assertTrue(provider1(obj1, 2) == 1)
        crash_provider1 = True
        self.assertTrue(provider1(obj1, 2) == 1)
        hg.set_provider_caching(False)
        self.assertRaises(Exception, provider1, obj1, 1)
        hg.set_provider_caching(True)
        hg.clear_all_attributes()

    def test_provider_no_cache_implies_force_recompute(self):
        global crash_provider1
        obj1 = Dummy(1)
        crash_provider1 = False
        self.assertTrue(provider1(obj1, 3) == 1)
        crash_provider1 = True
        self.assertTrue(provider1(obj1, 3) == 1)
        self.assertRaises(Exception, provider1, obj1, 3, no_cache=True)
        hg.clear_all_attributes()

    def test_provider_no_cache_doesnt_store_result(self):
        global crash_provider1
        obj1 = Dummy(1)
        crash_provider1 = False
        self.assertTrue(provider1(obj1, 2, no_cache=True) == 1)
        crash_provider1 = True
        self.assertRaises(Exception, provider1, obj1, 2)
        hg.clear_all_attributes()

    def test_provider_rename_attribute(self):
        global crash_provider1
        obj1 = Dummy(1)
        crash_provider1 = False
        self.assertTrue(provider1(obj1, "aa", attribute_name="xxx") == 1)
        crash_provider1 = True
        self.assertRaises(Exception, provider1, obj1, "aa")
        self.assertTrue(provider1(obj1, "aa", attribute_name="xxx") == 1)
        hg.clear_all_attributes()

    def test_argument_helper_accept_everything(self):
        self.assertTrue(accept_everything(2) == 1)
        self.assertTrue(accept_everything((3, 2)) == 1)
        self.assertTrue(accept_everything({3, 2}) == 1)
        self.assertTrue(accept_everything({"toto": 42}) == 1)

    def test_argument_helper_accept_RegularGraph2d(self):
        g = hg.get_4_adjacency_implicit_graph((2, 3))
        self.assertTrue(accept_RegularGraph2d(g) == 2)

        self.assertRaises(MyException, accept_RegularGraph2d, (4, 5), (2, 3))


if __name__ == '__main__':
    unittest.main()
