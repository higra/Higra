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


class Dummy(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)


class CptA(hg.Concept):
    _name = "CA"
    _description = "Concept A."
    _data_elements = {"caA": ("canonical element", None),
                      "ea1": ("element ca 1", "ea1")}
    _canonical_data_element = "caA"

    def __init__(self, **kwargs):
        super(CptA, self).__init__(**kwargs)

    @staticmethod
    def link(caA, ea1):
        hg.add_tag(caA, CptA)
        hg.set_attribute(caA, "ea1", ea1)


class CptB(CptA):
    _name = "CB"
    _description = "Concept B."
    _data_elements = {"caB": ("canonical element", None),
                      "caA": ("element cb 1", "caA"),
                      "eb2": ("element cb 2", "eb2")}
    _canonical_data_element = "caB"

    def __init__(self, **kwargs):
        super(CptB, self).__init__(**kwargs)

    @staticmethod
    def link(caB, caA, eb2):
        hg.add_tag(caB, CptB)
        hg.set_attribute(caB, "caA", caA)
        hg.set_attribute(caB, "eb2", eb2)


class CptC(hg.Concept):
    _name = "CC"
    _description = "Concept C."
    _data_elements = {"caC": ("canonical element", None),
                      "ec1": ("element c 1", "ec1")}
    _canonical_data_element = "caC"

    def __init__(self, **kwargs):
        super(CptA, self).__init__(**kwargs)

    @staticmethod
    def link(caC, ec1):
        hg.add_tag(caC, CptC)
        hg.set_attribute(caC, "ec1", ec1)


# simple implicit arg name
@hg.argument_helper(CptA)
def concept_consumer1(obj, ea1):
    return ea1


#  renamed argument
@hg.argument_helper(CptA(ea1="new_name"))
def concept_consumer2(obj, new_name):
    return new_name


# missing concepts elements in arguments
@hg.argument_helper(CptB)
def concept_consumer3(obj, ea1):
    return ea1

# chained concepts
@hg.argument_helper(CptA, ("ea1", CptC))
def concept_consumer4(obj, ec1):
    return ec1


class TestConcept(unittest.TestCase):

    def test_construct(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)

        CptA.link(o1, o2)
        CptB.link(o3, o1, o4)

        ca = CptA.construct(o1)
        cb = CptB.construct(o3)

        ref_ca = {"caA": o1, "ea1": o2}
        ref_cb = {"caB": o3, "eb2": o4, "caA": o1, "ea1": o2}

        self.assertTrue(ca == ref_ca)
        self.assertTrue(cb == ref_cb)

        hg.clear_all_attributes()

    def test_concept_consumer1(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        CptA.link(o1, o2)

        # call from well defined concept
        self.assertTrue(concept_consumer1(o1) == o2)

        # call from well defined concept
        self.assertTrue(concept_consumer1(obj=o1) == o2)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer1(o1, o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer1(o1, ea1=o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer1(obj=o1, ea1=o3) == o3)

        # call from free variable
        self.assertTrue(concept_consumer1(o3, o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer1(o3, ea1=o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer1(obj=o3, ea1=o4) == o4)

    def test_concept_consumer2(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        CptA.link(o1, o2)

        # call from well defined concept
        self.assertTrue(concept_consumer2(o1) == o2)

        # call from well defined concept
        self.assertTrue(concept_consumer2(obj=o1) == o2)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer2(o1, o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer2(o1, new_name=o3) == o3)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer2(obj=o1, new_name=o3) == o3)

        # call from free variable
        self.assertTrue(concept_consumer2(o3, o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer2(o3, new_name=o4) == o4)

        # call from free variable
        self.assertTrue(concept_consumer2(obj=o3, new_name=o4) == o4)

    def test_concept_consumer3(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        o5 = Dummy(5)

        CptA.link(o1, o2)
        CptB.link(o3, o1, o4)

        # call from well defined concept
        self.assertTrue(concept_consumer3(o3) == o2)

        # call from well defined concept
        self.assertTrue(concept_consumer3(obj=o3) == o2)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer3(o3, o5) == o5)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer3(o3, ea1=o5) == o5)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer3(obj=o3, ea1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer3(o4, o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer3(o4, ea1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer3(obj=o4, ea1=o5) == o5)

    def test_concept_consumer4(self):
        o1 = Dummy(1)
        o2 = Dummy(2)
        o3 = Dummy(3)
        o4 = Dummy(4)
        o5 = Dummy(5)

        CptA.link(o1, o2)
        CptC.link(o2, o3)

        # call from well defined concept
        self.assertTrue(concept_consumer4(o1) == o3)

        # call from well defined concept
        self.assertTrue(concept_consumer4(obj=o1) == o3)

        # call from well defined concept with argument replacement
        self.assertTrue(concept_consumer4(o1, o4) == o4)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer4(o1, ec1=o4) == o4)

        # call from well defined concept with named argument replacement
        self.assertTrue(concept_consumer4(obj=o1, ec1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer4(o4, o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer4(o4, ec1=o5) == o5)

        # call from free variable
        self.assertTrue(concept_consumer4(obj=o4, ec1=o5) == o5)


if __name__ == '__main__':
    unittest.main()
