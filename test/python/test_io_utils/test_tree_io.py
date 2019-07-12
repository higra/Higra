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
import numpy as np
import higra as hg

import os
import os.path


def silent_remove(filename):
    try:
        os.remove(filename)
    except:
        pass


class TestTreeIO(unittest.TestCase):

    def test_treeReadWrite(self):
        filename = "testTreeIO.graph"
        silent_remove(filename)

        parents = np.asarray((5, 5, 6, 6, 6, 7, 7, 7), dtype=np.uint64)
        tree = hg.Tree(parents)

        attr1 = np.asarray((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        attr2 = np.asarray((8, 7, 6, 5, 4, 3, 2, 1), dtype=np.int32)

        hg.save_tree(filename, tree, {"attr1": attr1, "attr2": attr2})

        tree, attributes = hg.read_tree(filename)
        silent_remove(filename)

        self.assertTrue(np.allclose(tree.parents(), parents))

        self.assertTrue("attr1" in attributes)
        self.assertTrue(np.allclose(attr1, attributes["attr1"]))

        self.assertTrue("attr2" in attributes)
        self.assertTrue(np.allclose(attr2, attributes["attr2"]))

    def test_print_partition_tree(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))
        s = hg.print_partition_tree(tree, altitudes=np.asarray([0, 0, 0, 0, 0, 100, 1100, 20000]),
                                    scale="log",
                                    return_string=True)
        ref = ("0------\n"
               "      |\n"
               "    100------------------------------------------------------------------------------------------------\n"
               "      |                                                                                               |\n"
               "1------                                                                                               |\n"
               "                                                                                                      |\n"
               "                                                                                                   2e+04\n"
               "2-------------                                                                                        |\n"
               "             |                                                                                        |\n"
               "             |                                                                                        |\n"
               "3---------1.1e+03--------------------------------------------------------------------------------------\n"
               "             |\n"
               "             |\n"
               "             |\n"
               "4-------------")

        self.assertTrue(ref == s)

    # this is really just a non throw test...
    def test_print_partition_tree(self):
        tree = hg.Tree((5, 5, 6, 6, 6, 7, 7, 7))

        for o in ['altitudes', 'area', 'none', 'leaves']:
            for s in ["linear", "log"]:
                hg.print_partition_tree(tree,
                                        altitudes=np.asarray([0, 0, 0, 0, 0, 100, 1100, 20000]),
                                        attribute=np.asarray([0, 0, 0, 0, 0, 100, 1100, 20000]),
                                        ordering=o,
                                        scale=s,
                                        return_string=True)


if __name__ == '__main__':
    unittest.main()
