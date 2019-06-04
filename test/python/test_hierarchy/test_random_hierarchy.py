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
import numpy as np


class TestRandomHierarchy(unittest.TestCase):

    def test_random_binary_partition_tree_perfectly_balanced(self):
        size = 32
        tree, altitudes = hg.random_binary_partition_tree(size, 0)
        depth = hg.attribute_depth(tree)
        for i in range(6):
            num_nodes = 2**i
            self.assertTrue(np.sum(depth == i) == num_nodes)

    def test_random_binary_partition_tree_perfectly_unbalanced(self):
        size = 32
        tree, altitudes = hg.random_binary_partition_tree(size, 1)
        depth = hg.attribute_depth(tree)
        for i in range(32):
            num_nodes = 1 if i == 0 else 2
            self.assertTrue(np.sum(depth == i) == num_nodes)
