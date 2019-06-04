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


def dendrogram_purity_naif(tree, leaf_labels):
    from itertools import combinations

    lcaf = hg.make_lca_fast(tree)
    area = hg.attribute_area(tree)
    max_label = np.max(leaf_labels)
    label_histo = np.zeros((tree.num_leaves(), max_label + 1), dtype=np.int64)
    label_histo[np.arange(tree.num_leaves()), leaf_labels] = 1
    label_histo = hg.accumulate_sequential(tree, label_histo, hg.Accumulators.sum)
    class_purity = label_histo / area[:, None]

    count = 0
    total = 0
    for label in set(leaf_labels):
        same = leaf_labels == label
        same_indices, = same.nonzero()

        if len(same_indices) < 2:
            continue

        pairs = list(combinations(same_indices, 2))
        count += len(pairs)

        pairs = np.asarray(pairs, dtype=np.int64)
        lcas = lcaf.lca(pairs[:, 0], pairs[:, 1])
        total += np.sum(class_purity[lcas, label])

    return total / count


class TestDendrogramPurity(unittest.TestCase):

    def test_dendrogram_purity(self):
        tree = hg.Tree((5, 5, 6, 7, 7, 6, 8, 8, 8))
        labels = np.asarray((1, 1, 0, 1, 0), dtype=np.int32)
        p = hg.dendrogram_purity(tree, labels)
        self.assertTrue(p == 0.65)

        tree = hg.Tree((5, 5, 5, 6, 6, 7, 7, 7))
        labels = np.asarray((1, 1, 0, 1, 0), dtype=np.int32)
        p = hg.dendrogram_purity(tree, labels)
        self.assertTrue(np.allclose(p, 0.5666666666666667))

    def test_dendrogram_purity_random(self):
        g = hg.get_4_adjacency_graph((10, 10))
        np.random.seed(42)
        for i in range(100):
            ew = np.random.randint(0, 20, g.num_edges())
            tree, _ = hg.quasi_flat_zones_hierarchy(g, ew)
            labels = np.random.randint(0, 10, (100,))
            v1 = hg.dendrogram_purity(tree, labels)
            v2 = dendrogram_purity_naif(tree, labels)
            self.assertTrue(np.allclose(v1, v2))
