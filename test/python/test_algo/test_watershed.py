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


class TestWatershed(unittest.TestCase):

    def test_watershed(self):
        g = hg.get_4_adjacency_graph((4, 4))
        edge_weights = np.asarray((1, 2, 5, 5, 5, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 3, 5, 4, 0, 7, 0, 3, 4, 0))

        labels = hg.labelisation_watershed(g, edge_weights)
        expected = ((1, 1, 1, 2),
                    (1, 1, 2, 2),
                    (1, 1, 3, 3),
                    (1, 1, 3, 3))
        self.assertTrue(np.allclose(labels, expected))

    def test_seeded_watershed(self):
        g = hg.get_4_adjacency_graph((4, 4))
        edge_weights = np.asarray((1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0))

        seeds = np.asarray(((1, 1, 0, 0),
                            (1, 0, 0, 0),
                            (0, 0, 0, 0),
                            (1, 1, 2, 2)))

        labels = hg.labelisation_seeded_watershed(g, edge_weights, seeds)

        expected = np.asarray(((1, 1, 2, 2),
                               (1, 1, 2, 2),
                               (1, 1, 2, 2),
                               (1, 1, 2, 2)))
        self.assertTrue(np.all(labels == expected))

        seeds = np.asarray(((1, 1, 9, 9),
                            (1, 9, 9, 9),
                            (9, 9, 9, 9),
                            (2, 2, 3, 3)))

        labels = hg.labelisation_seeded_watershed(g, edge_weights, seeds, background_label=9)

        expected = np.asarray(((1, 1, 3, 3),
                               (1, 1, 3, 3),
                               (2, 2, 3, 3),
                               (2, 2, 3, 3)))
        self.assertTrue(np.all(labels == expected))

    def test_seeded_watershed_split_minima(self):
        g = hg.get_4_adjacency_graph((2, 4))
        edge_weights = np.asarray((0, 1, 0, 2, 0, 2, 0, 1, 2, 1))
        #   x0x0x0x
        #   1 2 2 0
        #   x1x2x1x

        seeds = np.asarray(((1, 0, 0, 2),
                            (0, 0, 0, 0)))

        labels = hg.labelisation_seeded_watershed(g, edge_weights, seeds)

        expected = np.asarray(((1, 1, 1, 2),
                               (1, 1, 2, 2)))
        # other possible results:
        # ((1, 1, 2, 2),
        # (1, 1, 2, 2))
        # or
        # ((1, 2, 2, 2),
        # (1, 1, 2, 2))
        self.assertTrue(np.all(labels == expected))

    def test_seeded_watershed_disconnected_seeds(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 0, 2, 0, 0, 1, 2))
        #   x1x2x
        #   0 0 0
        #   x1x2x

        seeds = np.asarray(((5, 7, 5),
                            (0, 0, 0)))

        labels = hg.labelisation_seeded_watershed(g, edge_weights, seeds)

        expected = np.asarray(((5, 7, 5),
                               (5, 7, 5)))
        self.assertTrue(np.all(labels == expected))

    def test_seeded_watershed_seeds_not_in_minima(self):
        g = hg.get_4_adjacency_graph((2, 4))
        edge_weights = np.asarray((0, 2, 0, 2, 1, 2, 2, 1, 0, 0))
        #   x0x0x1x
        #   2 2 2 2
        #   x1x0x0x

        seeds = np.asarray(((0, 0, 0, 1),
                            (2, 0, 0, 0)))

        labels = hg.labelisation_seeded_watershed(g, edge_weights, seeds)

        expected = np.asarray(((1, 1, 1, 1),
                               (2, 2, 2, 2)))
        self.assertTrue(np.all(labels == expected))


if __name__ == '__main__':
    unittest.main()
