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

    def test_seeded_watershed_type_conversion(self):
        g = hg.get_4_adjacency_graph((4, 4))
        edge_weights = np.asarray((1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0)) / 10.0

        seeds = np.asarray(((1, 1, 0, 0),
                            (1, 0, 0, 0),
                            (0, 0, 0, 0),
                            (1, 1, 2, 2)), dtype=np.int32)

        labels = hg.labelisation_seeded_watershed(g, edge_weights, seeds)

        expected = np.asarray(((1, 1, 2, 2),
                               (1, 1, 2, 2),
                               (1, 1, 2, 2),
                               (1, 1, 2, 2)))
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


class TestIncrementalWatershed(unittest.TestCase):

    def test_basic(self):
        g = hg.get_4_adjacency_graph((4, 4))
        edge_weights = np.asarray((1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0))

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        iws.add_seeds(np.array([0, 1, 4]), np.array([1, 1, 1]))
        iws.add_seeds(np.array([12, 13]), np.array([2, 2]))
        iws.add_seeds(np.array([14, 15]), np.array([3, 3]))

        labels = iws.get_labeling()
        labels_flat = labels.ravel()

        self.assertEqual(labels_flat[0], 1)
        self.assertEqual(labels_flat[1], 1)
        self.assertEqual(labels_flat[4], 1)
        self.assertEqual(labels_flat[12], 2)
        self.assertEqual(labels_flat[13], 2)
        self.assertEqual(labels_flat[14], 3)
        self.assertEqual(labels_flat[15], 3)

        # All vertices should be labeled
        self.assertTrue(np.all(labels_flat != 0))

    def test_remove_seed(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 0, 2, 0, 0, 1, 2))

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        iws.add_seeds(np.array([0]), np.array([1]))
        iws.add_seeds(np.array([2]), np.array([2]))
        iws.add_seeds(np.array([4]), np.array([3]))

        labels3 = iws.get_labeling().ravel()
        self.assertEqual(labels3[0], 1)
        self.assertEqual(labels3[2], 2)
        self.assertEqual(labels3[4], 3)

        # Remove seed at vertex 4
        iws.remove_seeds(np.array([4]))

        labels2 = iws.get_labeling().ravel()
        self.assertEqual(labels2[0], 1)
        self.assertEqual(labels2[2], 2)
        self.assertIn(labels2[4], [1, 2])

    def test_shared_labels(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 0, 2, 0, 0, 1, 2))

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        iws.add_seeds(np.array([0, 2, 1]), np.array([5, 5, 7]))

        labels = iws.get_labeling().ravel()
        self.assertEqual(labels[0], 5)
        self.assertEqual(labels[2], 5)
        self.assertEqual(labels[1], 7)

    def test_no_seeds(self):
        g = hg.get_4_adjacency_graph((2, 2))
        edge_weights = np.asarray((1, 2, 3, 4))

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        labels = iws.get_labeling().ravel()
        self.assertTrue(np.all(labels == 0))

    def test_single_seed(self):
        g = hg.get_4_adjacency_graph((2, 2))
        edge_weights = np.asarray((1, 2, 3, 4))

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        iws.add_seeds(np.array([0]), np.array([1]))

        labels = iws.get_labeling().ravel()
        self.assertTrue(np.all(labels == 1))

    def test_add_then_remove_all(self):
        g = hg.get_4_adjacency_graph((2, 2))
        edge_weights = np.asarray((1, 2, 3, 4))

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        iws.add_seeds(np.array([0, 3]), np.array([1, 2]))
        labels = iws.get_labeling().ravel()
        self.assertTrue(np.all(labels != 0))

        iws.remove_seeds(np.array([0, 3]))
        labels = iws.get_labeling().ravel()
        self.assertTrue(np.all(labels == 0))

    def test_consistency_with_seeded_watershed(self):
        g = hg.get_4_adjacency_graph((4, 4))
        edge_weights = np.asarray((1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0))

        seeds = np.asarray(((1, 1, 0, 0),
                            (1, 0, 0, 0),
                            (0, 0, 0, 0),
                            (1, 1, 2, 2)))

        expected = hg.labelisation_seeded_watershed(g, edge_weights, seeds)

        iws = hg.IncrementalWatershedCut(g, edge_weights)
        seed_vertices = np.where(seeds.ravel() != 0)[0]
        seed_labels = seeds.ravel()[seed_vertices]
        iws.add_seeds(seed_vertices, seed_labels)

        labels = iws.get_labeling()
        self.assertTrue(np.all(labels == expected))

    def test_incremental_equals_batch(self):
        g = hg.get_4_adjacency_graph((4, 4))
        edge_weights = np.asarray((1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1, 5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0))

        # Add all seeds at once
        iws_batch = hg.IncrementalWatershedCut(g, edge_weights)
        iws_batch.add_seeds(np.array([0, 1, 4, 14, 15]), np.array([1, 1, 1, 2, 2]))

        # Add seeds one by one
        iws_incr = hg.IncrementalWatershedCut(g, edge_weights)
        iws_incr.add_seeds(np.array([0]), np.array([1]))
        iws_incr.add_seeds(np.array([1]), np.array([1]))
        iws_incr.add_seeds(np.array([4]), np.array([1]))
        iws_incr.add_seeds(np.array([14]), np.array([2]))
        iws_incr.add_seeds(np.array([15]), np.array([2]))

        self.assertTrue(np.all(iws_batch.get_labeling() == iws_incr.get_labeling()))

    def test_add_remove_readd(self):
        g = hg.get_4_adjacency_graph((2, 3))
        edge_weights = np.asarray((1, 0, 2, 0, 0, 1, 2))

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        iws.add_seeds(np.array([0, 5]), np.array([1, 2]))
        labels_first = iws.get_labeling().copy()

        iws.remove_seeds(np.array([5]))
        iws.add_seeds(np.array([5]), np.array([2]))
        labels_readd = iws.get_labeling()

        self.assertTrue(np.all(labels_first == labels_readd))

    def test_larger_grid(self):
        g = hg.get_4_adjacency_graph((10, 10))
        rng = np.random.RandomState(42)
        edge_weights = rng.rand(g.num_edges()).astype(np.float64)

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        # Place seeds at corners
        iws.add_seeds(np.array([0, 9, 90, 99]), np.array([1, 2, 3, 4]))

        labels = iws.get_labeling().ravel()

        # Each seed vertex has the correct label
        self.assertEqual(labels[0], 1)
        self.assertEqual(labels[9], 2)
        self.assertEqual(labels[90], 3)
        self.assertEqual(labels[99], 4)

        # All vertices labeled
        self.assertTrue(np.all(labels != 0))

        # Exactly 4 distinct labels
        self.assertEqual(len(np.unique(labels)), 4)


if __name__ == '__main__':
    unittest.main()
