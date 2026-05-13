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

    def test_batch_remove_equals_sequential(self):
        g = hg.get_4_adjacency_graph((4, 4))
        edge_weights = np.asarray((1, 2, 5, 5, 4, 8, 1, 4, 3, 4, 4, 1,
                                   5, 2, 6, 2, 5, 2, 0, 7, 0, 3, 4, 0))
        sv = np.array([0, 1, 4, 14, 15])
        sl = np.array([1, 1, 1, 2, 2])

        a = hg.IncrementalWatershedCut(g, edge_weights)
        a.add_seeds(sv, sl)
        a.remove_seeds(np.array([1, 14]))

        b = hg.IncrementalWatershedCut(g, edge_weights)
        b.add_seeds(sv, sl)
        b.remove_seeds(np.array([1]))
        b.remove_seeds(np.array([14]))

        self.assertTrue(np.all(a.get_labeling() == b.get_labeling()))

    def test_batch_remove_both_sides_of_edge(self):
        g = hg.get_4_adjacency_graph((2, 2))
        edge_weights = np.asarray((1, 2, 3, 4))
        iws = hg.IncrementalWatershedCut(g, edge_weights)
        iws.add_seeds(np.array([0, 3]), np.array([1, 2]))
        iws.remove_seeds(np.array([0, 3]))
        self.assertTrue(np.all(iws.get_labeling().ravel() == 0))

    def test_interactive_churn(self):
        """
        Regression test for scenario 07: interactive single seed churn with label changes.
        Adds 10 seeds on a 100x100 grid, then alternately removes and re-adds 5 of them
        with changed labels, verifying consistency with labelisation_seeded_watershed
        at each step.
        """
        g = hg.get_4_adjacency_graph((100, 100))
        rng = np.random.RandomState(42)
        edge_weights = rng.rand(g.num_edges()).astype(np.float64)

        # Deterministic vertices matching scenario 07
        sv = np.array([4098, 8671, 7466, 737, 5621,
                       5954, 3197, 7184, 7657, 3043], dtype=np.int64)
        initial_labels = np.arange(1, 11, dtype=np.int64)
        changed_label = 99

        iws = hg.IncrementalWatershedCut(g, edge_weights)

        # Add all 10 seeds
        iws.add_seeds(sv, initial_labels)
        labels = iws.get_labeling().ravel()

        # Baseline via full seeded watershed
        seeds = np.zeros(g.num_vertices(), dtype=np.int64)
        seeds[sv] = initial_labels
        baseline = hg.labelisation_seeded_watershed(g, edge_weights, seeds.reshape(100, 100))
        self.assertTrue(np.all(labels == baseline.ravel()))

        # Churn: remove and re-add first 5 seeds with label 99
        for v in sv[:5]:
            # Remove
            iws.remove_seeds(np.array([v]))
            seeds[v] = 0
            labels = iws.get_labeling().ravel()
            baseline = hg.labelisation_seeded_watershed(
                g, edge_weights, seeds.reshape(100, 100))
            self.assertTrue(np.all(labels == baseline.ravel()))

            # Re-add with changed label
            iws.add_seeds(np.array([v]), np.array([changed_label]))
            seeds[v] = changed_label
            labels = iws.get_labeling().ravel()
            baseline = hg.labelisation_seeded_watershed(
                g, edge_weights, seeds.reshape(100, 100))
            self.assertTrue(np.all(labels == baseline.ravel()))

    def test_decuts_ancestor_descendant(self):
        """
        Batch remove producing two de-cuts where one BPT node is the ancestor
        of the other. Exercises Pass 2a across BPT levels.

        Path graph 1x8 with edge weights chosen so the canonical BPT is the
        balanced binary tree (e_01, e_23, e_45, e_67 fuse first; then e_12,
        e_56; then e_34 at the root). Seeds at 0, 1, 3, 4 create cuts at
        p_01 (depth 2), p_0123 (depth 1) and root (depth 1 of right side).
        Removing 0 and 3 in one batch triggers de-cuts at p_01 and p_0123,
        with p_01 a descendant of p_0123 in the BPT.
        """
        g = hg.get_4_adjacency_graph((1, 8))
        edge_weights = np.asarray([1, 5, 2, 7, 3, 6, 4], dtype=np.float64)

        sv = np.array([0, 1, 3, 4], dtype=np.int64)
        sl = np.array([10, 20, 30, 40], dtype=np.int64)

        iws = hg.IncrementalWatershedCut(g, edge_weights)
        iws.add_seeds(sv, sl)
        iws.remove_seeds(np.array([0, 3], dtype=np.int64))

        seeds = np.zeros(g.num_vertices(), dtype=np.int64)
        seeds[1] = 20
        seeds[4] = 40
        expected = hg.labelisation_seeded_watershed(
            g, edge_weights, seeds.reshape(1, 8))
        self.assertTrue(np.all(iws.get_labeling() == expected))

    def test_add_seed_already_exists_raises(self):
        g = hg.get_4_adjacency_graph((2, 2))
        edge_weights = np.asarray((1, 2, 3, 4))

        iws = hg.IncrementalWatershedCut(g, edge_weights)
        iws.add_seeds(np.array([0]), np.array([1]))
        with self.assertRaises(Exception):
            iws.add_seeds(np.array([0]), np.array([2]))

    def test_add_seed_label_zero_raises(self):
        g = hg.get_4_adjacency_graph((2, 2))
        edge_weights = np.asarray((1, 2, 3, 4))

        iws = hg.IncrementalWatershedCut(g, edge_weights)
        with self.assertRaises(Exception):
            iws.add_seeds(np.array([0]), np.array([0]))


if __name__ == '__main__':
    unittest.main()
