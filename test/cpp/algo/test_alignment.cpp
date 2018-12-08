/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>
#include "higra/algo/alignment.hpp"
#include "higra/image/graph_image.hpp"
#include "../test_utils.hpp"


using namespace hg;

BOOST_AUTO_TEST_SUITE(alignment_test);

    BOOST_AUTO_TEST_CASE(test_project_fine_to_coarse_labelisation) {

        array_1d<index_t> fine_labels{0, 1, 2, 3, 4, 2, 3, 4, 2};
        array_1d<index_t> coarse_labels{0, 1, 1, 0, 2, 2, 0, 2, 2};

        auto map = project_fine_to_coarse_labelisation(fine_labels, coarse_labels);

        array_1d<index_t> ref_map{0, 1, 2, 0, 2};
        BOOST_CHECK(ref_map == map);
    }

    BOOST_AUTO_TEST_CASE(test_hierarchy_alignment) {

        auto g = get_4_adjacency_graph({3, 3});
        array_1d<int> fine_labels{0, 1, 2, 3, 4, 2, 3, 4, 2};

        auto aligner = make_hierarchy_aligner_from_labelisation(g, fine_labels);

        hg::tree t(array_1d<index_t>{9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2};
        auto sm = aligner.align_hierarchy(t, altitudes);
        auto sm_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm);

        array_1d<int> sm_coarse{2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0};
        auto sm2 = aligner.align_hierarchy(g, sm_coarse);
        auto sm2_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm2);

        array_2d<int> sm_k_ref = {{0, 2, 0, 1, 0},
                                  {0, 2, 1, 1, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0}};

        BOOST_CHECK(sm_k == sm_k_ref);
        BOOST_CHECK(sm2_k == sm_k_ref);
    }

    BOOST_AUTO_TEST_CASE(test_hierarchy_alignment2) {

        auto g = get_4_adjacency_graph({3, 3});
        array_1d<int> edge_weights{1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1};

        auto aligner = make_hierarchy_aligner_from_graph_cut(g, edge_weights);

        hg::tree t(array_1d<index_t>{9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2};
        auto sm = aligner.align_hierarchy(t, altitudes);
        auto sm_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm);

        array_1d<int> sm_coarse{2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0};
        auto sm2 = aligner.align_hierarchy(g, sm_coarse);
        auto sm2_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm2);

        array_2d<int> sm_k_ref = {{0, 2, 0, 1, 0},
                                  {0, 2, 1, 1, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0}};

        BOOST_CHECK(sm_k == sm_k_ref);
        BOOST_CHECK(sm2_k == sm_k_ref);
    }

BOOST_AUTO_TEST_SUITE_END();