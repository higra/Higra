/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/algo/alignment.hpp"
#include "higra/image/graph_image.hpp"
#include "../test_utils.hpp"
#include "higra/algo/tree.hpp"

using namespace hg;

namespace test_alignment {

    TEST_CASE("project fine to coarse labelisation", "[alignment]") {

        array_1d<index_t> fine_labels{0, 1, 2, 3, 4, 2, 3, 4, 2};
        array_1d<index_t> coarse_labels{0, 1, 1, 0, 2, 2, 0, 2, 2};

        auto map = project_fine_to_coarse_labelisation(fine_labels, coarse_labels);

        array_1d<index_t> ref_map{0, 1, 2, 0, 2};
        REQUIRE((ref_map == map));
    }

    TEST_CASE("hierarchy alignement", "[alignment]") {

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

        REQUIRE((sm_k == sm_k_ref));
        REQUIRE((sm2_k == sm_k_ref));
    }

    TEST_CASE("hierarchy alignment 2", "[alignment]") {

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

        REQUIRE((sm_k == sm_k_ref));
        REQUIRE((sm2_k == sm_k_ref));
    }

    TEST_CASE("hierarchy alignment with rag", "[alignment]") {

        auto g = get_4_adjacency_graph({3, 3});
        array_1d<int> edge_weights{1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1};

        auto aligner = make_hierarchy_aligner_from_graph_cut(g, edge_weights);


        hg::tree t(array_1d<index_t>{9, 10, 10, 9, 11, 11, 9, 11, 11, 13, 12, 12, 13, 13});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2};

        auto res = supervertices_hierarchy(t, altitudes);
        auto &tree_res = res.tree;
        auto &supervertex_labelisation_res = res.supervertex_labelisation;
        auto &node_map_res = res.node_map;

        altitudes = xt::index_view(altitudes, node_map_res);

        std::cout << tree_res.parents() << std::endl;
        std::cout << supervertex_labelisation_res << std::endl;
        std::cout << altitudes << std::endl;

        auto sm = aligner.align_hierarchy(supervertex_labelisation_res, tree_res, altitudes);
        auto sm_k = graph_4_adjacency_2_khalimsky(g, {3, 3}, sm);

        array_2d<int> sm_k_ref = {{0, 2, 0, 1, 0},
                                  {0, 2, 1, 1, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0},
                                  {0, 2, 0, 0, 0}};

        REQUIRE((sm_k == sm_k_ref));
    }

}