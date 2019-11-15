/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "../test_utils.hpp"
#include "higra/algo/tree_fusion.hpp"
#include "higra/hierarchy/component_tree.hpp"
#include "higra/image/graph_image.hpp"

using namespace hg;

namespace test_tree_fusion {

    TEST_CASE("tree_fusion_depth_map 1", "[tree_fusion]") {
        array_1d<int> p1{5, 5, 6, 6, 6, 7, 7, 7};
        array_1d<int> p2{7, 7, 6, 5, 5, 6, 7, 7};

        tree t1(p1);
        tree t2(p2);

        auto res = tree_fusion_depth_map(std::vector<tree *>{&t1, &t2});

        array_1d<int> expected{2, 2, 2, 3, 3};

        auto diff = xt::eval(expected - res);
        REQUIRE(xt::sum(diff - diff(0))() == 0);
    }

    TEST_CASE("tree_fusion_depth_map 2", "[tree_fusion]") {

        array_1d<int> p1{4, 4, 6, 5, 6, 6, 6};

        array_1d<int> p2{4, 5, 5, 5, 5, 5};

        auto g = get_4_adjacency_implicit_graph({1, 5});

        tree t1(p1);
        tree t2(p2);

        auto res = tree_fusion_depth_map(std::vector<tree *>{&t1, &t2});

        array_1d<int> expected{3, 2, 1, 2};

        auto diff = xt::eval(expected - res);
        REQUIRE(xt::sum(diff - diff(0))() == 0);
    }

    TEST_CASE("tree_fusion_depth_map 3", "[tree_fusion]") {

        array_1d<int> im1{0, 0, 0, 0, 0, 0, 0,
                          3, 3, 3, 2, 1, 1, 1,
                          3, 3, 3, 2, 1, 1, 1,
                          3, 3, 3, 2, 1, 1, 1,
                          2, 2, 2, 2, 1, 1, 1,
                          1, 1, 1, 1, 1, 0, 0};

        array_1d<int> im2{0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 2, 1, 1, 1, 2, 0,
                          0, 1, 1, 1, 1, 2, 0,
                          0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0};

        auto g = get_4_adjacency_implicit_graph({6, 7});

        auto t1 = component_tree_max_tree(g, im1);
        auto t2 = component_tree_max_tree(g, im2);

        auto res = tree_fusion_depth_map(std::vector<tree *>{&t1.tree, &t2.tree});

        array_1d<int> expected{0, 0, 0, 0, 0, 0, 0,
                               3, 3, 3, 2, 1, 1, 1,
                               3, 4, 3, 2, 2, 3, 1,
                               3, 3, 3, 2, 2, 3, 1,
                               2, 2, 2, 2, 1, 1, 1,
                               1, 1, 1, 1, 1, 0, 0};

        auto diff = xt::eval(expected - res);
        REQUIRE(xt::sum(diff - diff(0))() == 0);
    }


}