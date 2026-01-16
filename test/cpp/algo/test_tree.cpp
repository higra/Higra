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
#include "higra/graph.hpp"
#include "higra/algo/tree.hpp"
#include "higra/structure/array.hpp"
#include <xtensor/views/xindex_view.hpp>

using namespace hg;

namespace test_tree_algorithm {

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;

    TEST_CASE("tree reconstruct leaf data", "[tree_algorithm]") {

        auto tree = data.t;
        array_2d<int> input{{1, 8},
                            {2, 7},
                            {3, 6},
                            {4, 5},
                            {5, 4},
                            {6, 3},
                            {7, 2},
                            {8, 1}};

        array_1d<bool> condition{true, false, true, false, true, true, false, false};

        auto output = reconstruct_leaf_data(tree, input, condition);
        array_2d<int> ref{{8, 1},
                          {2, 7},
                          {7, 2},
                          {4, 5},
                          {7, 2}};
        REQUIRE(xt::allclose(ref, output));
    }

    TEST_CASE("tree labelisation horizontal cut", "[tree_algorithm]") {

        auto tree = data.t;
        array_1d<double> altitudes{0, 0, 0, 0, 0, 1, 0, 2};

        array_1d<int> ref_t0{1, 2, 3, 3, 3};
        array_1d<int> ref_t1{1, 1, 2, 2, 2};
        array_1d<int> ref_t2{1, 1, 1, 1, 1};

        auto output_t0 = labelisation_horizontal_cut_from_threshold(tree, altitudes, 0);
        auto output_t1 = labelisation_horizontal_cut_from_threshold(tree, altitudes, 1);
        auto output_t2 = labelisation_horizontal_cut_from_threshold(tree, altitudes, 2);

        REQUIRE(is_in_bijection(ref_t0, output_t0));
        REQUIRE(is_in_bijection(ref_t1, output_t1));
        REQUIRE(is_in_bijection(ref_t2, output_t2));
    }

    TEST_CASE("tree labelisation supervertices", "[tree_algorithm]") {

        auto tree = data.t;
        array_1d<double> altitudes{0, 0, 0, 0, 0, 1, 0, 2};

        array_1d<int> ref{0, 1, 2, 2, 2};

        auto output = labelisation_hierarchy_supervertices(tree, altitudes);

        REQUIRE(is_in_bijection(ref, output));
        REQUIRE(xt::amin(output)() == 0);
        REQUIRE(xt::amax(output)() == 2);
    }

    TEST_CASE("tree supervertices hierarchy", "[tree_algorithm]") {

        tree t(array_1d<index_t>{9, 9, 9, 10, 10, 12, 13, 11, 11, 14, 12, 15, 13, 14, 15, 15});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3};
        auto res = supervertices_hierarchy(t, altitudes);
        auto &tree_res = res.tree;
        auto &supervertex_labelisation_res = res.supervertex_labelisation;
        auto &node_map_res = res.node_map;

        tree tree_ref(array_1d<index_t>{5, 4, 4, 6, 5, 6, 6});
        REQUIRE(test_tree_isomorphism(tree_res, tree_ref));

        array_1d<index_t> supervertex_labelisation_ref{0, 0, 0, 1, 1, 1, 2, 3, 3};
        REQUIRE(is_in_bijection(supervertex_labelisation_ref, supervertex_labelisation_res));

        array_1d<index_t> node_map_ref{9, 12, 6, 11, 13, 14, 15};
        REQUIRE((node_map_ref == node_map_res));

    }

    TEST_CASE("tree test isomorphism", "[tree_algorithm]") {

        tree t1(array_1d<index_t>{5, 5, 6, 6, 7, 8, 7, 8, 8});
        tree t2(array_1d<index_t>{6, 6, 5, 5, 7, 7, 8, 8, 8});
        tree t3(array_1d<index_t>{7, 7, 5, 5, 6, 6, 8, 8, 8});

        REQUIRE(test_tree_isomorphism(t1, t2));
        REQUIRE(test_tree_isomorphism(t2, t1));
        REQUIRE(test_tree_isomorphism(t1, t3));
        REQUIRE(test_tree_isomorphism(t3, t1));
        REQUIRE(test_tree_isomorphism(t2, t3));
        REQUIRE(test_tree_isomorphism(t3, t2));

        tree t4(array_1d<index_t>{5, 5, 7, 6, 6, 8, 7, 8, 8});

        REQUIRE(!test_tree_isomorphism(t1, t4));
        REQUIRE(!test_tree_isomorphism(t2, t4));
        REQUIRE(!test_tree_isomorphism(t3, t4));
        REQUIRE(!test_tree_isomorphism(t4, t1));
        REQUIRE(!test_tree_isomorphism(t4, t2));
        REQUIRE(!test_tree_isomorphism(t4, t3));
    }

    TEST_CASE("tree binary labelisation from markers", "[tree_algorithm]") {

        tree t(array_1d<index_t>{9, 9, 9, 10, 10, 12, 13, 11, 11, 14, 12, 15, 13, 14, 15, 15});
        array_1d<char> object_marker{0, 1, 0, 1, 0, 0, 0, 0, 0};
        array_1d<char> background_marker{1, 0, 0, 0, 0, 0, 1, 0, 0};

        auto labelisation = binary_labelisation_from_markers(t, object_marker, background_marker);

        array_1d<char> ref_labelisation{0, 1, 0, 1, 1, 1, 0, 0, 0};

        REQUIRE((labelisation == ref_labelisation));
    }

    TEST_CASE("tree sort hierarchy w.r.t. altitudes", "[tree_algorithm]") {
        tree t(array_1d<index_t>{8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14});
        array_1d<int> altitudes{0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2, 4, 6, 5, 7};

        auto res = sort_hierarchy_with_altitudes(t, altitudes);

        array_1d<index_t> ref_par{10, 10, 8, 8, 9, 9, 11, 12, 13, 11, 13, 12, 14, 14, 14};
        REQUIRE((ref_par == parents(res.tree)));

        array_1d<int> ref_altitudes{0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7};
        REQUIRE((ref_altitudes == xt::index_view(altitudes, res.node_map)));
    }

   TEST_CASE("sub tree", "[tree_sub_tree]") {
        tree t(array_1d<index_t>{8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14});

        // full tree
        auto res1 = sub_tree(t, 14);
        REQUIRE((t.parents() == res1.tree.parents()));
        REQUIRE((xt::arange<index_t>(num_vertices(t)) == res1.node_map));

        // normal
        auto res2 = sub_tree(t, 13);
        array_1d<index_t> ref2_par{4, 4, 5, 6, 5, 6, 6};
        array_1d<index_t> ref2_node_map{4, 5, 6, 7, 10, 11, 13};
        REQUIRE((ref2_par == res2.tree.parents()));
        REQUIRE((ref2_node_map == res2.node_map));

        // leaf
        auto res3 = sub_tree(t, 3);
        array_1d<index_t> ref3_par{0};
        array_1d<index_t> ref3_node_map{3};
        REQUIRE((ref3_par == res3.tree.parents()));
        REQUIRE((ref3_node_map == res3.node_map));
    }

}