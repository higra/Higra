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
#include "test_utils.hpp"
#include "higra/graph.hpp"
#include "higra/algo/tree.hpp"
#include "higra/structure/array.hpp"


using namespace hg;

BOOST_AUTO_TEST_SUITE(algo_tree);

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<long>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;

    BOOST_AUTO_TEST_CASE(test_reconstruct_leaf_data) {

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
        BOOST_CHECK(xt::allclose(ref, output));
    }

    BOOST_AUTO_TEST_CASE(test_labelisation_horizontal_cut) {

        auto tree = data.t;
        array_1d<double> altitudes{0, 0, 0, 0, 0, 1, 0, 2};

        array_1d<int> ref_t0{1, 2, 3, 3, 3};
        array_1d<int> ref_t1{1, 1, 2, 2, 2};
        array_1d<int> ref_t2{1, 1, 1, 1, 1};

        auto output_t0 = labelisation_horizontal_cut(tree, altitudes, 0);
        auto output_t1 = labelisation_horizontal_cut(tree, altitudes, 1);
        auto output_t2 = labelisation_horizontal_cut(tree, altitudes, 2);

        BOOST_CHECK(is_in_bijection(ref_t0, output_t0));
        BOOST_CHECK(is_in_bijection(ref_t1, output_t1));
        BOOST_CHECK(is_in_bijection(ref_t2, output_t2));
    }

    BOOST_AUTO_TEST_CASE(tree_isomorphism) {

        tree t1(array_1d<index_t>{5, 5, 6, 6, 7, 8, 7, 8, 8});
        tree t2(array_1d<index_t>{6, 6, 5, 5, 7, 7, 8, 8, 8});
        tree t3(array_1d<index_t>{7, 7, 5, 5, 6, 6, 8, 8, 8});

        BOOST_CHECK(testTreeIsomorphism(t1, t2));
        BOOST_CHECK(testTreeIsomorphism(t2, t1));
        BOOST_CHECK(testTreeIsomorphism(t1, t3));
        BOOST_CHECK(testTreeIsomorphism(t3, t1));
        BOOST_CHECK(testTreeIsomorphism(t2, t3));
        BOOST_CHECK(testTreeIsomorphism(t3, t2));

        tree t4(array_1d<index_t>{5, 5, 7, 6, 6, 8, 7, 8, 8});

        BOOST_CHECK(!testTreeIsomorphism(t1, t4));
        BOOST_CHECK(!testTreeIsomorphism(t2, t4));
        BOOST_CHECK(!testTreeIsomorphism(t3, t4));
        BOOST_CHECK(!testTreeIsomorphism(t4, t1));
        BOOST_CHECK(!testTreeIsomorphism(t4, t2));
        BOOST_CHECK(!testTreeIsomorphism(t4, t3));
    }

BOOST_AUTO_TEST_SUITE_END();