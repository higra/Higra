//
// Created by perretb on 29/04/18.
//
#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/graph.hpp"
#include "higra/algo/tree.hpp"
#include "higra/structure/array.hpp"


using namespace hg;

BOOST_AUTO_TEST_SUITE(algo_tree);


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