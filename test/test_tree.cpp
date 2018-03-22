//
// Created by user on 3/22/18.
//

#include <boost/test/unit_test.hpp>
#include "xtensor/xio.hpp"
#include "graph.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE(boost_treegraph);

    using namespace hg;
    using namespace std;


    struct _data {

        hg::tree t;

        _data() : t({5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;


    BOOST_AUTO_TEST_CASE(sizeTree) {
        auto t = data.t;
        BOOST_CHECK(t.root() == 7);
        BOOST_CHECK(t.num_vertices() == 8);
        BOOST_CHECK(t.num_edges() == 7);
        BOOST_CHECK(t.num_leaves() == 5);
    }

    BOOST_AUTO_TEST_CASE(treeFail) {
        BOOST_REQUIRE_THROW(hg::tree({5, 0, 6, 6, 6, 7, 7, 7}), std::runtime_error);
        BOOST_REQUIRE_THROW(hg::tree({5, 1, 6, 6, 6, 7, 7, 7}), std::runtime_error);
        BOOST_REQUIRE_THROW(hg::tree({5, 1, 6, 6, 6, 7, 7, 2}), std::runtime_error);
        BOOST_REQUIRE_THROW(hg::tree({2, 2, 4, 4, 4}), std::runtime_error);
    }


BOOST_AUTO_TEST_SUITE_END();