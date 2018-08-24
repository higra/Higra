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
#include "xtensor/xio.hpp"
#include "higra/graph.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/structure/lca_fast.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE(lca);

    using namespace hg;
    using namespace std;


    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;


    BOOST_AUTO_TEST_CASE(lca) {
        auto t = data.t;
        lca_fast lca(t);
        BOOST_CHECK(lca.lca(0, 0) == 0);
        BOOST_CHECK(lca.lca(3, 3) == 3);
        BOOST_CHECK(lca.lca(5, 5) == 5);
        BOOST_CHECK(lca.lca(7, 7) == 7);
        BOOST_CHECK(lca.lca(0, 1) == 5);
        BOOST_CHECK(lca.lca(1, 0) == 5);
        BOOST_CHECK(lca.lca(2, 3) == 6);
        BOOST_CHECK(lca.lca(2, 4) == 6);
        BOOST_CHECK(lca.lca(3, 4) == 6);
        BOOST_CHECK(lca.lca(5, 6) == 7);
        BOOST_CHECK(lca.lca(0, 2) == 7);
        BOOST_CHECK(lca.lca(1, 4) == 7);
        BOOST_CHECK(lca.lca(2, 6) == 6);
    }

    BOOST_AUTO_TEST_CASE(lcaV) {
        auto g = get_4_adjacency_graph({2, 2});
        tree t(array_1d<index_t> { 4, 4, 5, 5, 6, 6, 6 });
        lca_fast lca(t);
        auto l = lca.lca(edge_iterator(g));
        array_1d<decltype(g)::vertex_descriptor> ref{4, 6, 6, 5};
        BOOST_CHECK(xt::allclose(l, ref));
    }


BOOST_AUTO_TEST_SUITE_END();