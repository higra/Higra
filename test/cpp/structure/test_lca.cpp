/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "xtensor/xio.hpp"
#include "higra/graph.hpp"
#include "higra/image/graph_image.hpp"
#include "higra/structure/lca_fast.hpp"
#include "../test_utils.hpp"

namespace lca {

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<index_t>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;


    TEST_CASE("lca pairs of vertices", "[lca]") {
        auto t = data.t;
        lca_fast lca(t);
        REQUIRE(lca.lca(0, 0) == 0);
        REQUIRE(lca.lca(3, 3) == 3);
        REQUIRE(lca.lca(5, 5) == 5);
        REQUIRE(lca.lca(7, 7) == 7);
        REQUIRE(lca.lca(0, 1) == 5);
        REQUIRE(lca.lca(1, 0) == 5);
        REQUIRE(lca.lca(2, 3) == 6);
        REQUIRE(lca.lca(2, 4) == 6);
        REQUIRE(lca.lca(3, 4) == 6);
        REQUIRE(lca.lca(5, 6) == 7);
        REQUIRE(lca.lca(0, 2) == 7);
        REQUIRE(lca.lca(1, 4) == 7);
        REQUIRE(lca.lca(2, 6) == 6);
    }

    TEST_CASE("lca iterators", "[lca]") {
        auto g = get_4_adjacency_graph({2, 2});
        tree t(array_1d<index_t>{4, 4, 5, 5, 6, 6, 6});
        lca_fast lca(t);
        auto l = lca.lca(edge_iterator(g));
        array_1d<index_t> ref{4, 6, 6, 5};
        REQUIRE((l == ref));
    }

    TEST_CASE("lca tensors", "[lca]") {
        tree t(array_1d<index_t>{4, 4, 5, 5, 6, 6, 6});
        lca_fast lca(t);
        array_1d<index_t> v1{0, 0, 1, 3};
        array_1d<index_t> v2{0, 3, 0, 0};
        auto l = lca.lca(v1, v2);
        array_1d<index_t> ref{0, 6, 4, 6};
        REQUIRE((l == ref));
    }

    TEST_CASE("lca serialization", "[lca]") {
        tree t(array_1d<index_t>{4, 4, 5, 5, 6, 6, 6});
        lca_fast lca(t);
        array_1d<index_t> v1{0, 0, 1, 3};
        array_1d<index_t> v2{0, 3, 0, 0};


        auto state = lca.get_state();

        lca_fast lca2;
        lca2.set_state(state);

        array_1d<index_t> ref{0, 6, 4, 6};
        auto l = lca2.lca(v1, v2);
        REQUIRE((l == ref));

        lca_fast lca3;
        lca3.set_state(std::move(state));

        auto l2 = lca3.lca(v1, v2);
        REQUIRE((l2 == ref));
    }
}