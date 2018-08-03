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
#include "higra/algo/graph_image.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "test_utils.hpp"
#include "xtensor/xindex_view.hpp"

BOOST_AUTO_TEST_SUITE(hierarchyCore);

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<long>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;


    BOOST_AUTO_TEST_CASE(testBPTtrivial) {

        auto graph = get_4_adjacency_graph({1, 2});

        xt::xarray<double> edge_weights{2};

        auto res = bptCanonical(graph, edge_weights);
        auto tree = std::get<0>(res);
        auto altitude = std::get<1>(res);
        auto mst = std::get<2>(res);
        BOOST_CHECK(num_vertices(tree) == 3);
        BOOST_CHECK(num_edges(tree) == 2);
        BOOST_CHECK(xt::allclose(tree.parents(), xt::xarray<unsigned int>({2, 2, 2})));
        BOOST_CHECK(xt::allclose(altitude, xt::xarray<double>({0, 0, 2})));
        BOOST_CHECK(num_vertices(mst) == 2);
        BOOST_CHECK(num_edges(mst) == 1);

    }


    BOOST_AUTO_TEST_CASE(testBPT) {

        auto graph = get_4_adjacency_graph({2, 3});

        xt::xarray<double> edge_weights{1, 0, 2, 1, 1, 1, 2};

        auto res = bptCanonical(graph, edge_weights);
        auto tree = std::get<0>(res);
        auto altitude = std::get<1>(res);
        auto mst = std::get<2>(res);
        BOOST_CHECK(num_vertices(tree) == 11);
        BOOST_CHECK(num_edges(tree) == 10);
        BOOST_CHECK(xt::allclose(hg::parents(tree), xt::xarray<unsigned int>({6, 7, 9, 6, 8, 9, 7, 8, 10, 10, 10})));
        BOOST_CHECK(xt::allclose(altitude, xt::xarray<double>({0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2})));
        BOOST_CHECK(num_vertices(mst) == 6);
        BOOST_CHECK(num_edges(mst) == 5);
        std::vector<std::pair<index_t, index_t> > ref = {{0, 3},
                                                     {0, 1},
                                                     {1, 4},
                                                     {2, 5},
                                                     {1, 2}};
        for (index_t i = 0; i < (index_t)ref.size(); i++) {
            auto e = edge(i, mst);
            BOOST_CHECK(e == ref[i]);
        }
    }


    BOOST_AUTO_TEST_CASE(testTreeSimplification) {

        auto t = data.t;

        hg::array_1d<double> altitudes{0, 0, 0, 0, 0, 1, 2, 2};

        auto criterion = xt::equal(altitudes, xt::index_view(altitudes, t.parents()));

        auto res = hg::simplify_tree(t, criterion);
        auto nt = res.first;
        auto nm = res.second;

        BOOST_CHECK(num_vertices(nt) == 7);

        hg::array_1d<index_t> refp{5, 5, 6, 6, 6, 6, 6};
        BOOST_CHECK(refp == hg::parents(nt));

        hg::array_1d<index_t> refnm{0, 1, 2, 3, 4, 5, 7};
        BOOST_CHECK(refnm == nm);
    }

BOOST_AUTO_TEST_SUITE_END();