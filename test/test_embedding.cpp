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
#include "higra/structure/embedding.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xgenerator.hpp"
#include "xtensor/xinfo.hpp"
#include "xtensor/xeval.hpp"


BOOST_AUTO_TEST_SUITE(embedding);

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid1d) {
        hg::embedding_grid_1d e1{10};
        BOOST_CHECK(e1.size() == 10);
        BOOST_CHECK(e1.dimension() == 1);

        BOOST_CHECK(e1.contains({5}));
        BOOST_CHECK(!e1.contains({-2}));
        BOOST_CHECK(!e1.contains({12}));

        auto p1 = e1.lin2grid(2);
        std::vector<long> p2{2};

        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p2.begin()));
        hg::point_1d_i p3{15};

        BOOST_CHECK((e1.contains(p1)));
        BOOST_CHECK(!e1.contains(p3));
    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid2d) {
        hg::embedding_grid_2d e1{10, 5};
        BOOST_CHECK(e1.size() == 50);
        BOOST_CHECK(e1.dimension() == 2);

        hg::point_2d_i p1={{0, 3}};
        auto p1t = e1.lin2grid(3);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK((e1.grid2lin(p1t)) == 3);

        hg::point_2d_i p2{{2, 4}};
        auto p2t = e1.lin2grid(14);
        BOOST_CHECK(std::equal(p2.begin(), p2.end(), p2t.begin()));
        BOOST_CHECK(e1.grid2lin(p2) == 14);

        BOOST_CHECK(e1.contains(p1t));
        BOOST_CHECK(e1.contains(p2t));

        hg::point_2d_i p3{{-1l, 2l}};
        hg::point_2d_i p4{{6l, -1l}};
        hg::point_2d_i p5{{10l, 2l}};
        hg::point_2d_i p6{{6l, 5l}};
        BOOST_CHECK(!e1.contains(p3));
        BOOST_CHECK(!e1.contains(p4));
        BOOST_CHECK(!e1.contains(p5));
        BOOST_CHECK(!e1.contains(p6));
    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid3d) {
        hg::embedding_grid_3d e1{10, 5, 2};
        BOOST_CHECK(e1.size() == 100);
        BOOST_CHECK(e1.dimension() == 3);

        hg::point_3d_i p1{{3, 2, 1}};
        auto p1t = e1.lin2grid(35);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1) == 35);
    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGridFromXTensorShape) {
        xt::xarray<int> a = xt::zeros<int>({10, 5, 2});
        hg::embedding_grid_3d e1(a.shape());
        BOOST_CHECK(e1.size() == 100);
        BOOST_CHECK(e1.dimension() == 3);

        hg::point_3d_i p1{{3, 2, 1}};
        auto p1t = e1.lin2grid(35);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1) == 35);
    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGridFromXTensor) {
        xt::xarray<unsigned long> shape = {10, 5, 2};
        hg::embedding_grid_3d e1(shape);
        BOOST_CHECK(e1.size() == 100);
        BOOST_CHECK(e1.dimension() == 3);

        hg::point_3d_i p1 = {{3, 2, 1}};
        auto p1t = e1.lin2grid(35);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1) == 35);
    }


    BOOST_AUTO_TEST_CASE(grid2LinV) {
        xt::xarray<unsigned long> shape = {10, 5, 2};
        hg::embedding_grid_3d e1(shape);

        xt::xarray<long> coord = {{0, 0, 0},
                                  {0, 0, 1},
                                  {0, 0, 2},
                                  {3, 2, 1}};
        auto linCoord = e1.grid2lin(coord);
        BOOST_CHECK(linCoord.shape().size() == 1);
        BOOST_CHECK(linCoord.shape()[0] == 4);
        std::vector<unsigned long> resShape = {3};

        BOOST_CHECK(linCoord(0) == 0);
        BOOST_CHECK(linCoord(1) == 1);
        BOOST_CHECK(linCoord(2) == 2);
        BOOST_CHECK(linCoord(3) == 35);
    }


    BOOST_AUTO_TEST_CASE(lin2GridV) {
        xt::xarray<unsigned long> shape = {5, 10};
        hg::embedding_grid_2d e1(shape);

        xt::xarray<unsigned long> coordLin = {{0,  1,  2,  3},
                                              {22, 42, 43, 44}};
        xt::xarray<long> coords = {{{0, 0}, {0, 1}, {0, 2}, {0, 3}},
                                   {{2, 2}, {4, 2}, {4, 3}, {4, 4}}};

        auto res = e1.lin2grid(coordLin);
        BOOST_CHECK(res == coords);
    }

    BOOST_AUTO_TEST_CASE(containsV) {
        xt::xarray<unsigned long> shape = {5, 10};
        hg::embedding_grid_2d e1(shape);

        xt::xarray<long> coords{{{0, 0}, {3, 8}, {-1, 2}},
                                {{2, 4}, {5, 5}, {43, 44}}};

        xt::xarray<bool> ref{{true, true,  false},
                             {true, false, false}};

        auto res = e1.contains(coords);
        BOOST_CHECK(res == ref);
    }


BOOST_AUTO_TEST_SUITE_END();
