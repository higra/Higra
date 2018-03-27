#include <boost/test/unit_test.hpp>

#include "embedding.hpp"
#include "xtensor/xview.hpp"
//
// Created by user on 3/9/18.
//




BOOST_AUTO_TEST_SUITE(TestSuitePoint);

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid1d) {
        hg::embedding_grid e1{10};
        BOOST_CHECK(e1.size() == 10);
        BOOST_CHECK(e1.dims() == 1);

        BOOST_CHECK(e1.contains({5}));
        BOOST_CHECK(!e1.contains({-2}));
        BOOST_CHECK(!e1.contains({12}));

        auto p1 = e1.lin2grid(2);
        std::vector<long> p2{2};

        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p2.begin()));

        std::vector<long> p3{15};
        BOOST_CHECK(e1.contains(p1));
        BOOST_CHECK(!e1.contains(p3));
    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid2d) {
        hg::embedding_grid e1{10, 5};
        BOOST_CHECK(e1.size() == 50);
        BOOST_CHECK(e1.dims() == 2);

        std::vector<long> p1 = {0, 3};
        auto p1t = e1.lin2grid(3);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1t) == 3);

        std::vector<long> p2 = {2, 4};
        auto p2t = e1.lin2grid(14);
        BOOST_CHECK(std::equal(p2.begin(), p2.end(), p2t.begin()));
        BOOST_CHECK(e1.grid2lin(p2) == 14);

        BOOST_CHECK(e1.contains(p1t));
        BOOST_CHECK(e1.contains(p2t));

        std::vector<long> p3 = {-1, 2};
        std::vector<long> p4 = {6, -1};
        std::vector<long> p5 = {10, 2};
        std::vector<long> p6 = {6, 5};
        BOOST_CHECK(!e1.contains(p3));
        BOOST_CHECK(!e1.contains(p4));
        BOOST_CHECK(!e1.contains(p5));
        BOOST_CHECK(!e1.contains(p6));

    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGrid3d) {
        hg::embedding_grid e1{10, 5, 2};
        BOOST_CHECK(e1.size() == 100);
        BOOST_CHECK(e1.dims() == 3);

        std::vector<long> p1 = {3, 2, 1};
        auto p1t = e1.lin2grid(35);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1) == 35);
    }

    BOOST_AUTO_TEST_CASE(CreateEmbeddingGridFromXTensor) {
        xt::xarray<ulong> shape = {10, 5, 2};
        hg::embedding_grid e1 = hg::embedding_grid::make_embedding_grid(shape);
        BOOST_CHECK(e1.size() == 100);
        BOOST_CHECK(e1.dims() == 3);

        std::vector<long> p1 = {3, 2, 1};
        auto p1t = e1.lin2grid(35);
        BOOST_CHECK(std::equal(p1.begin(), p1.end(), p1t.begin()));
        BOOST_CHECK(e1.grid2lin(p1) == 35);
    }


    BOOST_AUTO_TEST_CASE(grid2LinV) {
        xt::xarray<ulong> shape = {10, 5, 2};
        hg::embedding_grid e1 = hg::embedding_grid::make_embedding_grid(shape);

        xt::xarray<long> coord = {{0, 0, 0},
                                  {0, 0, 1},
                                  {0, 0, 2},
                                  {3, 2, 1}};
        auto linCoord = e1.grid2linV(coord);
        BOOST_CHECK(linCoord.shape().size() == 1);
        BOOST_CHECK(linCoord.shape()[0] == 4);
        std::vector<ulong> resShape = {3};

        BOOST_CHECK(linCoord(0) == 0);
        BOOST_CHECK(linCoord(1) == 1);
        BOOST_CHECK(linCoord(2) == 2);
        BOOST_CHECK(linCoord(3) == 35);
    }


    BOOST_AUTO_TEST_CASE(lin2GridV) {
        xt::xarray<ulong> shape = {5, 10};
        hg::embedding_grid e1 = hg::embedding_grid::make_embedding_grid(shape);

        xt::xarray<ulong> coordLin = {{0,  1,  2,  3},
                                      {22, 42, 43, 44}};
        xt::xarray<long> coords = {{{0, 0}, {0, 1}, {0, 2}, {0, 3}},
                                   {{2, 2}, {4, 2}, {4, 3}, {4, 4}}};


        auto res = e1.lin2grid(coordLin);

        BOOST_CHECK(res == coords);
    }

    BOOST_AUTO_TEST_CASE(containsV) {
        xt::xarray<ulong> shape = {5, 10};
        hg::embedding_grid e1 = hg::embedding_grid::make_embedding_grid(shape);

        xt::xarray<long> coords{{{0, 0}, {3, 8}, {-1, 2}},
                                {{2, 4}, {5, 5}, {43, 44}}};

        xt::xarray<bool> ref{{true, true,  false},
                             {true, false, false}};

        auto res = e1.containsV(coords);

        BOOST_CHECK(res == ref);
    }


BOOST_AUTO_TEST_SUITE_END();
