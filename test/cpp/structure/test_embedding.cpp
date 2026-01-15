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
#include "higra/structure/embedding.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/generators/xgenerator.hpp"
#include "xtensor/io/xinfo.hpp"
#include "xtensor/core/xeval.hpp"

namespace embedding {


    TEST_CASE("create embedding grid 1d", "[embedding]") {
        hg::embedding_grid_1d e1{10};
        REQUIRE(e1.size() == 10);
        REQUIRE(e1.dimension() == 1);

        REQUIRE(e1.contains({5}));
        REQUIRE(!e1.contains({-2}));
        REQUIRE(!e1.contains({12}));

        auto p1 = e1.lin2grid(2);
        std::vector<long> p2{2};

        REQUIRE(std::equal(p1.begin(), p1.end(), p2.begin()));
        hg::point_1d_i p3{15};

        REQUIRE((e1.contains(p1)));
        REQUIRE(!e1.contains(p3));
    }

    TEST_CASE("create embedding grid 2d", "[embedding]") {
        hg::embedding_grid_2d e1{10, 5};
        REQUIRE(e1.size() == 50);
        REQUIRE(e1.dimension() == 2);

        hg::point_2d_i p1 = {{0, 3}};
        auto p1t = e1.lin2grid(3);
        REQUIRE(std::equal(p1.begin(), p1.end(), p1t.begin()));
        REQUIRE((e1.grid2lin(p1t)) == 3);

        hg::point_2d_i p2{{2, 4}};
        auto p2t = e1.lin2grid(14);
        REQUIRE(std::equal(p2.begin(), p2.end(), p2t.begin()));
        REQUIRE(e1.grid2lin(p2) == 14);

        REQUIRE(e1.contains(p1t));
        REQUIRE(e1.contains(p2t));

        hg::point_2d_i p3{{-1l, 2l}};
        hg::point_2d_i p4{{6l, -1l}};
        hg::point_2d_i p5{{10l, 2l}};
        hg::point_2d_i p6{{6l, 5l}};
        REQUIRE(!e1.contains(p3));
        REQUIRE(!e1.contains(p4));
        REQUIRE(!e1.contains(p5));
        REQUIRE(!e1.contains(p6));
    }

    TEST_CASE("create embedding grid 3d", "[embedding]") {
        hg::embedding_grid_3d e1{10, 5, 2};
        REQUIRE(e1.size() == 100);
        REQUIRE(e1.dimension() == 3);

        hg::point_3d_i p1{{3, 2, 1}};
        auto p1t = e1.lin2grid(35);
        REQUIRE(std::equal(p1.begin(), p1.end(), p1t.begin()));
        REQUIRE(e1.grid2lin(p1) == 35);
    }

    TEST_CASE("create embedding grid from xtensor shape", "[embedding]") {
        xt::xarray<int> a = xt::zeros<int>({10, 5, 2});
        hg::embedding_grid_3d e1(a.shape());
        REQUIRE(e1.size() == 100);
        REQUIRE(e1.dimension() == 3);

        hg::point_3d_i p1{{3, 2, 1}};
        auto p1t = e1.lin2grid(35);
        REQUIRE(std::equal(p1.begin(), p1.end(), p1t.begin()));
        REQUIRE(e1.grid2lin(p1) == 35);
    }

    TEST_CASE("create embedding grid from xtensor", "[embedding]") {
        xt::xarray<hg::index_t> shape = {10, 5, 2};
        hg::embedding_grid_3d e1(shape);
        REQUIRE(e1.size() == 100);
        REQUIRE(e1.dimension() == 3);

#ifndef _MSC_VER // vs 2017 ICE
        hg::point_3d_i p1 = {{3, 2, 1}};
        auto p1t = e1.lin2grid(35);
        REQUIRE(std::equal(p1.begin(), p1.end(), p1t.begin()));
        REQUIRE(e1.grid2lin(p1) == 35);
#endif // !_MSC_VER
    }

    TEST_CASE("grid to linear coordinates", "[embedding]") {
        xt::xarray<hg::index_t> shape = {10, 5, 2};
        hg::embedding_grid_3d e1(shape);

        xt::xarray<hg::index_t> coord = {{0, 0, 0},
                                         {0, 0, 1},
                                         {0, 0, 2},
                                         {3, 2, 1}};
        auto linCoord = e1.grid2lin(coord);
        REQUIRE(linCoord.shape().size() == 1);
        REQUIRE(linCoord.shape()[0] == 4);

        REQUIRE(linCoord(0) == 0);
        REQUIRE(linCoord(1) == 1);
        REQUIRE(linCoord(2) == 2);
        REQUIRE(linCoord(3) == 35);
    }

    TEST_CASE("linear coordinates to grid", "[embedding]") {
        xt::xarray<hg::index_t> shape = {5, 10};
        hg::embedding_grid_2d e1(shape);

        xt::xarray<hg::index_t> coordLin = {{0,  1,  2,  3},
                                            {22, 42, 43, 44}};
        xt::xarray<hg::index_t> coords = {{{0, 0}, {0, 1}, {0, 2}, {0, 3}},
                                          {{2, 2}, {4, 2}, {4, 3}, {4, 4}}};

        auto res = e1.lin2grid(coordLin);
        REQUIRE((res == coords));
    }

    TEST_CASE("contains", "[embedding]") {
        xt::xarray<hg::index_t> shape = {5, 10};
        hg::embedding_grid_2d e1(shape);

        xt::xarray<hg::index_t> coords{{{0, 0}, {3, 8}, {-1, 2}},
                                       {{2, 4}, {5, 5}, {43, 44}}};

        xt::xarray<bool> ref{{true, true,  false},
                             {true, false, false}};

        auto res = e1.contains(coords);
        REQUIRE((res == ref));
    }
}
