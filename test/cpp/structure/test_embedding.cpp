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

    TEST_CASE("embedding array methods with extra leading dimensions", "[embedding]") {
        hg::embedding_grid_2d e1{5, 10};

        // (2, 3, 2) coordinates -> (2, 3) results
        xt::xarray<hg::index_t> coords{{{0, 0}, {3, 8}, {-1, 2}},
                                       {{2, 4}, {5, 5}, {4, 9}}};
        auto lin = e1.grid2lin(coords);
        static_assert(std::is_same<typename decltype(lin)::value_type, std::size_t>::value,
                      "grid2lin returns unsigned linear coordinates");
        REQUIRE(lin.shape().size() == 2);
        REQUIRE(lin.shape()[0] == 2);
        REQUIRE(lin.shape()[1] == 3);
        REQUIRE(lin(0, 0) == 0);
        REQUIRE(lin(0, 1) == 38);
        REQUIRE(lin(1, 0) == 24);
        REQUIRE(lin(1, 2) == 49);

        auto inside = e1.contains(coords);
        xt::xarray<bool> inside_ref{{true, true,  false},
                                    {true, false, true}};
        REQUIRE((inside == inside_ref));

        // (2, 3) linear coordinates -> (2, 3, 2) coordinates
        xt::xarray<hg::index_t> lin_ref{{0,  38, 2},
                                        {24, 49, 13}};
        auto grid = e1.lin2grid(lin_ref);
        xt::xarray<hg::index_t> grid_ref{{{0, 0}, {3, 8}, {0, 2}},
                                         {{2, 4}, {4, 9}, {1, 3}}};
        REQUIRE((grid == grid_ref));

        // 0-d linear coordinate -> (2,) coordinates; (2,) coordinates -> 0-d linear coordinate
        xt::xarray<int> scalar = 13;
        auto g0 = e1.lin2grid(scalar);
        REQUIRE(g0.dimension() == 1);
        REQUIRE((g0 == xt::xarray<hg::index_t>{1, 3}));
        xt::xarray<int> point_coords{1, 3};
        auto l0 = e1.grid2lin(point_coords);
        REQUIRE(l0.dimension() == 0);
        REQUIRE(l0() == 13);
        auto c0 = e1.contains(point_coords);
        REQUIRE(c0.dimension() == 0);
        REQUIRE(c0());

        // lazy and strided inputs
        xt::xarray<hg::index_t> big{{0, 0, 9, 3, 8, 9, 4, 9, 9}};
        auto strided = xt::view(xt::reshape_view(big, {3, 3}), xt::all(), xt::range(0, 2));
        auto lin_strided = e1.grid2lin(strided);
        REQUIRE((lin_strided == xt::xarray<std::size_t>{0, 38, 49}));
        auto lin_lazy = e1.grid2lin(coords + 0);
        REQUIRE((lin_lazy == lin));
        auto grid_lazy = e1.lin2grid(xt::arange<hg::index_t>(3) * 10);
        REQUIRE((grid_lazy == xt::xarray<hg::index_t>{{0, 0}, {1, 0}, {2, 0}}));

        // empty leading dimension
        xt::xarray<hg::index_t> none = xt::zeros<hg::index_t>({0, 2});
        REQUIRE(e1.grid2lin(none).size() == 0);
        REQUIRE(e1.contains(none).shape()[0] == 0);
        REQUIRE(e1.lin2grid(xt::xarray<hg::index_t>::from_shape({0})).shape()[1] == 2);

        // the size of the last axis must be the dimension (or 1 for grid2lin, broadcast to every coordinate)
        xt::xarray<hg::index_t> bad = xt::zeros<hg::index_t>({3, 3});
        REQUIRE_THROWS(e1.grid2lin(bad));
        REQUIRE_THROWS(e1.contains(bad));
        xt::xarray<hg::index_t> ones = xt::ones<hg::index_t>({3, 1});
        REQUIRE((e1.grid2lin(ones) == xt::xarray<std::size_t>{11, 11, 11}));

        // shape from an array
        hg::embedding_grid_3d e3(xt::xarray<int>{2, 3, 4});
        REQUIRE((e3.shape() == hg::point_3d_i{{2, 3, 4}}));
        REQUIRE(e3.size() == 24);
        REQUIRE_THROWS(hg::embedding_grid_2d(xt::xarray<int>{2, 3, 4}));
        REQUIRE_THROWS(hg::embedding_grid_2d(xt::xarray<int>{2, 0}));
    }
}
