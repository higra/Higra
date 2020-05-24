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
#include "higra/algo/tree_monotonic_regression.hpp"


using namespace hg;

namespace test_tree_monotonic_regression {

    TEST_CASE("tree_monotonic_regression trivial", "[tree_monotonic_regression]") {
        hg::tree tree(xt::xarray<index_t>{7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11});
        array_1d<double> altitudes{0, 1, 0, 2, 0, 0, 0, 2, 3, 0, 5, 10};
        SECTION("max") {
            auto res = tree_monotonic_regression(tree, altitudes, "max");
            REQUIRE((altitudes == res));
        }SECTION("min") {
            auto res = tree_monotonic_regression(tree, altitudes, "min");
            REQUIRE((altitudes == res));
        }SECTION("least_square") {
            auto res = tree_monotonic_regression(tree, altitudes, "least_square");
            REQUIRE((altitudes == res));
        }SECTION("least_square_weighted") {
            auto res = tree_monotonic_regression(tree, altitudes, xt::arange<double>(1, 13), "least_square");
            REQUIRE((altitudes == res));
        }SECTION("not_suppported") {
            REQUIRE_THROWS(tree_monotonic_regression(tree, altitudes, "truc"));
        }
    }

    TEST_CASE("tree_monotonic_regression max", "[tree_monotonic_regression]") {
        hg::tree tree(xt::xarray<index_t>{7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11});
        array_1d<double> altitudes{0, 3, 0, 2, 0, 0, 0, 2, 3, 0, 5, 4};

        array_1d<double> ref{0, 3, 0, 2, 0, 0, 0, 3, 3, 0, 5, 5};
        auto res = tree_monotonic_regression(tree, altitudes, "max");
        REQUIRE((ref == res));
    }

    TEST_CASE("tree_monotonic_regression min", "[tree_monotonic_regression]") {
        hg::tree tree(xt::xarray<index_t>{7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11});
        array_1d<double> altitudes{0, 3, 0, 2, 0, 0, 0, 2, 3, 0, 5, 4};

        array_1d<double> ref{0, 2, 0, 2, 0, 0, 0, 2, 3, 0, 4, 4};
        auto res = tree_monotonic_regression(tree, altitudes, "min");
        REQUIRE((ref == res));
    }

    TEST_CASE("tree_monotonic_regression least square no weights", "[tree_monotonic_regression]") {
        hg::tree tree(xt::xarray<index_t>{5, 5, 6, 6, 7, 7, 7, 7});
        array_1d<double> altitudes{13, 14, 6, 8, 7, 11, 5, 10};

        array_1d<double> ref{12, 12, 6, 6.5, 7, 12, 6.5, 12};
        auto res = tree_monotonic_regression(tree, altitudes, "least_square");
        REQUIRE(xt::allclose(res, ref));
    }

    TEST_CASE("tree_monotonic_regression least square weighted", "[tree_monotonic_regression]") {
        hg::tree tree(xt::xarray<index_t>{5, 5, 6, 6, 7, 7, 7, 7});
        array_1d<double> altitudes{13, 14, 6, 8, 7, 11, 5, 10};
        array_1d<double> weights{1, 1, 1, 1, 1, 1, 2, 1};

        array_1d<double> ref{12, 12, 6, 6, 7, 12, 6, 12};
        auto res = tree_monotonic_regression(tree, altitudes, weights, "least_square");
        REQUIRE(xt::allclose(res, ref));
    }
}