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
        std::cout << res << std::endl;
        REQUIRE((ref == res));
    }
}