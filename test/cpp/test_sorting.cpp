/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/sorting.hpp"
#include "test_utils.hpp"

namespace test_sorting {

    using namespace hg;

    TEST_CASE("sort array scalar", "[sorting]") {
        array_1d<int> a1 = {5, 2, 1, 4, 9};
        hg::sort(a1);
        array_1d<int> ref = {1, 2, 4, 5, 9};
        REQUIRE((a1 == ref));

        array_1d<int> a2 = {5, 2, 1, 4, 9};
        hg::sort(a2, std::greater<int>());
        array_1d<int> ref2 = {9, 5, 4, 2, 1};
        REQUIRE((a2 == ref2));
    }

    TEST_CASE("arg sort array scalar", "[sorting]") {
        array_1d<int> a1 = {5, 2, 1, 4, 9};
        auto i1 = hg::arg_sort(a1);
        array_1d<int> ref = {2, 1, 3, 0 , 4};
        REQUIRE((i1 == ref));

        auto i2 = hg::arg_sort(a1, std::greater<int>());
        array_1d<int> ref2 = {4, 0, 3, 1, 2};
        REQUIRE((i2 == ref2));
    }

    TEST_CASE("stable arg sort array scalar", "[sorting]") {
        array_1d<int> a1 = {2, 2, 2, 2, 1, 1, 1, 1};
        auto i1 = hg::stable_arg_sort(a1);
        array_1d<int> ref = {4, 5, 6, 7, 0, 1, 2, 3};
        REQUIRE((i1 == ref));

        auto i2 = hg::stable_arg_sort(a1, std::greater<int>());
        array_1d<int> ref2 = {0, 1, 2, 3, 4, 5, 6, 7};
        REQUIRE((i2 == ref2));
    }

    TEST_CASE("sort array lexicographic", "[sorting]") {
        array_2d<int> a1 = {{2, 2, 1, 1, 3},
                            {2, 1, 1, 2, 0}};
        auto i1 = hg::arg_sort(xt::transpose(a1));
        array_1d<int> ref = {2, 3, 1, 0, 4};
        REQUIRE((i1 == ref));

        auto i2 = hg::arg_sort(xt::transpose(a1), std::greater<int>());
        array_1d<int> ref2 = {4, 0, 1, 3, 2};
        REQUIRE((i2 == ref2));
    }

    TEST_CASE("stable sort array lexicographic", "[sorting]") {
        array_2d<int> a1 = {{2, 2, 1, 1, 3},
                            {2, 2, 2, 1, 0}};
        auto i1 = hg::arg_sort(xt::transpose(a1));
        array_1d<int> ref = {3, 2, 0, 1, 4};
        REQUIRE((i1 == ref));

        auto i2 = hg::arg_sort(xt::transpose(a1), std::greater<int>());
        array_1d<int> ref2 = {4, 0, 1, 2, 3};
        REQUIRE((i2 == ref2));
    }
}