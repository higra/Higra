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
#include <utility>
#include <vector>

#ifdef HG_USE_TBB
    #include <oneapi/tbb/task_arena.h>
#endif

namespace test_sorting {

    using namespace hg;

    struct move_only_value {
        int key;
        index_t index;

        move_only_value(int key_, index_t index_) : key(key_), index(index_) {}
        move_only_value(const move_only_value &) = delete;
        move_only_value &operator=(const move_only_value &) = delete;
        move_only_value(move_only_value &&) = default;
        move_only_value &operator=(move_only_value &&) = default;
    };

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

    TEST_CASE("stable arg sort large array scalar", "[sorting]") {
        constexpr index_t size = 8192;
        array_1d<int> a = xt::arange<int>(size) % 23;
        auto indices = hg::stable_arg_sort(a);

        for (index_t i = 1; i < size; ++i) {
            REQUIRE(a(indices(i - 1)) <= a(indices(i)));
            if (a(indices(i - 1)) == a(indices(i))) {
                REQUIRE(indices(i - 1) < indices(i));
            }
        }
    }

    TEST_CASE("stable sort move-only values", "[sorting]") {
        std::vector<move_only_value> values;
        for (index_t index = 0; index < 8192; ++index) {
            values.emplace_back(index % 23, index);
        }

        hg::stable_sort(values.begin(), values.end(),
                [](const auto &left, const auto &right) { return left.key < right.key; });

        for (index_t index = 1; index < values.size(); ++index) {
            REQUIRE(values[index - 1].key <= values[index].key);
            if (values[index - 1].key == values[index].key) {
                REQUIRE(values[index - 1].index < values[index].index);
            }
        }
    }
}