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
#include <limits>
#include <utility>
#include <vector>
#include <type_traits>
#include "higra/accumulator/at_accumulator.hpp"

namespace test_at_accumulator {

    using namespace hg;

    TEST_CASE("test at_accumulator", "at_accumulator") {

        array_1d<index_t> indices{1, 1, -1, 2, 0};
        array_1d<int> weights{1, 2, 3, 4, 5};

        auto res = accumulate_at(indices, weights, accumulator_sum());
        array_nd<double> expected_res{5, 3, 4};
        REQUIRE((res == expected_res));

        array_2d<int> weights_vec{{1, 6},
                                  {2, 7},
                                  {3, 8},
                                  {4, 9},
                                  {5, 10}};
        auto res_vec = accumulate_at(indices, weights_vec, accumulator_sum());
        array_nd<double> expected_res_vec{
                {5, 10},
                {3, 13},
                {4, 9}};
        REQUIRE((res_vec == expected_res_vec));
    }

    TEST_CASE("test at_accumulator indices below -1", "at_accumulator") {
        array_1d<int> weights{1, 2, 3};
        array_2d<int> weights_vec{{1, 6},
                                  {2, 7},
                                  {3, 8}};
        for (index_t bad: {(index_t) -2, (index_t) -1000000, std::numeric_limits<index_t>::lowest()}) {
            array_1d<index_t> indices{0, bad, 1};
            REQUIRE_THROWS(accumulate_at(indices, weights, accumulator_sum()));
            REQUIRE_THROWS(accumulate_at(indices, weights_vec, accumulator_sum()));
        }
        // other index types are converted to index_t first
        array_1d<int> indices_int{0, -2, 1};
        REQUIRE_THROWS(accumulate_at(indices_int, weights, accumulator_sum()));
    }

    template<typename array_t, typename V>
    static void check_row(const array_t &a, index_t i, V v) {
        index_t row = (index_t) (a.size() / a.shape()[0]);
        for (index_t j = i * row; j < (i + 1) * row; j++) {
            REQUIRE(a.data()[j] == v);
        }
    }

    template<typename T>
    static void check_unreached_rows(const array_nd<T> &weights) {
        // rows 1 and 2 are not reached by any index
        array_1d<index_t> indices{0, 3, 3};
        for (index_t i: {1, 2}) {
            check_row(accumulate_at(indices, weights, accumulator_sum()), i, 0);
            check_row(accumulate_at(indices, weights, accumulator_mean()), i, 0);
            check_row(accumulate_at(indices, weights, accumulator_prod()), i, 1);
            check_row(accumulate_at(indices, weights, accumulator_min()), i, (std::numeric_limits<T>::max)());
            check_row(accumulate_at(indices, weights, accumulator_max()), i, std::numeric_limits<T>::lowest());
            check_row(accumulate_at(indices, weights, accumulator_first()), i, 0);
            check_row(accumulate_at(indices, weights, accumulator_last()), i, 0);
        }
        REQUIRE((accumulate_at(indices, weights, accumulator_counter()) == array_1d<index_t>{1, 0, 0, 2}));
        // weights increase with the index of the entry
        REQUIRE((accumulate_at(indices, weights, accumulator_argmin()) == array_1d<index_t>{0, -1, -1, 0}));
        REQUIRE((accumulate_at(indices, weights, accumulator_argmax()) == array_1d<index_t>{0, -1, -1, 1}));
    }

    TEST_CASE("test at_accumulator unreached rows", "at_accumulator") {
        check_unreached_rows(array_nd<double>{1, 2, 3});
        check_unreached_rows(array_nd<int>{{1, 6}, {2, 7}, {3, 8}});
    }

    TEST_CASE("test at_accumulator index results with small integer types", "at_accumulator") {
        array_1d<index_t> indices = hg::zeros<index_t>({300});
        array_1d<unsigned char> weights = hg::ones<unsigned char>({300});
        weights(299) = 0;
        weights(298) = 2;
        auto argmin = accumulate_at(indices, weights, accumulator_argmin());
        auto argmax = accumulate_at(indices, weights, accumulator_argmax());
        auto counter = accumulate_at(indices, weights, accumulator_counter());
        static_assert(std::is_same<typename decltype(argmin)::value_type, index_t>::value);
        static_assert(std::is_same<typename decltype(counter)::value_type, index_t>::value);
        REQUIRE((argmin == array_1d<index_t>{299}));
        REQUIRE((argmax == array_1d<index_t>{298}));
        REQUIRE((counter == array_1d<index_t>{300}));
    }

    TEST_CASE("test at_accumulator integer mean with many values", "at_accumulator") {
        array_1d<index_t> indices = hg::zeros<index_t>({256});
        array_1d<unsigned char> weights = hg::zeros<unsigned char>({256});
        REQUIRE((accumulate_at(indices, weights, accumulator_mean()) == array_1d<unsigned char>{0}));
        array_1d<short> weights2 = hg::zeros<short>({256});
        weights2(0) = 511;
        REQUIRE((accumulate_at(indices, weights2, accumulator_mean()) == array_1d<short>{1}));
    }

    TEST_CASE("test at_accumulator zero-element weights with large dimensions", "at_accumulator") {
        const index_t big = (index_t) 1 << 32;
        array_1d<index_t> one_index{0};
        array_1d<index_t> no_index = hg::zeros<index_t>({0});
        std::vector<std::pair<array_1d<index_t>, std::vector<index_t>>> cases{
                {one_index, {1, big, big, 0}},
                {one_index, {1, 0, big, big}},
                {no_index,  {0, big, big}},
        };
        for (auto &c: cases) {
            auto weights = array<double>(array_shape::from(c.second), 0);
            for (auto acc_res: {accumulate_at(c.first, weights, accumulator_sum()),
                                accumulate_at(c.first, weights, accumulator_first())}) {
                auto expected = c.second;
                expected[0] = (index_t) c.first.size();
                REQUIRE(acc_res.size() == 0);
                REQUIRE(std::equal(acc_res.shape().begin(), acc_res.shape().end(), expected.begin(), expected.end()));
            }
            auto argmin = accumulate_at(c.first, weights, accumulator_argmin());
            REQUIRE((index_t) argmin.size() == (index_t) c.first.size());
            if (argmin.size() > 0) {
                // one value (with zero components) is accumulated in row 0
                REQUIRE(argmin(0) == 0);
            }
        }
    }
}
