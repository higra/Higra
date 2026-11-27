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
#include "higra/accumulator/at_accumulator.hpp"

using namespace hg;

namespace accumulator {

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
}
