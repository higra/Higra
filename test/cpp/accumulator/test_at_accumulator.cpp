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
}