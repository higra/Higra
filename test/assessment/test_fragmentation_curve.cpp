/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>
#include "higra/assessment/fragmentation_curve.hpp"
#include "higra/image/graph_image.hpp"
#include "../test_utils.hpp"

using namespace hg;

BOOST_AUTO_TEST_SUITE(test_fragmentation_curve);

    BOOST_AUTO_TEST_CASE(test_assess_fragmentation_curve_BCE_optimal_cut) {
        tree t(array_1d<index_t>{8, 8, 9, 9, 10, 10, 11, 13, 12, 12, 11, 13, 14, 14, 14});
        array_1d<char> ground_truth{0, 0, 1, 1, 1, 2, 2, 2};

        auto res = assess_fragmentation_curve_BCE_optimal_cut(t, ground_truth, 200);
        auto &res_scores = res.scores;
        auto &res_k = res.k;

        array_1d<double> ref_scores{2.75, 4.5, 2 + 4.0 / 3 + 2.5, 2 + 4.0 / 3 + 2, 2 + 4.0 / 3 + 4.0 / 3,
                                    2 + 4.0 / 3 + 4.0 / 3, 4, 3};
        array_1d<size_t> ref_k{1, 2, 3, 4, 5, 6, 7, 8};

        BOOST_CHECK(res_scores == (ref_scores / num_leaves(t)));
        BOOST_CHECK(res_k == ref_k);
    }


BOOST_AUTO_TEST_SUITE_END();