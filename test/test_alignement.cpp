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
#include "higra/algo/alignement.hpp"
#include "test_utils.hpp"


using namespace hg;

BOOST_AUTO_TEST_SUITE(alignement_test);

    BOOST_AUTO_TEST_CASE(test_project_fine_to_coarse_labelisation) {

        array_1d<int> fine_labels{0, 1, 2, 3, 4, 2, 3, 4, 2};
        array_1d<int> coarse_labels{0, 1, 1, 0, 2, 2, 0, 2, 2};

        auto map = project_fine_to_coarse_labelisation(fine_labels, coarse_labels);

        array_1d<int> ref_map{0, 1, 2, 0, 2};
        BOOST_CHECK(ref_map == map);
    }

BOOST_AUTO_TEST_SUITE_END();