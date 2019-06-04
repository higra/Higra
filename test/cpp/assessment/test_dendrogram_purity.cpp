/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/assessment/dendrogram_purity.hpp"
#include "../test_utils.hpp"

using namespace hg;

namespace assessment_dendrogram_purity {

    TEST_CASE("dendrogram purity", "[dendrogram purity]") {
        SECTION("binary"){
            tree t(array_1d <index_t>{5,5,6,7,7,6,8,8,8});
            array_1d<int> labels{1,1,0,1,0};

            auto p = dendrogram_purity(t, labels);

            REQUIRE(almost_equal(p, 0.65));
        }
        SECTION("non binary"){
            tree t(array_1d <index_t>{5,5,5,6,6,7,7,7});
            array_1d<int> labels{1,1,0,1,0};

            auto p = dendrogram_purity(t, labels);

            REQUIRE(almost_equal(p, 0.5666666666666667));
        }
    }
}
