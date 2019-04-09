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
#include "higra/structure/point.hpp"

namespace point {

    using namespace hg;
    using namespace std;

    TEST_CASE("point 2d float create and arithmetic", "[point]") {
        point_2d_f p1{{1.5, 2.3}};
        point_2d_f p2{{2, 1}};

        point_2d_f ref{{2.5, 2.3}};

        REQUIRE(xt::allclose(ref, p1 + p2 - 1));
    }

    TEST_CASE("point 2d integer create and arithmetic", "[point]") {
        point_2d_i p1{{4, 2}};
        point_2d_i p2{{2, 3}};

        point_2d_i ref{{5, 4}};

        REQUIRE(xt::allclose(ref, p1 + p2 - 1));
    }
}