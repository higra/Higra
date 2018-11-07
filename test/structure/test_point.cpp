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
#include "higra/structure/point.hpp"


BOOST_AUTO_TEST_SUITE(point);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(point2dCreateAndArith) {
        point_2d_f p1{{1.5, 2.3}};
        point_2d_f p2{{2, 1}};

        point_2d_f ref{{2.5, 2.3}};


        BOOST_CHECK(xt::allclose(ref, p1 + p2 - 1));
    }


    BOOST_AUTO_TEST_CASE(point2iCreateAndArith) {
        point_2d_i p1{{4, 2}};
        point_2d_i p2{{2, 3}};

        point_2d_i ref{{5, 4}};


        BOOST_CHECK(xt::allclose(ref, p1 + p2 - 1));
    }

BOOST_AUTO_TEST_SUITE_END();