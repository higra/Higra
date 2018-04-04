//
// Created by user on 4/4/18.
//
#include <boost/test/unit_test.hpp>
#include "point.hpp"


BOOST_AUTO_TEST_SUITE(point);

using namespace hg;
using namespace std;


BOOST_AUTO_TEST_CASE(point2dCreateAndArith) {
        point2d p1{ 1.5, 2.3 };
        point2d p2{ 2, 1 };

        point2d ref{ 2.5, 2.3 };


        BOOST_CHECK(xt::allclose(ref, p1 + p2 - 1));
}


BOOST_AUTO_TEST_CASE(point2iCreateAndArith) {
        point2i p1{ 4, 2 };
        point2i p2{ 2, 3 };

        point2i ref{ 5, 4 };


        BOOST_CHECK(xt::allclose(ref, p1 + p2 - 1));
}

BOOST_AUTO_TEST_SUITE_END();