/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/structure/details/iterators.hpp"
#include "../../test_utils.hpp"


using namespace hg;

TEST_CASE("counting iterator", "[iterator]") {
    counting_iterator<> c1(0, 2);
    counting_iterator<> c2(10, 2);

    REQUIRE((*c1 == 0));
    REQUIRE((*c2 == 10));
    REQUIRE((c1[0] == 0));
    REQUIRE((c1[1] == 2));
    REQUIRE((c1[3] == 6));
    REQUIRE((c2[-2] == 6));
    REQUIRE((c1 <= c2));
    REQUIRE((c1 < c2));
    REQUIRE((c2 > c1));
    REQUIRE((c2 >= c1));
    REQUIRE((c1 + 5 == c2));
    auto c3 = ++c1;
    REQUIRE((*c3 == 2));
    REQUIRE((*c1 == 2));
    REQUIRE((*(c1+3) == 8));
    REQUIRE(((c2-c1) == 4));
    c3 = --c2;
    REQUIRE((*c3 == 8));
    REQUIRE((c2[-1] == 6));
    c3 = c1--;
    REQUIRE((*c1 == 0));
    REQUIRE((*c3 == 2));
    c3 = c1++;
    REQUIRE((*c1 == 2));
    REQUIRE((*c3 == 0));
}