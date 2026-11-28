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
#include <sstream>
#include <type_traits>
#include <vector>

namespace test_point {

    using namespace hg;
    using namespace std;

    static_assert(std::is_trivially_copyable<point_3d_i>::value, "points must be trivially copyable");
    static_assert(sizeof(point_3d_i) == 3 * sizeof(index_t), "points must be stored inline");
    static_assert(point_4d_f::size() == 4, "size is a compile-time constant");

    TEST_CASE("point 2d float create and arithmetic", "[point]") {
        point_2d_f p1{{1.5, 2.3}};
        point_2d_f p2{{2, 1}};
        point_2d_f one{{1, 1}};

        auto r = p1 + p2 - one;
        REQUIRE(r(0) == Catch::Approx(2.5));
        REQUIRE(r(1) == Catch::Approx(2.3));
    }

    TEST_CASE("point 2d integer create and arithmetic", "[point]") {
        point_2d_i p1{{4, 2}};
        point_2d_i p2{{2, 3}};
        point_2d_i one{{1, 1}};

        point_2d_i ref{{5, 4}};

        REQUIRE((p1 + p2 - one == ref));
        REQUIRE((p1 - p2 == point_2d_i{{2, -1}}));
        REQUIRE((p1 != p2));
    }

    TEST_CASE("point initialization and accessors", "[point]") {
        point_1d_i p1{15};
        REQUIRE(p1(0) == 15);
        REQUIRE(p1.size() == 1);

        point_2d_i flat{0, -1}; // brace elision
        point_2d_i nested{{0, -1}};
        REQUIRE((flat == nested));

        point_3d_i z{};
        REQUIRE(z(0) == 0);
        REQUIRE(z(1) == 0);
        REQUIRE(z(2) == 0);

        point_3d_i p{{3, 2, 1}};
        REQUIRE(p[0] == 3);
        REQUIRE(p(1) == 2);
        p(1) = 7;
        p[2] = 8;
        REQUIRE((p == point_3d_i{{3, 7, 8}}));
        REQUIRE(p.data() == &p(0));

        std::vector<index_t> values(p.begin(), p.end());
        REQUIRE((values == std::vector<index_t>{3, 7, 8}));
        const auto &cp = p;
        REQUIRE(std::equal(cp.cbegin(), cp.cend(), values.begin()));

        p.fill(-4);
        REQUIRE((p == point_3d_i{{-4, -4, -4}}));

        std::vector<point_2d_i> list{{{-1, 0}},
                                     {{0,  -1}},
                                     {0,   1},
                                     {1,   0}};
        REQUIRE(list.size() == 4);
        REQUIRE((list[2] == point_2d_i{{0, 1}}));
    }

    TEST_CASE("point elementwise min and max", "[point]") {
        point_3d_i a{{1, -5, 3}};
        point_3d_i b{{2, -6, 3}};
        REQUIRE((hg::min(a, b) == point_3d_i{{1, -6, 3}}));
        REQUIRE((hg::max(a, b) == point_3d_i{{2, -5, 3}}));
    }

    TEST_CASE("point conversion", "[point]") {
        point_2d_i pi{{3, -2}};
        point_2d_f pf = pi;
        REQUIRE(pf(0) == 3.0);
        REQUIRE(pf(1) == -2.0);
        pf(1) += 0.5;
        REQUIRE(pf(1) == -1.5);

        auto a = hg::to_array(pi);
        static_assert(std::is_same<decltype(a), hg::array<index_t>>::value, "to_array returns an hg::array");
        REQUIRE(a.dimension() == 1);
        REQUIRE(a.size() == 2);
        REQUIRE(a(0) == 3);
        REQUIRE(a(1) == -2);

        std::ostringstream os;
        os << pi;
        REQUIRE(os.str() == "{3, -2}");
    }
}
