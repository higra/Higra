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
#include <cmath>
#include <sstream>
#include "higra/algo/linear_energy_optimization.hpp"

using namespace hg;
using namespace std;
using namespace hg::linear_energy_optimization_internal;

using lep_t = piecewise_linear_energy_function_piece<double>;
using lef_t = piecewise_linear_energy_function<double>;

ostream &operator<<(ostream &os, const lep_t &rhs) {
    os << "((" << rhs.origin_x() << ", " << rhs.origin_y() << "), " << rhs.slope() << ")";
    return os;
}

ostream &operator<<(ostream &os, const lef_t &rhs) {
    os << "{";
    for (const auto &lp: rhs) {
        os << lp << ", ";
    }
    os << "}";
    return os;
}

namespace Catch {

    template<>
    struct StringMaker<lep_t> {
        static std::string convert(lep_t const &rhs) {
            std::ostringstream strs;
            strs << "((" << rhs.origin_x() << ", " << rhs.origin_y() << "), " << rhs.slope() << ")";
            return strs.str();
        }
    };

}

namespace test_linear_energy_function_optimization {

    TEST_CASE("test piecewise_linear_energy_function_piece", "[linear_energy_function_optimization]") {

        lep_t p1(1, 2, 3);
        lep_t p2(2, 1, 3);
        lep_t p3(1, 2, 3);

        REQUIRE(p1 != p2);
        REQUIRE(p1 == p3);

        REQUIRE(p1(1) == 2);
        REQUIRE(p1(0) == -1);
    }

    TEST_CASE("test piecewise_linear_energy_function sum", "[linear_energy_function_optimization]") {

        SECTION("edge case empty") {
            lep_t p1(1, 2, 3);
            lef_t f1(p1);
            lef_t empty;

            REQUIRE(empty.sum(empty) == empty);
            REQUIRE(f1.sum(empty) == f1);
            REQUIRE(empty.sum(f1) == f1);
        }SECTION("simple + simple") {
            lef_t f1({0, 0, 1});
            lef_t f2({0, 1, 1});

            lef_t r({0, 1, 2});
            REQUIRE(f1.sum(f2) == r);
            REQUIRE(f2.sum(f1) == r);
        }SECTION("compound + compound") {
            lef_t f1({{0, 0, 2},
                      {1, 2, 1}});
            lef_t f2({{0,   0,   1},
                      {0.5, 0.5, 0.5},
                      {2.5, 1.5, 0.1}});

            lef_t r({{0,   0,    3},
                     {0.5, 1.5,  2.5},
                     {1,   2.75, 1.5},
                     {2.5, 5,    1.1}});
            REQUIRE(f1.sum(f2) == r);
            REQUIRE(f2.sum(f1) == r);
        }

        SECTION("compound + compound with max pieces") {
            lef_t f1({{0, 0, 2},
                      {1, 2, 1}});
            lef_t f2({{0,   0,   1},
                      {0.5, 0.5, 0.5},
                      {2.5, 1.5, 0.1}});

            lef_t r({{0.,  0.25, 2.5},
                     {1,   2.75, 1.5},
                     {2.5, 5,    1.1}});
            REQUIRE(f1.sum(f2, 3) == r);
            REQUIRE(f2.sum(f1, 3) == r);
        }
    }

    TEST_CASE("test piecewise_linear_energy_function infimum", "[linear_energy_function_optimization]") {

        SECTION("simple no intersection") {
            lef_t f({0, 0, 1});
            lep_t p(0, 1, 1);

            lef_t r(f);
            REQUIRE(isinf(f.infimum(p)));
            REQUIRE(f == r);
        }SECTION("simple intersection") {
            lef_t f({0, 1, 1});
            lep_t p(0, 2, 0.5);

            lef_t r({{0, 1, 1},
                     {2, 3, 0.5}});
            REQUIRE(f.infimum(p) == 2);
            REQUIRE(f == r);
        }SECTION("compound intersection") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 6, 1);

            lef_t r({{0, 0, 5},
                     {1, 5, 3},
                     {2, 8, 1}});
            REQUIRE(f.infimum(p) == 2);
            REQUIRE(f == r);
        }SECTION("parallel edge case 1") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 20, 2);

            lef_t r({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            REQUIRE(isinf(f.infimum(p)));
            REQUIRE(f == r);
        }SECTION("parallel edge case 2") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 5, 2);

            lef_t r({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            REQUIRE(f.infimum(p) == 3);
            REQUIRE(f == r);
        }
        SECTION("parallel edge case 3") {
            lef_t f({{0, 0,  5},
                     {1, 5,  3},
                     {3, 11, 2}});
            lep_t p(0, 1, 2);

            lef_t r({{0,      0,      5},
                     {1. / 3, 5. / 3, 2}});
            REQUIRE(f.infimum(p) == Approx(1.0 / 3));
            REQUIRE(f == r);
        }
    }
}