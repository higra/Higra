/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/structure/details/light_axis_view.hpp"
#include "higra/structure/array.hpp"
#include "xtensor/views/xview.hpp"
#include "xtensor/views/xstrided_view.hpp"
#include "../../test_utils.hpp"

namespace light_axis_view {

    using namespace hg;
    using namespace std;
    using namespace xt;

    template<typename T>
    void print_it(T &&it) {
        std::cout << "{";
        for (auto v = it.begin(); v != it.end(); v++) {
            std::cout << *v << " ";
        }
        std::cout << "}" << std::endl;
    }


    template<typename T1, typename T2>
    bool compare(T1 &&v1, T2 &&v2) {
        auto b1 = v1.begin();
        auto b2 = v2.begin();
        auto e1 = v1.end();
        auto e2 = v2.end();

        for (; b1 != e1 && b2 != e2; b1++, b2++) {
            if (*b1 != *b2)
                return false;
        }

        if (b1 != e1 || b2 != e2)
            return false;
        return true;
    };

    TEST_CASE("view on array", "[light_axis_view]") {
        array_nd<int> a{
                {
                        {1, 2, 3},
                        {4,  5,  6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                }
        };
        auto v = make_light_axis_view<true>(a);
        auto v1 = view(a, 0);
        REQUIRE(compare(v, v1));

        v.set_position(1);
        auto v2 = view(a, 1);
        REQUIRE(compare(v, v2));
    }


    /* BOOST_AUTO_TEST_CASE(view_on_view_simple) {
 
         array_nd<int> a{{{1, 2, 3},
                                 {4,  5,  6}},
                         {{7, 8, 9},
                                 {10, 11, 12}}};
         auto va = view(a, 1);
         auto v = make_light_axis_view<true>(va);
         std::array<int, 3> v1{7, 8, 9};
         REQUIRE(compare(v, v1));
         std::cout << va << std::endl;
         print_it(v);
         v.set_position(1);
         print_it(v);
         std::array<int, 3> v2{10, 11, 12};
         REQUIRE(compare(v, v2));
     }*/

    TEST_CASE("view on xtensor view", "[light_axis_view]") {
        array_nd<int> a{
                {
                        {1, 2, 3},
                        {4,  5,  6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                }
        };
        auto va = view(a, xt::all(), xt::range(1, 2), xt::range(0, 3, 2));
        auto v = make_light_axis_view<true>(va);
        std::array<int, 2> v1{4, 6};
        REQUIRE(compare(v, v1));

        v.set_position(1);
        std::array<int, 2> v2{10, 12};
        REQUIRE(compare(v, v2));
    }

    TEST_CASE("view on xtensor strided view", "[light_axis_view]") {
        array_nd<int> a{
                {
                        {1, 2, 3},
                        {4,  5,  6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                }
        };
        auto va = strided_view(a, {1});
        auto v = make_light_axis_view<true>(va);
        std::array<int, 3> v1{7, 8, 9};
        REQUIRE(compare(v, v1));

        v.set_position(1);
        std::array<int, 3> v2{10, 11, 12};
        REQUIRE(compare(v, v2));
    }

    TEST_CASE("view on xtensor strided view complex", "[light_axis_view]") {
        array_nd<int> a{
                {
                        {1, 2, 3},
                        {4,  5,  6}
                },
                {
                        {7, 8, 9},
                        {10, 11, 12}
                }
        };
        auto va = strided_view(a, {xt::all(), xt::range(1, 2), xt::range(0, 3, 2)});
        auto v = make_light_axis_view<true>(va);
        std::array<int, 2> v1{4, 6};
        REQUIRE(compare(v, v1));

        v.set_position(1);
        std::array<int, 2> v2{10, 12};
        REQUIRE(compare(v, v2));
    }

}