#define HG_DEBUG

#define BOOST_TEST_MODULE higra_test

#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/structure/array.hpp"
#include "xtensor/xfixed.hpp"
#include "higra/utils.hpp"
#include "xtensor/xview.hpp"

BOOST_AUTO_TEST_SUITE(xp);


    BOOST_AUTO_TEST_CASE(xpp) {

        /*xt::xarray<int> a{{{1, 2, 3}, {4,  5,  6}},
                          {{7, 8, 9}, {10, 11, 12}}};

        auto va = xt::view(a, 0);

        for (int i = 0; i < 6; ++i) {
            std::cout << *(va.begin() + i) << " ";
        }
        std::cout << std::endl;
        // prints 1 2 2 3 4 4


        for (auto it = va.begin(); it != va.end(); it++) {
            std::cout << *(it) << " ";
        }
        std::cout << std::endl;
        // prints 1 2 3 4 5 6
*/
    }

BOOST_AUTO_TEST_SUITE_END();


