#define HG_DEBUG

#define BOOST_TEST_MODULE higra_test

#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/structure/array.hpp"
#include "xtensor/xoperation.hpp"
#include "xtensor/xscalar.hpp"
#include "higra/utils.hpp"


BOOST_AUTO_TEST_SUITE(xp);


    BOOST_AUTO_TEST_CASE(xpp) {

        xt::xarray<int> a1(1);
        xt::xarray<int> a2{{1, 2},
                           {3, 4}};


        xt::xtensor<int, 1> a{4};
        xt::xscalar<int> s(2);
        a1 *= a2;
        //xt::dynamic_view(s, {}) = a2;
        //int i = s;
        //     a=a2;

        //std::cout << a2;
    }

BOOST_AUTO_TEST_SUITE_END();


