
#define BOOST_TEST_MODULE higra_test

#include <boost/test/unit_test.hpp>


#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"


using namespace std;

/**
 * xtensor test... more or less poc functions.
 */


BOOST_AUTO_TEST_SUITE(XTensor);


    BOOST_AUTO_TEST_CASE(tensor_indexAccess) {

        xt::xarray<double> arr1
                {{1.0, 2.0, 3.0},
                 {4.0, 5.0, 6.0},
                 {7.0, 8.0, 9.0}};


        BOOST_CHECK(arr1(1, 2) == 6.0);

        xt::xarray<int> arr2{1, 2, 3, 4, 5, 6, 7, 8, 9};

        BOOST_CHECK(arr2(8) == 9);

    }

    BOOST_AUTO_TEST_CASE(tensor_broadcast) {

        xt::xarray<double> arr1
                {{1.0, 2.0, 3.0, 4.0},
                 {2.0, 5.0, 7.0, 5.0},
                 {2.0, 5.0, 7.0, 3.0}};

        xt::xarray<double> arr2
                {5.0, 6.0, 7.0, 1.0};

        auto res = arr1 + arr2;

        BOOST_CHECK(res(1, 2) == 7.0 + 7.0);

    }

    BOOST_AUTO_TEST_CASE(tensor_coordinatesAccess) {

        xt::xarray<double> arr
                {{1.0, 2.0,  3.0, 4.0},
                 {0.0, 5.0,  7.0, 5.0},
                 {9.0, 11.0, 8.0, 10.0}};


        auto val = arr[{1, 2}];
        BOOST_CHECK(val == 7.0);

        vector<ulong> coordVec{1, 2};
        val = arr[coordVec];
        BOOST_CHECK(val == 7.0);


        xt::xarray<double> coordArr{1, 2};
        val = arr[coordArr];
        BOOST_CHECK(val == 7.0);
    }

    BOOST_AUTO_TEST_CASE(tensor_linearAccess) {
        xt::xarray<int> a = {{{1, 2, 3}, {4,  5,  6}},
                             {{7, 8, 9}, {10, 11, 12}}};
        for (int i = 0; i < 12; ++i) {
            BOOST_CHECK(a(i) - 1 == i);
        }
    }


BOOST_AUTO_TEST_SUITE_END();






