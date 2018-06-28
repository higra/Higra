//
// Created by user on 4/2/18.
//

#include <boost/test/unit_test.hpp>

#include "higra/structure/fibonacci_heap.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE(fibonacci_heap);

    using namespace hg;
    using namespace std;


    BOOST_AUTO_TEST_CASE(testPool) {
        fibonacci_heap_internal::object_pool<int> pool;
        


    }



BOOST_AUTO_TEST_SUITE_END();