//
// Created by user on 3/22/18.
//

#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/accumulator/accumulator.hpp"
#include "higra/accumulator/tree_accumulator.hpp"
#include <functional>

BOOST_AUTO_TEST_SUITE(treeAccumulator);

    using namespace hg;
    using namespace std;

    struct _data {

        hg::tree t;

        _data() : t(xt::xarray<long>{5, 5, 6, 6, 6, 7, 7, 7}) {
        }

    } data;

    BOOST_AUTO_TEST_CASE(treeAccumulator) {

        auto tree = data.t;

        xt::xtensor<ulong, 1> input{1, 1, 1, 1, 1, 1, 1, 1};
        xt::xtensor<ulong, 1> res1{0, 0, 0, 0, 0, 0, 0, 0};

        accumulate_parallel(tree, input, res1, hg::accumulator_sum<ulong>());
        xt::xtensor<ulong, 1> ref1{0, 0, 0, 0, 0, 2, 3, 2};
        BOOST_CHECK(xt::allclose(ref1, res1));

        xt::xtensor<ulong, 1> res2{1, 1, 1, 1, 1, 0, 0, 0};
        accumulate_sequential(tree, res2, hg::accumulator_sum<ulong>());
        xt::xtensor<ulong, 1> ref2{1, 1, 1, 1, 1, 2, 3, 5};
        BOOST_CHECK(xt::allclose(ref2, res2));

        xt::xtensor<ulong, 1> res3{1, 1, 1, 1, 1, 1, 1, 1};
        accumulate_and_combine_sequential(tree, input, res3, hg::accumulator_max<ulong>(), std::plus<ulong>());
        xt::xtensor<ulong, 1> ref3{1, 1, 1, 1, 1, 2, 2, 3};
        BOOST_CHECK(xt::allclose(ref3, res3));

    }

BOOST_AUTO_TEST_SUITE_END();