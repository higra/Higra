/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <boost/test/unit_test.hpp>
#include "../test_utils.hpp"
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

        xt::xtensor<unsigned long, 1> input{1, 1, 1, 1, 1, 1, 1, 1};

        auto res1 = accumulate_parallel(tree, input, hg::accumulator_sum());
        xt::xtensor<unsigned long, 1> ref1{0, 0, 0, 0, 0, 2, 3, 2};
        BOOST_CHECK(xt::allclose(ref1, res1));

        xt::xtensor<unsigned long, 1> vertex_data{1, 1, 1, 1, 1};
        auto res2 = accumulate_sequential(tree, vertex_data, hg::accumulator_sum());
        xt::xtensor<unsigned long, 1> ref2{1, 1, 1, 1, 1, 2, 3, 5};
        BOOST_CHECK(xt::allclose(ref2, res2));

        auto res3 = accumulate_and_combine_sequential(tree, input, vertex_data, hg::accumulator_max(),
                                                      std::plus<unsigned long>());
        xt::xtensor<unsigned long, 1> ref3{1, 1, 1, 1, 1, 2, 2, 3};
        BOOST_CHECK(xt::allclose(ref3, res3));

    }


    BOOST_AUTO_TEST_CASE(treeAccumulatorVect) {

        auto tree = data.t;

        xt::xtensor<unsigned long, 2> input{{1, 0},
                                            {1, 1},
                                            {1, 2},
                                            {1, 3},
                                            {1, 4},
                                            {1, 5},
                                            {1, 6},
                                            {1, 7}};

        auto res1 = accumulate_parallel(tree, input, hg::accumulator_min());
        auto longmax = std::numeric_limits<unsigned long>::max();
        xt::xtensor<unsigned long, 2> ref1{{longmax, longmax},
                                           {longmax, longmax},
                                           {longmax, longmax},
                                           {longmax, longmax},
                                           {longmax, longmax},
                                   {1,        0},
                                   {1,        2},
                                   {1,        5}};
        BOOST_CHECK(xt::allclose(ref1, res1));

        xt::xtensor<unsigned long, 2> vertex_data{{1, 0},
                                                  {1, 1},
                                                  {1, 2},
                                                  {1, 3},
                                                  {1, 4}};
        auto res2 = accumulate_sequential(tree, vertex_data, hg::accumulator_sum());
        xt::xtensor<unsigned long, 2> ref2{{1, 0},
                                           {1, 1},
                                           {1, 2},
                                           {1, 3},
                                           {1, 4},
                                           {2, 1},
                                           {3, 9},
                                           {5, 10}};
        BOOST_CHECK(xt::allclose(ref2, res2));

        auto res3 = accumulate_and_combine_sequential(tree, input, vertex_data, hg::accumulator_sum(),
                                                      std::plus<unsigned long>());
        xt::xtensor<unsigned long, 2> ref3{{1, 0},
                                   {1, 1},
                                   {1, 2},
                                   {1, 3},
                                   {1, 4},
                                   {3, 6},
                                   {4, 15},
                                   {8, 28}};
        BOOST_CHECK(xt::allclose(ref3, res3));

    }


    BOOST_AUTO_TEST_CASE(treePropagate) {
        auto tree = data.t;
        array_1d<int> input{1, 2, 3, 4, 5, 6, 7, 8};
        array_1d<bool> condition{true, false, true, false, true, true, false, false};

        auto output = propagate_parallel(tree, input);
        array_1d<int> ref{6, 6, 7, 7, 7, 8, 8, 8};
        BOOST_CHECK(xt::allclose(ref, output));

        auto output2 = propagate_parallel(tree, input, condition);
        array_1d<int> ref2{6, 2, 7, 4, 7, 8, 7, 8};
        BOOST_CHECK(xt::allclose(ref2, output2));

        auto output3 = propagate_sequential(tree, input, condition);
        array_1d<int> ref3{8, 2, 7, 4, 7, 8, 7, 8};
        BOOST_CHECK(xt::allclose(ref3, output3));
    }

    BOOST_AUTO_TEST_CASE(treePropagateVect) {
        auto tree = data.t;
        array_2d<int> input{{1, 8},
                            {2, 7},
                            {3, 6},
                            {4, 5},
                            {5, 4},
                            {6, 3},
                            {7, 2},
                            {8, 1}};

        array_1d<bool> condition{true, false, true, false, true, true, false, false};

        auto output = propagate_parallel(tree, input);
        array_2d<int> ref{{6, 3},
                          {6, 3},
                          {7, 2},
                          {7, 2},
                          {7, 2},
                          {8, 1},
                          {8, 1},
                          {8, 1}};
        BOOST_CHECK(xt::allclose(ref, output));

        auto output2 = propagate_parallel(tree, input, condition);
        array_2d<int> ref2{{6, 3},
                           {2, 7},
                           {7, 2},
                           {4, 5},
                           {7, 2},
                           {8, 1},
                           {7, 2},
                           {8, 1}};
        BOOST_CHECK(xt::allclose(ref2, output2));

        auto output3 = propagate_sequential(tree, input, condition);
        array_2d<int> ref3{{8, 1},
                           {2, 7},
                           {7, 2},
                           {4, 5},
                           {7, 2},
                           {8, 1},
                           {7, 2},
                           {8, 1}};
        BOOST_CHECK(xt::allclose(ref3, output3));
    }

BOOST_AUTO_TEST_SUITE_END();