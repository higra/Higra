//
// Created by user on 4/15/18.
//
#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "accumulator.hpp"

#include <vector>


BOOST_AUTO_TEST_SUITE(TestSuiteAccumulator);

    template<typename T, typename acc_t>
    auto applyAcc(const T &values, acc_t acc) {
        acc.reset();
        for (auto &v : values)
            acc.accumulate(v);

        return acc.result();
    };

    auto isclose(double a, double b) {
        return abs(a - b) < 10e-5;
    }

    BOOST_AUTO_TEST_CASE(accumulatorScalar) {

        std::vector<double> values{-5, 10, 5, 2, -2};
        BOOST_CHECK((applyAcc(values, hg::accumulator_max<double>())) == 10);
        BOOST_CHECK((applyAcc(values, hg::accumulator_min<double>())) == -5);
        BOOST_CHECK((applyAcc(values, hg::accumulator_sum<double>())) == 10);
        BOOST_CHECK((applyAcc(values, hg::accumulator_counter<double>())) == 5);
        BOOST_CHECK((isclose(applyAcc(values, hg::accumulator_mean<double>()), 2)));
        BOOST_CHECK((isclose(applyAcc(values, hg::accumulator_prod<double>()), 1000)));

    }

BOOST_AUTO_TEST_SUITE_END();