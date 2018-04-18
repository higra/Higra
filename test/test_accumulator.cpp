//
// Created by user on 4/15/18.
//
#include "higra/structure/array.hpp"
#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/accumulator/accumulator.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include <vector>


BOOST_AUTO_TEST_SUITE(accumulator);

    template<typename acc_t>
    auto applyAcc(const hg::array_nd<double> &values, acc_t acc) {
        hg::acc_reset(acc);
        if (values.dimension() == 1)

            for (auto v: values)
                hg::acc_accumulate(v, acc);

        return hg::acc_result(acc);
    };

    template<typename acc_t>
    auto applyAccV(const hg::array_nd<double> &values, acc_t acc) {
        hg::acc_reset(acc);

        for (auto it = xt::axis_begin(values); it != xt::axis_end(values); it++)
            hg::acc_accumulate(*it, acc);

        return hg::acc_result(acc);
    };

    template<typename acc_t>
    auto applyAccG(const hg::array_nd<double> &values, acc_t acc) {
        hg::acc_reset(acc);
        if (values.dimension() == 1) {
            for (auto v: values)
                hg::acc_accumulate(v, acc);
        } else {
            for (auto it = xt::axis_begin(values); it != xt::axis_end(values); it++)
                hg::acc_accumulate(*it, acc);
        }
        return hg::acc_result(acc);
    };

    auto isclose(double a, double b) {
        return abs(a - b) < 10e-5;
    }

    BOOST_AUTO_TEST_CASE(accumulatorScalar) {

        hg::array_nd<double> values{-5, 10, 5, 2, -2};
        double r = applyAccG(values, hg::accumulator_max<double>())();
        BOOST_CHECK(r == 10.0);
        BOOST_CHECK((applyAccG(values, hg::accumulator_min<double>()))() == -5);
        BOOST_CHECK((applyAccG(values, hg::accumulator_sum<double>()))() == 10);
        BOOST_CHECK((applyAccG(values, hg::accumulator_counter()))() == 5);
        BOOST_CHECK((isclose(applyAccG(values, hg::accumulator_mean<double>())(), 2)));
        BOOST_CHECK((isclose(applyAccG(values, hg::accumulator_prod<double>())(), 1000)));

    }

    BOOST_AUTO_TEST_CASE(accumulatorVectorial) {

        hg::array_nd<double> values{{{0,  1}, {1,  2}},
                                    {{5,  9}, {-1, 4}},
                                    {{-2, 2}, {1,  -1}}};

        auto res1 = applyAccG(values, hg::accumulator_sum<double>());

        hg::array_nd<double> ref1{{3, 12},
                                  {1, 5}};
        BOOST_CHECK(xt::allclose(res1, ref1));

        auto res2 = applyAccV(values, hg::accumulator_mean<double>());
        hg::array_nd<double> ref2{{1,       4},
                                  {1.0 / 3, 5.0 / 3}};
        BOOST_CHECK(xt::allclose(res2, ref2));

        auto res3 = applyAccV(values, hg::accumulator_prod<double>());
        hg::array_nd<double> ref3{{0,  18},
                                  {-1, -8}};
        BOOST_CHECK(xt::allclose(res3, ref3));

    }

BOOST_AUTO_TEST_SUITE_END();