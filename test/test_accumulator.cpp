//
// Created by user on 4/15/18.
//
#include "higra/structure/array.hpp"
#include <boost/test/unit_test.hpp>
#include "test_utils.hpp"
#include "higra/accumulator/accumulator.hpp"
#include "higra/structure/details/light_axis_view.hpp"
#include <vector>


BOOST_AUTO_TEST_SUITE(accumulator);


    template<bool vec, typename acc_t>
    auto applyAcc(const hg::array_nd<double> &values, acc_t accFactory) {
        auto inview = hg::make_light_axis_view<vec>(values);
        std::vector<std::size_t> data_shape(values.shape().begin() + 1, values.shape().end());

        auto out_shape = acc_t::get_output_shape(data_shape);
        if (out_shape.empty())
            out_shape.push_back(1);
        hg::array_nd<double> storage = hg::array_nd<double>::from_shape(out_shape);
        auto acc = accFactory.template make_accumulator<vec>(storage);
        acc.initialize();

        for (std::size_t i = 0; i < values.shape()[0]; i++) {
            inview.set_position(i);
            acc.accumulate(inview);
        }
        acc.finalize();
        return storage;
    };

    template<typename acc_t>
    auto applyAccG(const hg::array_nd<double> &values, acc_t acc) {
        if (values.dimension() == 1) {
            return applyAcc<false>(values, acc);
        } else {
            return applyAcc<true>(values, acc);
        }
    };

    auto isclose(double a, double b) {
        return std::abs(a - b) < 10e-5;
    }

    BOOST_AUTO_TEST_CASE(accumulatorScalar) {

        hg::array_nd<double> values{-5, 10, 5, 2, -2};
        double r = applyAccG(values, hg::accumulator_max())();
        BOOST_CHECK(r == 10.0);
        BOOST_CHECK((applyAccG(values, hg::accumulator_min()))() == -5);
        BOOST_CHECK((applyAccG(values, hg::accumulator_sum()))() == 10);
        BOOST_CHECK((applyAccG(values, hg::accumulator_counter()))() == 5);
        BOOST_CHECK((isclose(applyAccG(values, hg::accumulator_mean())(), 2)));
        BOOST_CHECK((isclose(applyAccG(values, hg::accumulator_prod())(), 1000)));

    }

    BOOST_AUTO_TEST_CASE(accumulatorVectorial) {

        hg::array_nd<double> values{{{0,  1}, {1,  2}},
                                    {{5,  9}, {-1, 4}},
                                    {{-2, 2}, {1,  -1}}};

        auto res1 = applyAccG(values, hg::accumulator_sum());

        hg::array_nd<double> ref1{{3, 12},
                                  {1, 5}};
        BOOST_CHECK(xt::allclose(res1, ref1));

        auto res2 = applyAccG(values, hg::accumulator_mean());
        hg::array_nd<double> ref2{{1,       4},
                                  {1.0 / 3, 5.0 / 3}};
        BOOST_CHECK(xt::allclose(res2, ref2));

        auto res3 = applyAccG(values, hg::accumulator_prod());
        hg::array_nd<double> ref3{{0,  18},
                                  {-1, -8}};
        BOOST_CHECK(xt::allclose(res3, ref3));

    }

BOOST_AUTO_TEST_SUITE_END();