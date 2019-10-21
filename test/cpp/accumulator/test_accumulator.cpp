/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/
#include "../test_utils.hpp"
#include "higra/structure/array.hpp"
#include "higra/accumulator/accumulator.hpp"
#include "higra/structure/details/light_axis_view.hpp"
#include <vector>

namespace accumulator {

    template<bool vec, typename acc_t>
    auto applyAcc(const hg::array_nd<double> &values, acc_t accFactory) {
        auto inview = hg::make_light_axis_view<vec>(values);
        std::vector<size_t> data_shape(values.shape().begin() + 1, values.shape().end());

        auto out_shape = acc_t::get_output_shape(data_shape);
        if (out_shape.empty())
            out_shape.push_back(1);
        hg::array_nd<double> storage = hg::array_nd<double>::from_shape(out_shape);
        auto acc = accFactory.template make_accumulator<vec>(storage);
        acc.initialize();

        for (hg::index_t i = 0; i < (hg::index_t) values.shape()[0]; i++) {
            inview.set_position(i);
            acc.accumulate(inview.begin());
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

    TEST_CASE("accumulator scalar", "[accumulator]") {
        hg::array_nd<double> values{-5, 10, -20, 5, 2, -2};
        double r = applyAccG(values, hg::accumulator_max())();
        REQUIRE(r == 10.0);
        REQUIRE((applyAccG(values, hg::accumulator_min()))() == -20);
        REQUIRE((applyAccG(values, hg::accumulator_sum()))() == -10);
        REQUIRE((applyAccG(values, hg::accumulator_counter()))() == 6);
        REQUIRE((applyAccG(values, hg::accumulator_first()))() == -5);
        REQUIRE((applyAccG(values, hg::accumulator_last()))() == -2);
        REQUIRE((applyAccG(values, hg::accumulator_argmin()))() == 2);
        REQUIRE((applyAccG(values, hg::accumulator_argmax()))() == 1);
        REQUIRE((isclose(applyAccG(values, hg::accumulator_mean())(), (-5 + 10 + -20 + 5 + 2 + -2) / 6.0)));
        REQUIRE((isclose(applyAccG(values, hg::accumulator_prod())(), -5 * 10 * -20 * 5 * 2 * -2)));

    }

    TEST_CASE("accumulator vectorial", "[accumulator]") {

        hg::array_nd<double> values{{{0,  1}, {1,  2}},
                                    {{5,  9}, {-1, 4}},
                                    {{-2, 2}, {1,  -1}}};

        auto res1 = applyAccG(values, hg::accumulator_sum());

        hg::array_nd<double> ref1{{3, 12},
                                  {1, 5}};
        REQUIRE(xt::allclose(res1, ref1));

        auto res2 = applyAccG(values, hg::accumulator_mean());
        hg::array_nd<double> ref2{{1,       4},
                                  {1.0 / 3, 5.0 / 3}};
        REQUIRE(xt::allclose(res2, ref2));

        auto res3 = applyAccG(values, hg::accumulator_prod());
        hg::array_nd<double> ref3{{0,  18},
                                  {-1, -8}};
        REQUIRE(xt::allclose(res3, ref3));

        auto res4 = applyAccG(values, hg::accumulator_first());
        hg::array_nd<double> ref4{{0, 1},
                                  {1, 2}};
        REQUIRE(xt::allclose(res4, ref4));

        auto res5 = applyAccG(values, hg::accumulator_last());
        hg::array_nd<double> ref5{{-2, 2},
                                  {1,  -1}};
        REQUIRE(xt::allclose(res5, ref5));

        hg::array_nd<double> values2{{0,  1},
                                     {-1, -2},
                                     {5,  9},
                                     {-1, 4},
                                     {-2, 10},
                                     {1,  -1}};

        auto res6 = applyAccG(values2, hg::accumulator_argmin())();
        REQUIRE(res6 == 1);

        auto res7 = applyAccG(values2, hg::accumulator_argmax())();
        REQUIRE(res7 == 2);

    }
}