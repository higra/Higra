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
#include <limits>
#include <type_traits>
#include <vector>

namespace test_accumulator {

    // accumulates the first num_values rows of values (all rows if num_values < 0)
    template<bool vec, typename acc_t>
    auto applyAcc(const hg::array_nd<double> &values, acc_t accFactory, hg::index_t num_values = -1) {
        if (num_values < 0)
            num_values = (hg::index_t) values.shape()[0];
        auto inview = hg::make_light_axis_view<vec>(values);
        std::vector<size_t> data_shape(values.shape().begin() + 1, values.shape().end());

        auto out_shape = acc_t::get_output_shape(data_shape);
        if (out_shape.empty())
            out_shape.push_back(1);
        using output_t = hg::accumulator_output_value_t<acc_t, double>;
        auto storage = hg::array_nd<output_t>::from_shape(hg::array_shape::from(out_shape));
        auto acc = hg::accumulator_detail::make_accumulator<vec, double>(
                accFactory, storage, hg::details::light_axis_view_row_size(values));
        acc.initialize();

        for (hg::index_t i = 0; i < num_values; i++) {
            inview.set_position(i);
            acc.accumulate(inview.begin());
        }
        acc.finalize();
        return storage;
    };

    template<typename acc_t>
    auto applyAccG(const hg::array_nd<double> &values, acc_t acc, hg::index_t num_values = -1) {
        if (values.dimension() == 1) {
            return applyAcc<false>(values, acc, num_values);
        } else {
            return applyAcc<true>(values, acc, num_values);
        }
    };

    auto isclose(double a, double b) {
        return std::abs(a - b) < 10e-5;
    }

    TEST_CASE("accumulator scalar", "[accumulator]") {
        hg::array_nd<double> values{-5, 10, -20, 5, 2, -2};
        double r = applyAccG(values, hg::accumulator_max())(0);
        REQUIRE(r == 10.0);
        REQUIRE((applyAccG(values, hg::accumulator_min()))(0) == -20);
        REQUIRE((applyAccG(values, hg::accumulator_sum()))(0) == -10);
        REQUIRE((applyAccG(values, hg::accumulator_counter()))(0) == 6);
        REQUIRE((applyAccG(values, hg::accumulator_first()))(0) == -5);
        REQUIRE((applyAccG(values, hg::accumulator_last()))(0) == -2);
        REQUIRE((applyAccG(values, hg::accumulator_argmin()))(0) == 2);
        REQUIRE((applyAccG(values, hg::accumulator_argmax()))(0) == 1);
        REQUIRE((isclose(applyAccG(values, hg::accumulator_mean())(0), (-5 + 10 + -20 + 5 + 2 + -2) / 6.0)));
        REQUIRE((isclose(applyAccG(values, hg::accumulator_prod())(0), -5 * 10 * -20 * 5 * 2 * -2)));

    }

    TEST_CASE("accumulator vectorial", "[accumulator]") {

        hg::array_nd<double> values{{{0,  1}, {1,  2}},
                                    {{5,  9}, {-1, 4}},
                                    {{-2, 2}, {1,  -1}}};

        auto res1 = applyAccG(values, hg::accumulator_sum());

        hg::array_nd<double> ref1{{3, 12},
                                  {1, 5}};
        REQUIRE(hg::allclose(res1, ref1));

        auto res2 = applyAccG(values, hg::accumulator_mean());
        hg::array_nd<double> ref2{{1,       4},
                                  {1.0 / 3, 5.0 / 3}};
        REQUIRE(hg::allclose(res2, ref2));

        auto res3 = applyAccG(values, hg::accumulator_prod());
        hg::array_nd<double> ref3{{0,  18},
                                  {-1, -8}};
        REQUIRE(hg::allclose(res3, ref3));

        auto res4 = applyAccG(values, hg::accumulator_first());
        hg::array_nd<double> ref4{{0, 1},
                                  {1, 2}};
        REQUIRE(hg::allclose(res4, ref4));

        auto res5 = applyAccG(values, hg::accumulator_last());
        hg::array_nd<double> ref5{{-2, 2},
                                  {1,  -1}};
        REQUIRE(hg::allclose(res5, ref5));

        hg::array_nd<double> values2{{0,  1},
                                     {-1, -2},
                                     {5,  9},
                                     {-1, 4},
                                     {-2, 10},
                                     {1,  -1}};

        auto res6 = applyAccG(values2, hg::accumulator_argmin())(0);
        REQUIRE(res6 == 1);

        auto res7 = applyAccG(values2, hg::accumulator_argmax())(0);
        REQUIRE(res7 == 2);

    }

    TEST_CASE("accumulator output value types", "[accumulator]") {
        using hg::accumulator_output_value_t;
        using hg::index_t;
        static_assert(std::is_same<accumulator_output_value_t<hg::accumulator_counter, float>, index_t>::value);
        static_assert(std::is_same<accumulator_output_value_t<hg::accumulator_argmin, unsigned char>, index_t>::value);
        static_assert(std::is_same<accumulator_output_value_t<const hg::accumulator_argmax, double>, index_t>::value);
        static_assert(std::is_same<accumulator_output_value_t<hg::accumulator_sum, float>, float>::value);
        static_assert(std::is_same<accumulator_output_value_t<hg::accumulator_first, int>, int>::value);
        REQUIRE(hg::accumulator_argmin::get_output_shape(std::vector<size_t>{2, 3}).empty());
        REQUIRE(hg::accumulator_argmax::get_output_shape(std::vector<size_t>{2}).empty());
    }

    TEST_CASE("accumulator empty sequence", "[accumulator]") {
        hg::array_nd<double> scalar{1, 2};
        hg::array_nd<double> vectorial{{1, 2}, {3, 4}};

        for (const auto &values: {scalar, vectorial}) {
            auto row = (values.dimension() == 1) ? 1 : 2;
            auto all_equal = [row](const auto &res, double v) {
                REQUIRE((hg::index_t) res.size() == row);
                for (hg::index_t i = 0; i < row; i++) {
                    REQUIRE(res.data()[i] == v);
                }
            };
            all_equal(applyAccG(values, hg::accumulator_sum(), 0), 0);
            all_equal(applyAccG(values, hg::accumulator_mean(), 0), 0);
            all_equal(applyAccG(values, hg::accumulator_prod(), 0), 1);
            all_equal(applyAccG(values, hg::accumulator_min(), 0), (std::numeric_limits<double>::max)());
            all_equal(applyAccG(values, hg::accumulator_max(), 0), std::numeric_limits<double>::lowest());
            all_equal(applyAccG(values, hg::accumulator_first(), 0), 0);
            all_equal(applyAccG(values, hg::accumulator_last(), 0), 0);
            REQUIRE((applyAccG(values, hg::accumulator_counter(), 0))(0) == 0);
            REQUIRE((applyAccG(values, hg::accumulator_argmin(), 0)).size() == 1);
            REQUIRE((applyAccG(values, hg::accumulator_argmin(), 0))(0) == -1);
            REQUIRE((applyAccG(values, hg::accumulator_argmax(), 0))(0) == -1);
        }
    }

    TEST_CASE("accumulator argmin argmax extreme values and ties", "[accumulator]") {
        const double big = (std::numeric_limits<double>::max)();
        const double low = std::numeric_limits<double>::lowest();
        // values equal to the extreme value of the type are selected
        REQUIRE((applyAccG(hg::array_nd<double>{big, big}, hg::accumulator_argmin()))(0) == 0);
        REQUIRE((applyAccG(hg::array_nd<double>{low, low}, hg::accumulator_argmax()))(0) == 0);
        REQUIRE((applyAccG(hg::array_nd<double>{{big, big}, {big, big}}, hg::accumulator_argmin()))(0) == 1);
        REQUIRE((applyAccG(hg::array_nd<double>{{low, low}, {low, low}}, hg::accumulator_argmax()))(0) == 1);

        // scalar ties: the earlier value is kept
        REQUIRE((applyAccG(hg::array_nd<double>{1, 1, 2}, hg::accumulator_argmin()))(0) == 0);
        REQUIRE((applyAccG(hg::array_nd<double>{2, 2, 1}, hg::accumulator_argmax()))(0) == 0);
        // vectorial values: component-wise order, equal vectors keep the later value,
        // incomparable vectors keep the earlier one
        REQUIRE((applyAccG(hg::array_nd<double>{{1, 1}, {1, 1}, {2, 0}}, hg::accumulator_argmin()))(0) == 1);
        REQUIRE((applyAccG(hg::array_nd<double>{{1, 2}, {2, 1}, {3, 3}}, hg::accumulator_argmin()))(0) == 0);
        REQUIRE((applyAccG(hg::array_nd<double>{{1, 1}, {1, 1}, {0, 2}}, hg::accumulator_argmax()))(0) == 1);
        REQUIRE((applyAccG(hg::array_nd<double>{{1, 2}, {2, 1}, {0, 0}}, hg::accumulator_argmax()))(0) == 0);
    }
}
