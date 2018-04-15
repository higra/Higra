//
// Created by user on 4/15/18.
//

#pragma once

#include <functional>
#include "xtensor/xexpression.hpp"
#include "xtensor/xstrided_view.hpp"

namespace hg {


#define HG_ACCUMULATORS (min)(max)(mean)(counter)(sum)(prod)

    namespace accumulator_internal {

        template<
                typename result_value_t,
                typename acc_value_t,
                typename inner_storage_t,
                typename reset_fun_t = typename std::function<void(inner_storage_t &)>,
                typename accumulate_fun_t = typename std::function<void(inner_storage_t &, const acc_value_t &)>,
                typename result_fun_t = typename std::function<acc_value_t(inner_storage_t &)>
        >
        class accumulator {
            inner_storage_t storage;
            reset_fun_t _reset;
            accumulate_fun_t _accumulate;
            result_fun_t _result;


        public:

            accumulator(reset_fun_t reset, accumulate_fun_t accumulate, result_fun_t result) : _reset(reset),
                                                                                               _accumulate(accumulate),
                                                                                               _result(result) {
                _reset(storage);
            }

            void reset() {
                _reset(storage);
            }

            void accumulate(const acc_value_t &v) {
                _accumulate(storage, v);
            }

            auto result() {
                return _result(storage);
            }

        };

    }

    template<typename value_t,
            typename = std::enable_if_t<std::is_arithmetic<value_t>::value> >
    auto accumulator_max() {
        return accumulator_internal::accumulator<value_t, value_t, value_t>(
                [](value_t &s) {
                    s = std::numeric_limits<value_t>::lowest();
                },
                [](value_t &s, const value_t &v) {
                    if (v > s)
                        s = v;
                },
                [](value_t &s) {
                    return s;
                });
    }

    template<typename value_t,
            typename = std::enable_if_t<std::is_arithmetic<value_t>::value> >
    auto accumulator_min() {
        return accumulator_internal::accumulator<value_t, value_t, value_t>(
                [](value_t &s) {
                    s = std::numeric_limits<value_t>::max();
                },
                [](value_t &s, const value_t &v) {
                    if (v < s)
                        s = v;
                },
                [](value_t &s) {
                    return s;
                });
    }

    template<typename value_t>
    auto accumulator_counter() {
        return accumulator_internal::accumulator<std::size_t, value_t, std::size_t>(
                [](std::size_t &s) {
                    s = 0;
                },
                [](std::size_t &s, const value_t &) {
                    ++s;
                },
                [](std::size_t &s) {
                    return s;
                });
    }

    template<typename value_t,
            typename = std::enable_if_t<std::is_arithmetic<value_t>::value> >
    auto accumulator_sum() {
        return accumulator_internal::accumulator<value_t, value_t, value_t>(
                [](value_t &s) {
                    s = 0;
                },
                [](value_t &s, const value_t &v) {
                    s += v;
                },
                [](value_t &s) {
                    return s;
                });
    }

    template<typename value_t,
            typename = std::enable_if_t<std::is_arithmetic<value_t>::value> >
    auto accumulator_prod() {
        return accumulator_internal::accumulator<value_t, value_t, value_t>(
                [](value_t &s) {
                    s = 1;
                },
                [](value_t &s, const value_t &v) {
                    s *= v;
                },
                [](value_t &s) {
                    return s;
                });
    }

    template<typename value_t,
            typename = std::enable_if_t<std::is_arithmetic<value_t>::value> >
    auto accumulator_mean() {
        return accumulator_internal::accumulator<value_t, value_t, std::pair<value_t, std::size_t>>(
                [](std::pair<value_t, std::size_t> &s) {
                    s.first = 0;
                    s.second = 0;
                },
                [](std::pair<value_t, std::size_t> &s, const value_t &v) {
                    s.first += v;
                    s.second++;
                },
                [](std::pair<value_t, std::size_t> &s) {
                    return s.first / s.second;
                });
    }

}