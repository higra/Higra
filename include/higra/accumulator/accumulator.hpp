//
// Created by user on 4/18/18.
//

#pragma once

#include "xtensor/xview.hpp"
#include "xtensor/xeval.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xoperation.hpp"
#include "../utils.hpp"


namespace hg {

#define HG_ACCUMULATORS (min)(max)(mean)(counter)(sum)(prod)

    enum class accumulators {
        mean,
        min,
        max,
        counter,
        sum,
        prod
    };


    /*
     * Accumulators free functions
     */
    template<typename T>
    void acc_reset(T &acc) {
        xt::view_all(acc.storage) = acc.reset_value;
    }

    //template<typename E, typename T>
    //void acc_accumulate(E & val, T & acc);

    template<typename T>
    auto acc_result(T &acc) {
        return acc.storage;
    }

    /*
     * Accumulator sum
     */
    template<typename T>
    struct accumulator_sum {
        using value_t = T;
        const value_t reset_value = 0;
        xt::xarray<value_t> storage{reset_value};
    };

    template<typename T1, typename T2>
    void acc_accumulate(const T1 &value, accumulator_sum<T2> &acc) {
        acc.storage += value;
    }

    /*
     * Accumulator min
     */
    template<typename T>
    struct accumulator_min {
        using value_t = T;
        const value_t reset_value = std::numeric_limits<value_t>::max();
        xt::xarray<value_t> storage{reset_value};
    };

    template<typename T1, typename T2>
    void acc_accumulate(const T1 &value, accumulator_min<T2> &acc) {
        if (xt::all(acc.storage > value))
            xt::view_all(acc.storage) = value;
    }

    /*
    * Accumulator max
    */
    template<typename T>
    struct accumulator_max {
        using value_t = T;
        const value_t reset_value = std::numeric_limits<value_t>::lowest();
        xt::xarray<value_t> storage{reset_value};
    };


    template<typename T1, typename T2>
    void acc_accumulate(const T1 &value, accumulator_max<T2> &acc) {
        if (xt::all(acc.storage < value))
            xt::view_all(acc.storage) = value;
    }

    /*
    * Accumulator counter
    */
    struct accumulator_counter {
        using value_t = std::size_t;
        const value_t reset_value = 0;
        value_t storage{reset_value};
    };

    template<typename T>
    void acc_accumulate(const T &value, accumulator_counter &acc) {
        acc.storage++;
    }

    inline
    auto acc_result(accumulator_counter &acc) {
        return xt::xscalar<std::size_t>(acc.storage);
    }


    /*
   * Accumulator mean
   */
    template<typename T>
    struct accumulator_mean {
        using value_t = T;
        xt::xarray<value_t> storage{0};
        std::size_t count{0};
    };

    template<typename T>
    void acc_reset(accumulator_mean<T> &acc) {
        xt::dynamic_view(acc.storage, {}) = 0;
        acc.count = 0;
    }

    template<typename T1, typename T2>
    void acc_accumulate(const T1 &value, accumulator_mean<T2> &acc) {
        acc.storage += value;
        acc.count++;
    }

    template<typename T>
    auto acc_result(accumulator_mean<T> &acc) {
        return xt::eval(acc.storage / acc.count);
    }

    /*
    * Accumulator prod
    */
    template<typename T>
    struct accumulator_prod {
        using value_t = T;
        const value_t reset_value = 1;
        xt::xarray<value_t> storage{reset_value};
    };


    template<typename T1, typename T2>
    void acc_accumulate(const T1 &value, accumulator_prod<T2> &acc) {
        acc.storage *= value;
    }

}
