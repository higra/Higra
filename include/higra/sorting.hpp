/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include "structure/array.hpp"

#ifdef HG_USE_TBB

#include "tbb/parallel_sort.h"
#include "tbb-ssort/parallel_stable_sort.h"

#else

#include <algorithm>

#endif

namespace hg {


    template<typename RandomAccessIterator, typename Compare>
    void stable_sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
#ifdef HG_USE_TBB
        pss::parallel_stable_sort(xs, xe, comp);
#else
        std::stable_sort(xs, xe, comp);
#endif
    }


    template<typename RandomAccessIterator, typename Compare>
    void sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
#ifdef HG_USE_TBB
        tbb::parallel_sort(xs, xe, comp);
#else
        std::sort(xs, xe, comp);
#endif
    }


    template<typename RandomAccessIterator>
    void stable_sort(RandomAccessIterator xs, RandomAccessIterator xe) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        hg::stable_sort(xs, xe, std::less<T>());
    }

    template<typename T, typename Compare>
    void stable_sort(xt::xexpression<T> &arrayx, Compare comp) {
        auto &array = arrayx.derived_cast();
        hg_assert_1d_array(array);
        hg::stable_sort(array.begin(), array.end(), comp);
    }

    template<typename T>
    void stable_sort(xt::xexpression<T> &arrayx) {
        hg::stable_sort(arrayx, std::less<typename T::value_type>());
    }

    template<typename RandomAccessIterator>
    void sort(RandomAccessIterator xs, RandomAccessIterator xe) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        hg::sort(xs, xe, std::less<T>());
    }

    template<typename T, typename Compare>
    void sort(xt::xexpression<T> &arrayx, Compare comp) {
        auto &array = arrayx.derived_cast();
        hg_assert_1d_array(array);
        hg::sort(array.begin(), array.end(), comp);
    }

    template<typename T>
    void sort(xt::xexpression<T> &arrayx) {
        hg::sort(arrayx, std::less<typename T::value_type>());
    }

#define HIGRA_ARG_SORT(sort_function)                                                                   \
    auto &array = arrayx.derived_cast();                                                                \
    hg_assert(array.dimension() > 0 && array.dimension() <= 2,                                          \
          "Array dimension must be strictly positive and lower or equal than 2.");                      \
    array_1d <index_t> indices = xt::arange<index_t>(array.shape()[0]);                                 \
    if (array.dimension() == 1) {                                                                       \
        sort_function(indices.begin(), indices.end(),                                                   \
        [&array, &comp](index_t i, index_t j) { return comp(array(i), array(j)); });                    \
    } else {                                                                                            \
    index_t dim = array.shape()[1];                                                                     \
    sort_function(indices.begin(), indices.end(),                                                       \
        [&array, &comp, dim](index_t i, index_t j) {                                                    \
            for (index_t k = 0; k < dim; ++k) {                                                         \
                if (comp(array(i, k), array(j, k))) {                                                   \
                    return true;                                                                        \
                } else if (comp(array(j, k), array(i, k))) {                                            \
                    return false;                                                                       \
                }                                                                                       \
            }                                                                                           \
            return false;                                                                               \
        });                                                                                             \
    }                                                                                                   \
return indices;

    template<typename T, typename Compare>
    auto arg_sort(const xt::xexpression<T> &arrayx, Compare comp) {
        HIGRA_ARG_SORT(hg::sort);
    }

    template<typename T>
    auto arg_sort(const xt::xexpression<T> &arrayx) {
        return arg_sort(arrayx, std::less<typename T::value_type>());
    }

    template<typename T, typename Compare>
    auto stable_arg_sort(const xt::xexpression<T> &arrayx, Compare comp) {
        HIGRA_ARG_SORT(hg::stable_sort);
    }

    template<typename T>
    auto stable_arg_sort(const xt::xexpression<T> &arrayx) {
        return stable_arg_sort(arrayx, std::less<typename T::value_type>());
    }

#undef HIGRA_ARG_SORT
}