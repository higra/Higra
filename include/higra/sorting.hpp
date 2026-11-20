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

#include <algorithm>
#include <iterator>

#ifdef HG_USE_TBB

    #include <memory>
    #include <new>
    #include <oneapi/tbb/parallel_invoke.h>
    #include <oneapi/tbb/parallel_sort.h>

#endif

namespace hg {

#ifdef HG_USE_TBB
    namespace sorting_detail {

        /**
         * Stable merge of the sorted ranges [xs, xe) and [ys, ye) into the uninitialized storage starting at zs.
         * Elements are move-constructed into zs; on ties, elements of the first range come first.
         * Large merges are split in two independent merges executed with oneTBB.
         */
        template<typename RandomAccessIterator1, typename RandomAccessIterator2, typename T, typename Compare>
        void parallel_move_merge(RandomAccessIterator1 xs, RandomAccessIterator1 xe,
                                 RandomAccessIterator2 ys, RandomAccessIterator2 ye,
                                 T *zs, Compare comp) {
            const auto merge_cut_off = typename std::iterator_traits<RandomAccessIterator1>::difference_type(2048);
            const auto size = (xe - xs) + (ye - ys);

            if (size <= merge_cut_off) {
                while (xs != xe && ys != ye) {
                    if (comp(*ys, *xs)) {
                        new(zs++) T(std::move(*ys++));
                    } else {
                        new(zs++) T(std::move(*xs++));
                    }
                }
                while (xs != xe) {
                    new(zs++) T(std::move(*xs++));
                }
                while (ys != ye) {
                    new(zs++) T(std::move(*ys++));
                }
                return;
            }

            RandomAccessIterator1 xm;
            RandomAccessIterator2 ym;
            if (xe - xs >= ye - ys) {
                xm = xs + (xe - xs) / 2;
                ym = std::lower_bound(ys, ye, *xm, comp);
            } else {
                ym = ys + (ye - ys) / 2;
                xm = std::upper_bound(xs, xe, *ym, comp);
            }

            const auto zm = zs + (xm - xs) + (ym - ys);
            oneapi::tbb::parallel_invoke(
                    [&] { parallel_move_merge(xs, xm, ys, ym, zs, comp); },
                    [&] { parallel_move_merge(xm, xe, ym, ye, zm, comp); });
        }

        /**
         * Parallel stable merge sort: both halves are sorted in parallel with oneTBB, then merged in parallel
         * through a temporary buffer. Small ranges are sorted with std::stable_sort.
         */
        template<typename RandomAccessIterator, typename Compare>
        void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
            const auto sort_cut_off = typename std::iterator_traits<RandomAccessIterator>::difference_type(4096);
            const auto size = xe - xs;

            if (size <= sort_cut_off) {
                std::stable_sort(xs, xe, comp);
                return;
            }

            const auto xm = xs + size / 2;
            oneapi::tbb::parallel_invoke(
                    [&] { parallel_stable_sort(xs, xm, comp); },
                    [&] { parallel_stable_sort(xm, xe, comp); });

            using value_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
            std::allocator<value_type> allocator;
            value_type *buffer = allocator.allocate(size);
            parallel_move_merge(xs, xm, xm, xe, buffer, comp);
            std::move(buffer, buffer + size, xs);
            for (auto index = size; index != 0; --index) {
                buffer[index - 1].~value_type();
            }
            allocator.deallocate(buffer, size);
        }
    }
#endif


    template<typename RandomAccessIterator, typename Compare>
    void stable_sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
#ifdef HG_USE_TBB
        sorting_detail::parallel_stable_sort(xs, xe, comp);
#else
        std::stable_sort(xs, xe, comp);
#endif
    }


    template<typename RandomAccessIterator, typename Compare>
    void sort(RandomAccessIterator xs, RandomAccessIterator xe, Compare comp) {
#ifdef HG_USE_TBB
        oneapi::tbb::parallel_sort(xs, xe, comp);
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
    void stable_sort(array_view<T> array, Compare comp) {
        hg_assert_1d_array(array);
        hg::stable_sort(array.begin(), array.end(), comp);
    }

    template<typename T>
    void stable_sort(array_view<T> arrayx) {
        hg::stable_sort(arrayx, std::less<T>());
    }

    template<typename RandomAccessIterator>
    void sort(RandomAccessIterator xs, RandomAccessIterator xe) {
        typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
        hg::sort(xs, xe, std::less<T>());
    }

    template<typename T, typename Compare>
    void sort(array_view<T> array, Compare comp) {
        hg_assert_1d_array(array);
        hg::sort(array.begin(), array.end(), comp);
    }

    template<typename T>
    void sort(array_view<T> arrayx) {
        hg::sort(arrayx, std::less<T>());
    }

#define HIGRA_ARG_SORT(sort_function)                                                                   \
    const auto &array = arrayx;                                                                         \
    hg_assert(array.dimension() > 0 && array.dimension() <= 2,                                          \
          "Array dimension must be strictly positive and lower or equal than 2.");                      \
    array_1d <index_t> indices = hg::arange<index_t>(array.shape()[0]);                                 \
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
    auto arg_sort(array_view<const T> arrayx, Compare comp) {
        HIGRA_ARG_SORT(hg::sort);
    }

    template<typename T>
    auto arg_sort(array_view<const T> arrayx) {
        return arg_sort(arrayx, std::less<T>());
    }

    template<typename T, typename Compare>
    auto stable_arg_sort(array_view<const T> arrayx, Compare comp) {
        HIGRA_ARG_SORT(hg::stable_sort);
    }

    template<typename T>
    auto stable_arg_sort(array_view<const T> arrayx) {
        return stable_arg_sort(arrayx, std::less<T>());
    }

#undef HIGRA_ARG_SORT
}