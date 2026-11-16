/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#include "../../utils.hpp"
#include "array.hpp"

namespace hg {

    namespace array_internal {

        template<class T>
        using sum_type_t = std::conditional_t<
                std::is_floating_point_v<T>, T,
                std::conditional_t<std::is_same_v<T, bool> || std::is_signed_v<T>, int64_t, uint64_t>>;

        template<class T>
        constexpr bool is_nan(const T &v) noexcept {
            if constexpr (std::is_floating_point_v<T>) {
                return v != v;
            } else {
                (void) v;
                return false;
            }
        }

        // Value equality across element types, without signed/unsigned conversion surprises.
        template<class T, class U>
        constexpr bool equal_values(const T &a, const U &b) noexcept {
            if constexpr (std::is_integral_v<T> && std::is_integral_v<U> &&
                          std::is_signed_v<T> != std::is_signed_v<U>) {
                if constexpr (std::is_signed_v<T>) {
                    return a >= 0 && static_cast<std::make_unsigned_t<T>>(a) == b;
                } else {
                    return b >= 0 && a == static_cast<std::make_unsigned_t<U>>(b);
                }
            } else {
                return a == b;
            }
        }

        template<class Shape0, class... Shapes>
        void assert_same_shapes(const char *function, const Shape0 &s0, const Shapes &... s) {
            hg_assert(((s0 == s) && ...),
                      std::string(function) + ": all arrays must have the same shape.");
        }

        // Chunked reduction: chunk_fn(b, e) reduces [b, e) to R, then combine(acc, r) folds the
        // chunk results in chunk order, starting from the result of the first chunk.
        template<class R, class ChunkFn, class Combine>
        R reduce_chunks(index_t n, ChunkFn chunk_fn, Combine combine) {
            constexpr index_t chunk = HG_ARRAY_REDUCE_CHUNK;
            if (n <= chunk) {
                return chunk_fn(index_t(0), n);
            }
            std::vector<R> partials(static_cast<std::size_t>((n + chunk - 1) / chunk));
            detail::for_each_chunk(n, [&partials, &chunk_fn](index_t c, index_t b, index_t e) {
                partials[static_cast<std::size_t>(c)] = chunk_fn(b, e);
            });
            R acc = partials[0];
            for (std::size_t c = 1; c < partials.size(); ++c) {
                acc = combine(acc, partials[c]);
            }
            return acc;
        }

        // Sum of [b, e) with 4 interleaved accumulators: s_k sums the elements i with
        // (i - b) % 4 == k, and the result is (s_0 + s_1) + (s_2 + s_3).
        template<class Acc, class T>
        Acc sum_range(const T *data, index_t b, index_t e) noexcept {
            Acc s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            index_t i = b;
            for (; i + 4 <= e; i += 4) {
                s0 += static_cast<Acc>(data[i]);
                s1 += static_cast<Acc>(data[i + 1]);
                s2 += static_cast<Acc>(data[i + 2]);
                s3 += static_cast<Acc>(data[i + 3]);
            }
            if (i < e) s0 += static_cast<Acc>(data[i]);
            if (i + 1 < e) s1 += static_cast<Acc>(data[i + 1]);
            if (i + 2 < e) s2 += static_cast<Acc>(data[i + 2]);
            return (s0 + s1) + (s2 + s3);
        }

        // Index of the first extremal element of [b, e) (b < e); the first NaN wins.
        template<class T, class Better>
        index_t arg_extremum_range(const T *data, index_t b, index_t e, Better better) noexcept {
            index_t best = b;
            if (is_nan(data[b])) {
                return b;
            }
            for (index_t i = b + 1; i < e; ++i) {
                if constexpr (std::is_floating_point_v<T>) {
                    if (is_nan(data[i])) {
                        return i;
                    }
                }
                if (better(data[i], data[best])) {
                    best = i;
                }
            }
            return best;
        }

        template<class T, class Better>
        index_t arg_extremum(array_view<const T> a, Better better, const char *function) {
            assert_cpu_accessible(a.device());
            hg_assert(a.size() > 0, std::string(function) + ": the array cannot be empty.");
            const T *data = a.data();
            return reduce_chunks<index_t>(
                    a.size(),
                    [data, better](index_t b, index_t e) { return arg_extremum_range(data, b, e, better); },
                    [data, better](index_t acc, index_t r) {
                        if (is_nan(data[acc])) return acc;
                        if (is_nan(data[r]) || better(data[r], data[acc])) return r;
                        return acc;
                    });
        }

        template<class Pred>
        bool any_of(index_t n, Pred pred) {
            if (!detail::use_parallel(n)) {
                for (index_t i = 0; i < n; ++i) {
                    if (pred(i)) return true;
                }
                return false;
            }
            std::atomic<bool> found{false};
            detail::for_each_chunk(n, [&found, &pred](index_t, index_t b, index_t e) {
                if (found.load(std::memory_order_relaxed)) return;
                for (index_t i = b; i < e; ++i) {
                    if (pred(i)) {
                        found.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
            });
            return found.load();
        }

        // Throws unless every index of ip[0 .. n) lies in [0, bound). One unsigned comparison per
        // index (negative values become large unsigned values), accumulated without branches so
        // that the loop vectorizes; the offending value is only searched for on failure.
        template<class I>
        void check_indices(const I *ip, index_t n, index_t bound, const char *function) {
            static_assert(is_index_type_v<I>, "Indices must be integers.");
            using U = std::uint64_t;
            const U ubound = static_cast<U>(bound);
            const U invalid = reduce_chunks<U>(
                    n,
                    [ip, ubound](index_t b, index_t e) {
                        U bad = 0;
                        for (index_t i = b; i < e; ++i) {
                            bad |= static_cast<U>(static_cast<U>(static_cast<index_t>(ip[i])) >= ubound);
                        }
                        return bad;
                    },
                    [](U acc, U r) { return acc | r; });
            if (invalid != 0) {
                for (index_t i = 0; i < n; ++i) {
                    const index_t v = static_cast<index_t>(ip[i]);
                    hg_assert(v >= 0 && v < bound,
                              std::string(function) + ": index " + std::to_string(ip[i]) + " at position " +
                              std::to_string(i) + " is out of bounds for an axis of size " +
                              std::to_string(bound) + ".");
                }
            }
        }

        /**
         * Whether ``n`` elements at ``p`` and ``m`` elements at ``q`` share at least one byte.
         * Addresses are compared as integers: ``p`` and ``q`` may point into unrelated buffers.
         */
        template<class T, class U>
        bool ranges_overlap(const T *p, index_t n, const U *q, index_t m) noexcept {
            if (n <= 0 || m <= 0) {
                return false;
            }
            const auto p0 = reinterpret_cast<std::uintptr_t>(p);
            const auto q0 = reinterpret_cast<std::uintptr_t>(q);
            const auto p1 = p0 + static_cast<std::uintptr_t>(n) * sizeof(T);
            const auto q1 = q0 + static_cast<std::uintptr_t>(m) * sizeof(U);
            return p0 < q1 && q0 < p1;
        }

        /**
         * Data of ``src``, or of a copy of ``src`` held in ``storage`` if ``src`` overlaps the
         * ``n`` elements at ``dst``, so that writing to ``dst`` does not change what is read.
         */
        template<class T, class U>
        const T *snapshot_if_overlap(array_view<const T> src, const U *dst, index_t n, array<T> &storage) {
            if (!ranges_overlap(dst, n, src.data(), src.size())) {
                return src.data();
            }
            storage = array<T>(src.shape());
            copy_elements(storage.data(), src.data(), src.size());
            return storage.data();
        }

        /**
         * With ``HG_ARRAY_CHECKS``: throws if the ``n`` elements at ``out`` overlap those at ``in``
         * without being exactly the same memory range.
         */
        template<class R, class T>
        void assert_no_partial_overlap(const char *function, const R *out, const T *in, index_t n) {
#ifdef HG_ARRAY_CHECKS
            const bool same_range = static_cast<const void *>(out) == static_cast<const void *>(in) &&
                                    sizeof(R) == sizeof(T);
            hg_assert(same_range || !ranges_overlap(out, n, in, n),
                      std::string(function) + ": the output partially overlaps an input.");
#else
            (void) function;
            (void) out;
            (void) in;
            (void) n;
#endif
        }

        template<class R, class F, class... T>
        void map_loop(bool parallel, R *out, index_t n, F &f, const T *... in) {
            auto body = [out, &f, in...](index_t b, index_t e) {
                for (index_t i = b; i < e; ++i) {
                    out[i] = std::invoke(f, in[i]...);
                }
            };
            detail::parallel_blocks_if(parallel && detail::use_parallel(n), n, 1024, body);
        }

        template<class F, class... T>
        using map_result_t = std::decay_t<std::invoke_result_t<F &, const T &...>>;
    }

    // ------------------------------------------------------------------------------------------
    // Creation
    // ------------------------------------------------------------------------------------------

    /**
     * Array of the given shape with uninitialized elements.
     */
    template<class T>
    array<T> empty(const array_shape &shape, allocator &alloc = cpu_allocator()) {
        return array<T>(shape, alloc);
    }

    template<class T>
    array<T> full(const array_shape &shape, const array_internal::identity_t<T> &value,
                  allocator &alloc = cpu_allocator()) {
        return array<T>(shape, value, alloc);
    }

    template<class T>
    array<T> zeros(const array_shape &shape, allocator &alloc = cpu_allocator()) {
        return array<T>(shape, T(0), alloc);
    }

    template<class T>
    array<T> ones(const array_shape &shape, allocator &alloc = cpu_allocator()) {
        return array<T>(shape, T(1), alloc);
    }

    /**
     * 1d array ``[start, start + step, ...]`` of the values below ``stop`` (above if ``step < 0``),
     * like ``numpy.arange``: ``ceil((stop - start) / step)`` elements, element ``i`` equal to
     * ``start + i * step``. ``step`` cannot be 0. For integer types, the number of elements and the
     * elements are exact over the whole range of ``T``, even when ``stop - start`` or ``step`` is
     * not representable in ``T``.
     */
    template<class T>
    array<T> arange(const array_internal::identity_t<T> &start, const array_internal::identity_t<T> &stop,
                    const array_internal::identity_t<T> &step = T(1)) {
        static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "arange requires a numeric type.");
        hg_assert(step != T(0), "arange: step cannot be 0.");
        if constexpr (std::is_floating_point_v<T>) {
            const double count = std::ceil((static_cast<double>(stop) - static_cast<double>(start)) /
                                           static_cast<double>(step));
            const index_t n = count > 0 ? static_cast<index_t>(count) : 0;
            auto result = array<T>(array_shape(n));
            T *out = result.data();
            // sequential, like fill (see array_internal::fill)
            for (index_t i = 0; i < n; ++i) {
                out[i] = static_cast<T>(start + static_cast<T>(i) * step);
            }
            return result;
        } else {
            // Exact count and elements for the whole range of T: the count is computed in unsigned
            // 64-bit arithmetic, which cannot overflow. Elements are computed modulo 2^64 (2^32 for
            // types of at most 4 bytes, which keeps the loop as wide as T); every element lies
            // between start and stop, so the result modulo 2^bits(T) is the exact value.
            using U = std::uint64_t;
            using E = std::conditional_t<(sizeof(T) <= 4), std::uint32_t, std::uint64_t>;
            U count;
            if ((step > T(0) && stop <= start) || (step < T(0) && stop >= start)) {
                count = 0;
            } else if (step > T(0)) {
                count = (static_cast<U>(stop) - static_cast<U>(start) - 1) / static_cast<U>(step) + 1;
            } else {
                count = (static_cast<U>(start) - static_cast<U>(stop) - 1) / (U(0) - static_cast<U>(step)) + 1;
            }
            // array_shape rejects a count above the largest index_t
            auto result = array<T>(array_shape(count));
            T *out = result.data();
            const index_t n = result.size();
            const E ustart = static_cast<E>(start);
            const E ustep = static_cast<E>(step);
            // sequential, like fill (see array_internal::fill)
            for (index_t i = 0; i < n; ++i) {
                out[i] = static_cast<T>(ustart + static_cast<E>(i) * ustep);
            }
            return result;
        }
    }

    /**
     * 1d array ``[0, 1, ..., stop - 1]``.
     */
    template<class T>
    array<T> arange(const array_internal::identity_t<T> &stop) {
        return arange<T>(T(0), stop, T(1));
    }

    /**
     * Uninitialized array with the shape and element type of ``a``; the allocator of ``a`` if ``a``
     * is an array, the CPU allocator if it is a view.
     */
    template<class T>
    array<T> empty_like(array_view<const T> a) {
        return array<T>(a.shape());
    }

    template<class T>
    array<T> empty_like(const array<T> &a) {
        return array<T>(a.shape(), a.get_allocator());
    }

    template<class T>
    array<T> full_like(array_view<const T> a, const array_internal::identity_t<T> &value) {
        return array<T>(a.shape(), value);
    }

    template<class T>
    array<T> full_like(const array<T> &a, const array_internal::identity_t<T> &value) {
        return array<T>(a.shape(), value, a.get_allocator());
    }

    template<class T>
    array<T> zeros_like(array_view<const T> a) {
        return array<T>(a.shape(), T(0));
    }

    template<class T>
    array<T> zeros_like(const array<T> &a) {
        return array<T>(a.shape(), T(0), a.get_allocator());
    }

    template<class T>
    array<T> ones_like(array_view<const T> a) {
        return array<T>(a.shape(), T(1));
    }

    template<class T>
    array<T> ones_like(const array<T> &a) {
        return array<T>(a.shape(), T(1), a.get_allocator());
    }

    // ------------------------------------------------------------------------------------------
    // Copy
    // ------------------------------------------------------------------------------------------

    /**
     * ``dst[i] = static_cast<U>(src[i])`` for every element; shapes must be equal.
     * ``dst`` and ``src`` must not partially overlap.
     */
    template<class U, class T>
    void assign(array_view<U> dst, array_view<const T> src) {
        static_assert(!std::is_const_v<U>, "assign: the destination must be mutable.");
        array_internal::assert_cpu_accessible(dst.device());
        array_internal::assert_cpu_accessible(src.device());
        array_internal::assert_same_shapes("assign", dst.shape(), src.shape());
        U *out = dst.data();
        const T *in = src.data();
        if constexpr (std::is_same_v<U, T>) {
            if (out == in) {
                return;
            }
            array_internal::copy_elements(out, in, src.size());
        } else {
            detail::parallel_blocks(src.size(), 4096, [out, in](index_t b, index_t e) {
                for (index_t i = b; i < e; ++i) {
                    out[i] = static_cast<U>(in[i]);
                }
            });
        }
    }

    /**
     * New array with the elements of ``a`` converted to ``U`` (``static_cast``).
     */
    template<class U, class T>
    array<U> cast(array_view<const T> a) {
        array_internal::assert_cpu_accessible(a.device());
        array<U> result(a.shape());
        assign<U, T>(result, a);
        return result;
    }

    // ------------------------------------------------------------------------------------------
    // Elementwise
    // ------------------------------------------------------------------------------------------

    /**
     * ``out[i] = f(a[i], b[i], ...)`` for every element; all shapes must be equal.
     * ``f`` is called sequentially on the calling thread.
     * ``out`` must not partially overlap any input; it may be exactly the same memory as an input
     * (in-place update). With ``HG_ARRAY_CHECKS``, a partial overlap throws.
     */
    template<class R, class F, class... T>
    void map_into(exec::seq_t, array_view<R> out, F &&f, array_view<const T>... in) {
        static_assert(sizeof...(T) >= 1, "map_into requires at least one input array.");
        array_internal::assert_cpu_accessible(out.device());
        (array_internal::assert_cpu_accessible(in.device()), ...);
        array_internal::assert_same_shapes("map_into", out.shape(), in.shape()...);
        (array_internal::assert_no_partial_overlap("map_into", out.data(), in.data(), out.size()), ...);
        array_internal::map_loop(false, out.data(), out.size(), f, in.data()...);
    }

    /**
     * As ``map_into(out, f, ...)``, but ``f`` may be called concurrently (with ``HG_USE_TBB``,
     * above ``HG_ARRAY_PARALLEL_THRESHOLD`` elements). The same non-overlap precondition applies.
     */
    template<class R, class F, class... T>
    void map_into(exec::par_t, array_view<R> out, F &&f, array_view<const T>... in) {
        static_assert(sizeof...(T) >= 1, "map_into requires at least one input array.");
        array_internal::assert_cpu_accessible(out.device());
        (array_internal::assert_cpu_accessible(in.device()), ...);
        array_internal::assert_same_shapes("map_into", out.shape(), in.shape()...);
        (array_internal::assert_no_partial_overlap("map_into", out.data(), in.data(), out.size()), ...);
        array_internal::map_loop(true, out.data(), out.size(), f, in.data()...);
    }

    /**
     * ``map_into(exec::seq, out, f, ...)``: ``out`` must not partially overlap any input.
     */
    template<class R, class F, class... T>
    void map_into(array_view<R> out, F &&f, array_view<const T>... in) {
        map_into(exec::seq, out, std::forward<F>(f), in...);
    }

    /**
     * New array ``r`` with ``r[i] = f(a[i], b[i], ...)``; all shapes must be equal and the element
     * type is the decayed result type of ``f``. ``f`` is called sequentially on the calling thread.
     */
    template<class F, class T0, class... T>
    auto map(exec::seq_t, F &&f, array_view<const T0> in0, array_view<const T>... in)
    -> array<array_internal::map_result_t<F, T0, T...>> {
        array<array_internal::map_result_t<F, T0, T...>> result(in0.shape());
        map_into(exec::seq, result, std::forward<F>(f), in0, in...);
        return result;
    }

    /**
     * As ``map(f, ...)``, but ``f`` may be called concurrently (with ``HG_USE_TBB``, above
     * ``HG_ARRAY_PARALLEL_THRESHOLD`` elements).
     */
    template<class F, class T0, class... T>
    auto map(exec::par_t, F &&f, array_view<const T0> in0, array_view<const T>... in)
    -> array<array_internal::map_result_t<F, T0, T...>> {
        array<array_internal::map_result_t<F, T0, T...>> result(in0.shape());
        map_into(exec::par, result, std::forward<F>(f), in0, in...);
        return result;
    }

    template<class F, class T0, class... T>
    auto map(F &&f, array_view<const T0> in0, array_view<const T>... in)
    -> array<array_internal::map_result_t<F, T0, T...>> {
        return map(exec::seq, std::forward<F>(f), in0, in...);
    }

    // ------------------------------------------------------------------------------------------
    // Reductions
    // ------------------------------------------------------------------------------------------

    /**
     * Smallest element. Throws on empty input. NaN propagates (NumPy ``min`` convention).
     */
    template<class T>
    T min(array_view<const T> a) {
        return a.data()[array_internal::arg_extremum(a, std::less<T>(), "min")];
    }

    /**
     * Largest element. Throws on empty input. NaN propagates (NumPy ``max`` convention).
     */
    template<class T>
    T max(array_view<const T> a) {
        return a.data()[array_internal::arg_extremum(a, std::greater<T>(), "max")];
    }

    /**
     * ``{min(a), max(a)}``. Throws on empty input.
     */
    template<class T>
    std::pair<T, T> minmax(array_view<const T> a) {
        array_internal::assert_cpu_accessible(a.device());
        hg_assert(a.size() > 0, "minmax: the array cannot be empty.");
        const T *data = a.data();
        using pair = std::pair<T, T>;
        auto chunk = [data](index_t b, index_t e) {
            T lo = data[b], hi = data[b];
            if (array_internal::is_nan(lo)) return pair(lo, hi);
            for (index_t i = b + 1; i < e; ++i) {
                const T v = data[i];
                if constexpr (std::is_floating_point_v<T>) {
                    if (v != v) return pair(v, v);
                }
                if (v < lo) lo = v;
                if (hi < v) hi = v;
            }
            return pair(lo, hi);
        };
        return array_internal::reduce_chunks<pair>(a.size(), chunk, [](const pair &acc, const pair &r) {
            if (array_internal::is_nan(acc.first)) return acc;
            if (array_internal::is_nan(r.first)) return r;
            return pair(r.first < acc.first ? r.first : acc.first, acc.second < r.second ? r.second : acc.second);
        });
    }

    /**
     * Flat index of the first smallest element (the first NaN if any). Throws on empty input.
     */
    template<class T>
    index_t argmin(array_view<const T> a) {
        return array_internal::arg_extremum(a, std::less<T>(), "argmin");
    }

    /**
     * Flat index of the first largest element (the first NaN if any). Throws on empty input.
     */
    template<class T>
    index_t argmax(array_view<const T> a) {
        return array_internal::arg_extremum(a, std::greater<T>(), "argmax");
    }

    namespace array_internal {

        /**
         * ``max(a)``, or ``std::numeric_limits<T>::lowest()`` if ``a`` is empty. NaN propagates
         * as in ``max``.
         */
        template<class T>
        T max_or_lowest(array_view<const T> a) {
            return a.size() == 0 ? std::numeric_limits<T>::lowest() : hg::max(a);
        }
    }

    /**
     * Sum of the elements, accumulated in ``Acc``: by default ``int64_t`` for ``bool`` and signed
     * integers, ``uint64_t`` for unsigned integers, and ``T`` for floating point types.
     *
     * The array is cut into chunks of ``HG_ARRAY_REDUCE_CHUNK`` elements; each chunk is summed with
     * 4 interleaved accumulators, and the chunk sums are added in chunk order. The result is
     * therefore identical with and without TBB and for any number of threads.
     */
    template<class Acc = void, class T>
    auto sum(array_view<const T> a) {
        using acc_t = std::conditional_t<std::is_void_v<Acc>, array_internal::sum_type_t<T>, Acc>;
        array_internal::assert_cpu_accessible(a.device());
        const T *data = a.data();
        return array_internal::reduce_chunks<acc_t>(
                a.size(),
                [data](index_t b, index_t e) { return array_internal::sum_range<acc_t>(data, b, e); },
                [](acc_t acc, acc_t r) { return acc + r; });
    }

    /**
     * Number of elements different from 0 (NaN counts as non-zero).
     */
    template<class T>
    index_t count_nonzero(array_view<const T> a) {
        array_internal::assert_cpu_accessible(a.device());
        const T *data = a.data();
        return array_internal::reduce_chunks<index_t>(
                a.size(),
                [data](index_t b, index_t e) {
                    index_t c = 0;
                    for (index_t i = b; i < e; ++i) {
                        c += data[i] != T(0) ? 1 : 0;
                    }
                    return c;
                },
                [](index_t acc, index_t r) { return acc + r; });
    }

    /**
     * True if every element is non-zero (true for an empty array).
     */
    template<class T>
    bool all(array_view<const T> a) {
        array_internal::assert_cpu_accessible(a.device());
        const T *data = a.data();
        return !array_internal::any_of(a.size(), [data](index_t i) { return data[i] == T(0); });
    }

    /**
     * True if at least one element is non-zero (false for an empty array).
     */
    template<class T>
    bool any(array_view<const T> a) {
        array_internal::assert_cpu_accessible(a.device());
        const T *data = a.data();
        return array_internal::any_of(a.size(), [data](index_t i) { return data[i] != T(0); });
    }

    // ------------------------------------------------------------------------------------------
    // Gather / scatter along the first axis
    // ------------------------------------------------------------------------------------------

    /**
     * ``a[idx]`` (NumPy): rows of ``a`` at the positions given by the 1d index array ``idx``.
     * The result has shape ``(idx.size(), a.shape[1:]...)``. Indices must lie in
     * ``[0, a.shape(0))``; negative indices are rejected.
     */
    template<class T, class I>
    array<T> take(array_view<const T> a, array_view<const I> idx) {
        array_internal::assert_cpu_accessible(a.device());
        array_internal::assert_cpu_accessible(idx.device());
        hg_assert(a.dimension() >= 1, "take: the array must have at least one dimension.");
        hg_assert(idx.dimension() == 1, "take: the index array must be 1d.");
        array_shape shape = a.shape();
        shape[0] = idx.size();
        array<T> result(shape);
        const index_t rs = a.row_size();
        const index_t n0 = a.shape(0);
        const T *in = a.data();
        const I *ip = idx.data();
        T *out = result.data();
        const index_t n = idx.size();
        array_internal::check_indices(ip, n, n0, "take");
        if (rs == 1) {
            detail::parallel_blocks(n, 4096, [=](index_t b, index_t e) {
                for (index_t i = b; i < e; ++i) {
                    out[i] = in[ip[i]];
                }
            });
        } else if (rs > 1) {
            const index_t grain = rs >= 4096 ? 1 : 4096 / rs;
            auto body = [=](index_t b, index_t e) {
                for (index_t i = b; i < e; ++i) {
                    const index_t j = static_cast<index_t>(ip[i]);
                    for (index_t k = 0; k < rs; ++k) {
                        out[i * rs + k] = in[j * rs + k];
                    }
                }
            };
            // the parallel decision depends on the number of copied elements, not of rows
            detail::parallel_blocks_if(detail::use_parallel(result.size()), n, grain, body);
        }
        return result;
    }

    namespace array_internal {

        // put with snapshots of the inputs that overlap the destination (defined after put)
        template<class T, class I>
        void put_from_snapshot(array_view<T> a, array_view<const I> idx, const T &value);

        template<class T, class I>
        void put_from_snapshot(array_view<T> a, array_view<const I> idx, array_view<const T> values);
    }

    /**
     * ``a[idx] = value`` (NumPy): sets the rows of ``a`` at the positions in the 1d index array
     * ``idx``. The result is NumPy's even when ``idx`` or ``value`` refers to elements of ``a``:
     * the indices and the value are read before any write. Writes are sequential; with duplicate
     * indices, the last write wins.
     */
    template<class T, class I>
    void put(array_view<T> a, array_view<const I> idx, const array_internal::identity_t<T> &value) {
        array_internal::assert_cpu_accessible(a.device());
        array_internal::assert_cpu_accessible(idx.device());
        hg_assert(a.dimension() >= 1, "put: the array must have at least one dimension.");
        hg_assert(idx.dimension() == 1, "put: the index array must be 1d.");
        const T v = value; // value may be an element of a
        T *out = a.data();
        if (array_internal::ranges_overlap(out, a.size(), idx.data(), idx.size())) {
            // indices stored in a: validated and read from a copy (separate function, so that
            // this common path stays small)
            array_internal::put_from_snapshot(a, idx, v);
            return;
        }
        const index_t rs = a.row_size();
        const index_t n0 = a.shape(0);
        array_internal::check_indices(idx.data(), idx.size(), n0, "put");
        for (index_t i = 0; i < idx.size(); ++i) {
            const index_t j = static_cast<index_t>(idx.data()[i]);
            for (index_t k = 0; k < rs; ++k) {
                out[j * rs + k] = v;
            }
        }
    }

    /**
     * ``a[idx] = values`` (NumPy): row ``i`` of ``values`` is written to row ``idx[i]`` of ``a``.
     * ``values`` must have shape ``(idx.size(), a.shape[1:]...)``. The result is NumPy's even when
     * ``idx`` or ``values`` overlaps ``a``: the indices and the values are read before any write.
     * Writes are sequential; with duplicate indices, the last write wins.
     */
    template<class T, class I>
    void put(array_view<T> a, array_view<const I> idx, array_view<const array_internal::identity_t<T>> values) {
        array_internal::assert_cpu_accessible(a.device());
        array_internal::assert_cpu_accessible(idx.device());
        array_internal::assert_cpu_accessible(values.device());
        hg_assert(a.dimension() >= 1, "put: the array must have at least one dimension.");
        hg_assert(idx.dimension() == 1, "put: the index array must be 1d.");
        array_shape expected = a.shape();
        expected[0] = idx.size();
        hg_assert(values.shape() == expected, "put: values must have shape " +
                                              array_internal::shape_to_string(expected) + ".");
        T *out = a.data();
        if (array_internal::ranges_overlap(out, a.size(), idx.data(), idx.size()) ||
            array_internal::ranges_overlap(out, a.size(), values.data(), values.size())) {
            // indices or values stored in a: copied before any write, the indices are validated
            // and read from the copy (separate function, so that this common path stays small)
            array_internal::put_from_snapshot(a, idx, values);
            return;
        }
        const index_t rs = a.row_size();
        const index_t n0 = a.shape(0);
        const T *in = values.data();
        array_internal::check_indices(idx.data(), idx.size(), n0, "put");
        for (index_t i = 0; i < idx.size(); ++i) {
            const index_t j = static_cast<index_t>(idx.data()[i]);
            for (index_t k = 0; k < rs; ++k) {
                out[j * rs + k] = in[i * rs + k];
            }
        }
    }

    namespace array_internal {

        // The copies do not overlap a, so the nested put takes its common path.
        template<class T, class I>
        void put_from_snapshot(array_view<T> a, array_view<const I> idx, const T &value) {
            const array<I> idx_copy(idx);
            put(a, array_view<const I>(idx_copy), value);
        }

        template<class T, class I>
        void put_from_snapshot(array_view<T> a, array_view<const I> idx, array_view<const T> values) {
            array<I> idx_copy;
            const I *ip = snapshot_if_overlap(idx, a.data(), a.size(), idx_copy);
            array<T> values_copy;
            const T *vp = snapshot_if_overlap(values, a.data(), a.size(), values_copy);
            put(a, array_view<const I>(ip, idx.shape()), array_view<const T>(vp, values.shape()));
        }
    }

    // ------------------------------------------------------------------------------------------
    // Comparison
    // ------------------------------------------------------------------------------------------

    inline bool same_shape(const array_shape &a, const array_shape &b) noexcept {
        return a == b;
    }

    template<class T, class U>
    bool same_shape(array_view<const T> a, array_view<const U> b) noexcept {
        return a.shape() == b.shape();
    }

    /**
     * True if ``a`` and ``b`` have the same shape and equal elements (element types may differ;
     * integers are compared by value regardless of signedness; NaN is never equal).
     */
    template<class T, class U>
    bool operator==(array_view<const T> a, array_view<const U> b) {
        if (a.shape() != b.shape()) {
            return false;
        }
        array_internal::assert_cpu_accessible(a.device());
        array_internal::assert_cpu_accessible(b.device());
        const T *pa = a.data();
        const U *pb = b.data();
        return !array_internal::any_of(a.size(), [pa, pb](index_t i) {
            return !array_internal::equal_values(pa[i], pb[i]);
        });
    }

    template<class T, class U>
    bool operator!=(array_view<const T> a, array_view<const U> b) {
        return !(a == b);
    }

    /**
     * True if ``a`` and ``b`` have the same shape and ``|a - b| <= atol + rtol * |b|`` for every
     * element (NumPy ``allclose`` with ``equal_nan=False``): any NaN makes the result false,
     * infinities are close only to the same infinity. Arrays of different shapes are not close
     * (there is no broadcasting).
     */
    template<class T, class U>
    bool allclose(array_view<const T> a, array_view<const U> b, double rtol = 1e-5, double atol = 1e-8) {
        if (a.shape() != b.shape()) {
            return false;
        }
        array_internal::assert_cpu_accessible(a.device());
        array_internal::assert_cpu_accessible(b.device());
        const T *pa = a.data();
        const U *pb = b.data();
        return !array_internal::any_of(a.size(), [pa, pb, rtol, atol](index_t i) {
            const double x = static_cast<double>(pa[i]);
            const double y = static_cast<double>(pb[i]);
            if (x == y) {
                return false;
            }
            if (!std::isfinite(x) || !std::isfinite(y)) {
                return true;
            }
            return !(std::fabs(x - y) <= atol + rtol * std::fabs(y));
        });
    }
}
