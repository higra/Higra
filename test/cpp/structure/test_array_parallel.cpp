/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Array library on inputs above HG_ARRAY_PARALLEL_THRESHOLD: results must not depend on the
// use of TBB or on the number of threads.

#include "test_array_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>

#ifdef HG_USE_TBB

#include <tbb/global_control.h>

#endif

namespace test_array_parallel {

    using hg::array;
    using hg::array_shape;
    using hg::array_view;
    using hg::index_t;

    const index_t big = 3 * HG_ARRAY_PARALLEL_THRESHOLD + 17;

    // Reference implementation of the documented floating point sum: chunks of
    // HG_ARRAY_REDUCE_CHUNK elements, 4 interleaved accumulators per chunk combined as
    // (s0 + s1) + (s2 + s3), chunk sums added in order.
    template<class T>
    T reference_chunked_sum(const std::vector<T> &v) {
        const index_t n = (index_t) v.size();
        const index_t chunk = HG_ARRAY_REDUCE_CHUNK;
        T total = 0;
        for (index_t b = 0; b < std::max<index_t>(n, 1); b += chunk) {
            const index_t e = std::min(n, b + chunk);
            T s[4] = {0, 0, 0, 0};
            for (index_t i = b; i < e; ++i) {
                s[(i - b) % 4] += v[i];
            }
            const T cs = (s[0] + s[1]) + (s[2] + s[3]);
            total = b == 0 ? cs : total + cs;
        }
        return total;
    }

    template<class T>
    std::vector<T> random_values(index_t n, unsigned seed) {
        std::mt19937_64 gen(seed);
        std::uniform_real_distribution<double> mantissa(-1, 1);
        std::uniform_int_distribution<int> exponent(-20, 20);
        std::vector<T> v((std::size_t) n);
        for (auto &x: v) {
            x = (T) std::ldexp(mantissa(gen), exponent(gen));
        }
        return v;
    }

    template<class T>
    bool same_bits(T a, T b) {
        return std::memcmp(&a, &b, sizeof(T)) == 0;
    }

    TEST_CASE("array parallel_blocks covers the range once", "[array][array_parallel]") {
        for (index_t n: {index_t(0), index_t(1), index_t(1000), big}) {
            std::vector<std::atomic<int>> hits((std::size_t) n);
            hg::detail::parallel_blocks(n, 128, [&hits](index_t b, index_t e) {
                for (index_t i = b; i < e; ++i) hits[(std::size_t) i]++;
            });
            REQUIRE(std::all_of(hits.begin(), hits.end(), [](const std::atomic<int> &h) { return h == 1; }));
        }
        std::vector<std::atomic<int>> chunks(10);
        const index_t n = 9 * HG_ARRAY_REDUCE_CHUNK + 1;
        const index_t num = hg::detail::for_each_chunk(n, [&chunks, n](index_t c, index_t b, index_t e) {
            REQUIRE(b == c * HG_ARRAY_REDUCE_CHUNK);
            REQUIRE(e == std::min(n, b + HG_ARRAY_REDUCE_CHUNK));
            chunks[(std::size_t) c]++;
        });
        REQUIRE(num == 10);
        REQUIRE(std::all_of(chunks.begin(), chunks.end(), [](const std::atomic<int> &h) { return h == 1; }));
    }

    TEST_CASE("array built-in loops above the parallel threshold", "[array][array_parallel]") {
        auto a = hg::arange<index_t>(big);
        REQUIRE(a.size() == big);
        bool ok = true;
        for (index_t i = 0; i < big; ++i) ok = ok && a(i) == i;
        REQUIRE(ok);

        auto z = hg::zeros<double>({big});
        REQUIRE(hg::count_nonzero(z) == 0);
        auto f = hg::full<int>({big, 2}, 3);
        REQUIRE(hg::sum(f) == 6 * big);
        f.fill(1);
        REQUIRE(hg::all(f));

        array<index_t> c(a); // deep copy
        REQUIRE(c == a);
        auto d = hg::cast<double>(a);
        REQUIRE(hg::max(d) == (double) (big - 1));
        REQUIRE(hg::allclose(d, a));
        auto e = array<int>(array_shape(big));
        hg::assign(e, d);
        REQUIRE(e == a);
        e(big - 2) = -1;
        REQUIRE(e != a);
        REQUIRE(!hg::allclose(e, a));
        REQUIRE(hg::argmin(e) == big - 2);
        REQUIRE(hg::minmax(e) == std::pair<int, int>(-1, (int) (big - 1)));
        REQUIRE(hg::any(z) == false);

        // gather of scalars and of rows
        auto idx = hg::map([](index_t i) { return (big - 1 - i); }, a);
        auto t = hg::take(a, idx);
        REQUIRE(t == idx);
        array<int> rows(array_shape(big, 3));
        for (index_t i = 0; i < big; ++i)
            for (index_t k = 0; k < 3; ++k) rows(i, k) = (int) (i * 3 + k);
        auto tr = hg::take(rows, idx);
        REQUIRE(tr.shape() == array_shape(big, 3));
        REQUIRE(tr(0, 2) == (int) ((big - 1) * 3 + 2));
        REQUIRE(tr(big - 1, 1) == 1);
        REQUIRE_THROWS(hg::take(a, hg::full<index_t>({big}, big)));

        // exec::par map gives the same result as the sequential map
        auto sq = [](index_t x) { return x * x; };
        REQUIRE(hg::map(hg::exec::par, sq, a) == hg::map(sq, a));
    }

    TEST_CASE("array put with aliasing inputs above the parallel threshold", "[array][array_parallel]") {
        // reference: NumPy semantics, a[idx] = values with idx and values read before any write
        auto reference = [](std::vector<index_t> a, const std::vector<index_t> &idx,
                            const std::vector<index_t> &vals) {
            for (std::size_t i = 0; i < idx.size(); ++i) a[(std::size_t) idx[i]] = vals[i];
            return a;
        };
        auto check = [&]() {
            // values are the destination, reversed through the indices
            auto a = hg::arange<index_t>(big);
            const auto rev = hg::map([](index_t i) { return big - 1 - i; }, a);
            hg::put(a, rev, a);
            REQUIRE(a == rev);

            // indices are stored in the first half of the destination and point into it; the
            // second half must stay untouched
            const index_t half = big / 2;
            auto b = array<index_t>(array_shape(big));
            for (index_t i = 0; i < big; ++i) b(i) = i < half ? (i * 7919) % half : -5;
            const std::vector<index_t> b0(b.begin(), b.end());
            auto head = b.slice(0, half);
            const std::vector<index_t> idx0(b0.begin(), b0.begin() + half);
            hg::put(head, head, (index_t) 3);
            auto expected = b0;
            for (index_t i = 0; i < half; ++i) expected[(std::size_t) idx0[(std::size_t) i]] = 3;
            REQUIRE(std::vector<index_t>(b.begin(), b.end()) == expected);

            // values shifted by one element, duplicate indices (last write wins)
            auto c = hg::arange<index_t>(big);
            const std::vector<index_t> c0(c.begin(), c.end());
            std::vector<index_t> cidx((std::size_t) (big - 1));
            for (index_t i = 0; i < big - 1; ++i) cidx[(std::size_t) i] = (i * 31) % (big / 3);
            const std::vector<index_t> cvals(c0.begin() + 1, c0.end());
            hg::put(c, array_view<const index_t>(cidx.data(), {big - 1}), c.slice(1, big));
            REQUIRE(std::vector<index_t>(c.begin(), c.end()) == reference(c0, cidx, cvals));
        };
        check();
#ifdef HG_USE_TBB
        for (int threads: {1, 2, 8}) {
            tbb::global_control limit(tbb::global_control::max_allowed_parallelism, (std::size_t) threads);
            INFO("threads: " << threads);
            check();
        }
#endif
    }

    TEST_CASE("array sum is deterministic and bit-identical to the reference", "[array][array_parallel]") {
        const auto v = random_values<double>(big, 42);
        const array<double> a(array_view<const double>(v.data(), {big}));
        const double ref = reference_chunked_sum(v);
        REQUIRE(same_bits(hg::sum(a), ref));

        const auto vf = random_values<float>(big + 5, 7);
        const array<float> af(array_view<const float>(vf.data(), {(index_t) vf.size()}));
        REQUIRE(same_bits(hg::sum(af), reference_chunked_sum(vf)));

        // small inputs (single chunk) follow the same rule
        const auto vs = random_values<double>(1001, 3);
        REQUIRE(same_bits(hg::sum(array_view<const double>(vs.data(), {1001})), reference_chunked_sum(vs)));

#ifdef HG_USE_TBB
        for (int threads: {1, 2, 8}) {
            tbb::global_control limit(tbb::global_control::max_allowed_parallelism, (std::size_t) threads);
            INFO("threads: " << threads);
            REQUIRE(same_bits(hg::sum(a), ref));
            REQUIRE(same_bits(hg::sum(af), reference_chunked_sum(vf)));
        }
#endif
    }

    TEST_CASE("array argmin/argmax return the first index under ties", "[array][array_parallel]") {
        array<int> a(array_shape(big), 0);
        const index_t first = 2 * HG_ARRAY_PARALLEL_THRESHOLD + 3;
        for (index_t i: {big - 1, first + HG_ARRAY_REDUCE_CHUNK, first, first + 1}) {
            a(i) = -7;
        }
        for (index_t i: {big - 2, index_t(HG_ARRAY_REDUCE_CHUNK + 1), index_t(5 * HG_ARRAY_REDUCE_CHUNK)}) {
            a(i) = 7;
        }
        auto check = [&]() {
            REQUIRE(hg::argmin(a) == first);
            REQUIRE(hg::argmax(a) == HG_ARRAY_REDUCE_CHUNK + 1);
            REQUIRE(hg::min(a) == -7);
            REQUIRE(hg::max(a) == 7);
        };
        check();
#ifdef HG_USE_TBB
        for (int threads: {1, 2, 8}) {
            tbb::global_control limit(tbb::global_control::max_allowed_parallelism, (std::size_t) threads);
            INFO("threads: " << threads);
            check();
        }
#endif
    }

    TEST_CASE("array map without policy runs on the calling thread", "[array][array_parallel]") {
        const auto caller = std::this_thread::get_id();
        auto a = hg::arange<index_t>(big);
        std::set<std::thread::id> seen;
        auto r = hg::map([&seen](index_t x) {
            seen.insert(std::this_thread::get_id()); // not thread-safe: must stay sequential
            return x + 1;
        }, a);
        REQUIRE(seen.size() == 1);
        REQUIRE(*seen.begin() == caller);
        REQUIRE(r(big - 1) == big);

        seen.clear();
        auto out = array<index_t>(array_shape(big));
        hg::map_into(out, [&seen](index_t x, index_t y) {
            seen.insert(std::this_thread::get_id());
            return x + y;
        }, a, a);
        REQUIRE(seen.size() == 1);
        REQUIRE(*seen.begin() == caller);

        seen.clear();
        hg::map(hg::exec::seq, [&seen](index_t x) {
            seen.insert(std::this_thread::get_id());
            return x;
        }, a);
        REQUIRE(seen.size() == 1);

        // exec::par: may use other threads; the function must be thread-safe
        std::mutex m;
        std::set<std::thread::id> seen_par;
        auto rp = hg::map(hg::exec::par, [&](index_t x) {
            std::lock_guard<std::mutex> lock(m);
            seen_par.insert(std::this_thread::get_id());
            return x + 1;
        }, a);
        REQUIRE(rp == r);
        REQUIRE(!seen_par.empty());
    }
}
