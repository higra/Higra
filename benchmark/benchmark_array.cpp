/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

// Built-in loops of the array library (higra/structure/array/) against plain loops and the
// xtensor equivalents: fill, copy (assign), cast, take, sum, max and map.
//
// Benchmarks named BM_<op>_<impl>/<n>, impl in {hg, loop, xt}. With HG_USE_TBB, the hg built-ins
// run in parallel from HG_ARRAY_PARALLEL_THRESHOLD elements and the xtensor assignments use
// XTENSOR_USE_TBB when it is defined; map uses hg::exec::par in BM_map_hgpar. The BM_*_hg
// benchmarks also cover sizes 2^10 ... 2^22 to locate the parallel threshold.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <random>

#include "higra/structure/array/all.hpp"
#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/core/xnoalias.hpp"
#include "xtensor/core/xoperation.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include "xtensor/views/xindex_view.hpp"

namespace {

    using hg::index_t;

    void main_sizes(benchmark::internal::Benchmark *b) {
        b->Arg(10000)->Arg(1000000)->Arg(100000000)->Unit(benchmark::kMicrosecond);
    }

    void sweep_sizes(benchmark::internal::Benchmark *b) {
        for (index_t n = 1 << 10; n <= (1 << 22); n *= 4) {
            b->Arg(n);
        }
        b->Arg(10000)->Arg(1000000)->Arg(100000000)->Unit(benchmark::kMicrosecond);
    }

    hg::array<double> random_doubles(index_t n) {
        auto a = hg::array<double>(hg::array_shape(n));
        std::mt19937_64 gen(1);
        std::uniform_real_distribution<double> d(0, 1);
        for (auto &x: a) x = d(gen);
        return a;
    }

    xt::xtensor<double, 1> to_xt(const hg::array<double> &a) {
        xt::xtensor<double, 1> r = xt::empty<double>({(std::size_t) a.size()});
        std::copy(a.begin(), a.end(), r.begin());
        return r;
    }

    hg::array<index_t> random_indices(index_t n) {
        auto a = hg::array<index_t>(hg::array_shape(n));
        std::mt19937_64 gen(2);
        std::uniform_int_distribution<index_t> d(0, n - 1);
        for (auto &x: a) x = d(gen);
        return a;
    }

    void set_bytes(benchmark::State &state, index_t n, std::size_t bytes_per_element) {
        state.SetBytesProcessed((int64_t) state.iterations() * n * (int64_t) bytes_per_element);
    }

    // ---- fill -------------------------------------------------------------------------------

    void BM_fill_hg(benchmark::State &state) {
        const index_t n = state.range(0);
        hg::array<double> a(hg::array_shape(n), 0.0);
        for (auto _: state) {
            a.fill(1.5);
            benchmark::ClobberMemory();
        }
        set_bytes(state, n, sizeof(double));
    }

    void BM_fill_loop(benchmark::State &state) {
        const index_t n = state.range(0);
        hg::array<double> a(hg::array_shape(n), 0.0);
        for (auto _: state) {
            double *p = a.data();
            for (index_t i = 0; i < n; ++i) p[i] = 1.5;
            benchmark::ClobberMemory();
        }
        set_bytes(state, n, sizeof(double));
    }

    void BM_fill_xt(benchmark::State &state) {
        const index_t n = state.range(0);
        xt::xtensor<double, 1> a = xt::zeros<double>({(std::size_t) n});
        for (auto _: state) {
            a.fill(1.5);
            benchmark::ClobberMemory();
        }
        set_bytes(state, n, sizeof(double));
    }

    // ---- copy -------------------------------------------------------------------------------

    void BM_copy_hg(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = random_doubles(n);
        hg::array<double> dst(hg::array_shape(n), 0.0);
        for (auto _: state) {
            hg::assign(dst, src);
            benchmark::ClobberMemory();
        }
        set_bytes(state, n, 2 * sizeof(double));
    }

    void BM_copy_loop(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = random_doubles(n);
        hg::array<double> dst(hg::array_shape(n), 0.0);
        for (auto _: state) {
            const double *s = src.data();
            double *d = dst.data();
            for (index_t i = 0; i < n; ++i) d[i] = s[i];
            benchmark::ClobberMemory();
        }
        set_bytes(state, n, 2 * sizeof(double));
    }

    void BM_copy_xt(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = to_xt(random_doubles(n));
        xt::xtensor<double, 1> dst = xt::zeros<double>({(std::size_t) n});
        for (auto _: state) {
            xt::noalias(dst) = src;
            benchmark::ClobberMemory();
        }
        set_bytes(state, n, 2 * sizeof(double));
    }

    // ---- cast (int64 -> double, result allocated) ------------------------------------------

    void BM_cast_hg(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = hg::arange<index_t>(n);
        for (auto _: state) {
            auto r = hg::cast<double>(src);
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 2 * sizeof(double));
    }

    void BM_cast_loop(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = hg::arange<index_t>(n);
        for (auto _: state) {
            auto r = hg::array<double>(hg::array_shape(n));
            const index_t *s = src.data();
            double *d = r.data();
            for (index_t i = 0; i < n; ++i) d[i] = (double) s[i];
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 2 * sizeof(double));
    }

    void BM_cast_xt(benchmark::State &state) {
        const index_t n = state.range(0);
        xt::xtensor<index_t, 1> src = xt::arange<index_t>(n);
        for (auto _: state) {
            xt::xtensor<double, 1> r = xt::cast<double>(src);
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 2 * sizeof(double));
    }

    // ---- take (random gather, result allocated) ---------------------------------------------

    void BM_take_hg(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = random_doubles(n);
        auto idx = random_indices(n);
        for (auto _: state) {
            auto r = hg::take(src, idx);
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 3 * sizeof(double));
    }

    void BM_take_loop(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = random_doubles(n);
        auto idx = random_indices(n);
        for (auto _: state) {
            auto r = hg::array<double>(hg::array_shape(n));
            const double *s = src.data();
            const index_t *ip = idx.data();
            double *d = r.data();
            for (index_t i = 0; i < n; ++i) d[i] = s[ip[i]];
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 3 * sizeof(double));
    }

    void BM_take_xt(benchmark::State &state) {
        const index_t n = state.range(0);
        auto src = to_xt(random_doubles(n));
        auto hidx = random_indices(n);
        xt::xtensor<index_t, 1> idx = xt::empty<index_t>({(std::size_t) n});
        std::copy(hidx.begin(), hidx.end(), idx.begin());
        for (auto _: state) {
            xt::xtensor<double, 1> r = xt::index_view(src, idx);
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 3 * sizeof(double));
    }

    // ---- sum --------------------------------------------------------------------------------

    void BM_sum_hg(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = random_doubles(n);
        for (auto _: state) {
            benchmark::DoNotOptimize(hg::sum(a));
        }
        set_bytes(state, n, sizeof(double));
    }

    void BM_sum_loop(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = random_doubles(n);
        for (auto _: state) {
            double s = 0;
            const double *p = a.data();
            for (index_t i = 0; i < n; ++i) s += p[i];
            benchmark::DoNotOptimize(s);
        }
        set_bytes(state, n, sizeof(double));
    }

    void BM_sum_xt(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = to_xt(random_doubles(n));
        for (auto _: state) {
            double s = xt::sum(a)();
            benchmark::DoNotOptimize(s);
        }
        set_bytes(state, n, sizeof(double));
    }

    // ---- max --------------------------------------------------------------------------------

    void BM_max_hg(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = random_doubles(n);
        for (auto _: state) {
            benchmark::DoNotOptimize(hg::max(a));
        }
        set_bytes(state, n, sizeof(double));
    }

    void BM_max_loop(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = random_doubles(n);
        for (auto _: state) {
            const double *p = a.data();
            double m = p[0];
            for (index_t i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
            benchmark::DoNotOptimize(m);
        }
        set_bytes(state, n, sizeof(double));
    }

    void BM_max_xt(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = to_xt(random_doubles(n));
        for (auto _: state) {
            double m = xt::amax(a)();
            benchmark::DoNotOptimize(m);
        }
        set_bytes(state, n, sizeof(double));
    }

    // ---- map: r = a * b + 1 (result allocated) ----------------------------------------------

    void BM_map_hg(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = random_doubles(n);
        auto b = random_doubles(n);
        for (auto _: state) {
            auto r = hg::map([](double x, double y) { return x * y + 1; }, a, b);
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 3 * sizeof(double));
    }

    void BM_map_hgpar(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = random_doubles(n);
        auto b = random_doubles(n);
        for (auto _: state) {
            auto r = hg::map(hg::exec::par, [](double x, double y) { return x * y + 1; }, a, b);
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 3 * sizeof(double));
    }

    void BM_map_loop(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = random_doubles(n);
        auto b = random_doubles(n);
        for (auto _: state) {
            auto r = hg::array<double>(hg::array_shape(n));
            const double *pa = a.data();
            const double *pb = b.data();
            double *d = r.data();
            for (index_t i = 0; i < n; ++i) d[i] = pa[i] * pb[i] + 1;
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 3 * sizeof(double));
    }

    void BM_map_xt(benchmark::State &state) {
        const index_t n = state.range(0);
        auto a = to_xt(random_doubles(n));
        auto b = to_xt(random_doubles(n));
        for (auto _: state) {
            xt::xtensor<double, 1> r = a * b + 1;
            benchmark::DoNotOptimize(r.data());
        }
        set_bytes(state, n, 3 * sizeof(double));
    }
}

BENCHMARK(BM_fill_hg)->Apply(sweep_sizes);
BENCHMARK(BM_fill_loop)->Apply(main_sizes);
BENCHMARK(BM_fill_xt)->Apply(main_sizes);
BENCHMARK(BM_copy_hg)->Apply(sweep_sizes);
BENCHMARK(BM_copy_loop)->Apply(main_sizes);
BENCHMARK(BM_copy_xt)->Apply(main_sizes);
BENCHMARK(BM_cast_hg)->Apply(sweep_sizes);
BENCHMARK(BM_cast_loop)->Apply(main_sizes);
BENCHMARK(BM_cast_xt)->Apply(main_sizes);
BENCHMARK(BM_take_hg)->Apply(sweep_sizes);
BENCHMARK(BM_take_loop)->Apply(main_sizes);
BENCHMARK(BM_take_xt)->Apply(main_sizes);
BENCHMARK(BM_sum_hg)->Apply(sweep_sizes);
BENCHMARK(BM_sum_loop)->Apply(main_sizes);
BENCHMARK(BM_sum_xt)->Apply(main_sizes);
BENCHMARK(BM_max_hg)->Apply(sweep_sizes);
BENCHMARK(BM_max_loop)->Apply(main_sizes);
BENCHMARK(BM_max_xt)->Apply(main_sizes);
BENCHMARK(BM_map_hg)->Apply(main_sizes);
BENCHMARK(BM_map_hgpar)->Apply(sweep_sizes);
BENCHMARK(BM_map_loop)->Apply(main_sizes);
BENCHMARK(BM_map_xt)->Apply(main_sizes);
