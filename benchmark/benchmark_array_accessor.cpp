//
// Created by perretb on 28/05/18.
//


//
// Created by user on 5/11/18.
//

#include <benchmark/benchmark.h>

#include "higra/graph.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xrandom.hpp"

using namespace xt;
using namespace hg;

std::size_t min_array_size = 10;
std::size_t max_array_size = 16;
std::size_t max_array2d_size = 12;


template<typename T>
int sum1d_fori(const xt::xexpression<T> &xa) {
    auto a = xa.derived_cast();
    auto s = a.shape();
    int sum = 0;
    for (index_t i = 0; i < (index_t)s[0]; ++i) {
        sum += a(i);
    }
    return sum;
}

template<typename T>
int sum2d_fori(const xt::xexpression<T> &xa) {
    auto a = xa.derived_cast();
    auto s = a.shape();
    int sum = 0;
    for (index_t i = 0; i < (index_t)s[0]; ++i) {
        for (index_t j = 0; j < (index_t)s[1]; ++j)
            sum += a(i, j);
    }
    return sum;
}

template<typename T>
int sum_it(const T &a) {
    int sum = 0;
    for (auto v = a.begin(); v != a.end(); v++)
        sum += *v;
    return sum;
}

static void BM_1Darray_access_cstyle_fori(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        std::vector<int> array;
        array.resize(size, 1);
        state.ResumeTiming();
        int sum = 0;
        for (long i = 0; i < size; ++i)
            sum += array[i];
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_1Darray_access_cstyle_fori)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_1Darray_access_cstyle_iter(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        std::vector<int> array;
        array.resize(size, 1);
        state.ResumeTiming();
        int sum = sum_it(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_1Darray_access_cstyle_iter)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_1Darray_access_xtensor_fori(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xtensor<int, 1> array({(ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum1d_fori(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_1Darray_access_xtensor_fori)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_1Darray_access_xtensor_iter(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xtensor<int, 1> array({(ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum_it(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_1Darray_access_xtensor_iter)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_1Darray_access_xarray_fori(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xarray<int> array({(ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum1d_fori(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_1Darray_access_xarray_fori)->Range(1 << min_array_size, 1 << max_array_size);

static void BM_1Darray_access_xarray_iter(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xarray<int> array({(ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum_it(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_1Darray_access_xarray_iter)->Range(1 << min_array_size, 1 << max_array_size);


static void BM_2Darray_access_cstyle_fori(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        std::vector<int> array;
        array.resize(size * size, 1);
        state.ResumeTiming();
        int sum = 0;
        for (long i = 0; i < size; ++i) {
            for (long j = 0; j < size; ++j)
                sum += array[i * size + j];
        }
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_2Darray_access_cstyle_fori)->Range(1 << min_array_size, 1 << max_array2d_size);

static void BM_2Darray_access_cstyle_iter(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        std::vector<int> array;
        array.resize(size * size, 1);
        state.ResumeTiming();
        int sum = sum_it(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_2Darray_access_cstyle_iter)->Range(1 << min_array_size, 1 << max_array2d_size);

static void BM_2Darray_access_xtensor_fori(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xtensor<int, 2> array({(ulong) size, (ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum2d_fori(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_2Darray_access_xtensor_fori)->Range(1 << min_array_size, 1 << max_array2d_size);

static void BM_2Darray_access_xtensor_iter(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xtensor<int, 2> array({(ulong) size, (ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum_it(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_2Darray_access_xtensor_iter)->Range(1 << min_array_size, 1 << max_array2d_size);

static void BM_2Darray_access_xarray_fori(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xarray<int> array({(ulong) size, (ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum2d_fori(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_2Darray_access_xarray_fori)->Range(1 << min_array_size, 1 << max_array2d_size);

static void BM_2Darray_access_xarray_iter(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();

        int size = state.range(0);
        xt::xarray<int> array({(ulong) size, (ulong) size}, 1);
        state.ResumeTiming();
        int sum = sum_it(array);
        bool flag;
        benchmark::DoNotOptimize(flag = (sum == size));
    }
}

BENCHMARK(BM_2Darray_access_xarray_iter)->Range(1 << min_array_size, 1 << max_array2d_size);