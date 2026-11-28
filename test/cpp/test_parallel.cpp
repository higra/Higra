/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "higra/parallel.hpp"
#include "higra/sorting.hpp"
#include "higra/structure/array.hpp"
#include "test_utils.hpp"
#include <cstring>
#include <random>
#include <vector>

#ifdef HG_USE_TBB
    #include <oneapi/tbb/info.h>
#endif

namespace test_parallel {

    using namespace hg;

    struct reset_num_threads {
        ~reset_num_threads() {
            set_num_threads(0);
        }
    };

    TEST_CASE("set and get the number of threads", "[parallel]") {
        reset_num_threads guard;
        set_num_threads(0);
        const index_t default_threads = get_num_threads();
#ifdef HG_USE_TBB
        REQUIRE(default_threads == static_cast<index_t>(oneapi::tbb::info::default_concurrency()));
        set_num_threads(1);
        REQUIRE(get_num_threads() == 1);
        set_num_threads(2);
        REQUIRE(get_num_threads() == 2);
        // a new limit replaces the previous one, also when it is larger
        set_num_threads(default_threads + 1);
        REQUIRE(get_num_threads() == default_threads + 1);
#else
        REQUIRE(default_threads == 1);
        set_num_threads(1);
        REQUIRE(get_num_threads() == 1);
        set_num_threads(4);
        REQUIRE(get_num_threads() == 1);
#endif
        set_num_threads(0);
        REQUIRE(get_num_threads() == default_threads);
    }

    TEST_CASE("set the number of threads rejects negative values", "[parallel]") {
        reset_num_threads guard;
        set_num_threads(2);
        const index_t before = get_num_threads();
        REQUIRE_THROWS(set_num_threads(-1));
        REQUIRE(get_num_threads() == before);
    }

    TEST_CASE("results do not depend on the number of threads", "[parallel]") {
        reset_num_threads guard;
        const index_t n = 3 * HG_ARRAY_PARALLEL_THRESHOLD + 17;
        std::mt19937 gen(7);
        std::uniform_real_distribution<double> dist(-1, 1);
        std::uniform_int_distribution<int> idist(0, 100);
        array_1d<double> values = array_1d<double>::from_shape({n});
        array_1d<int> keys = array_1d<int>::from_shape({n});
        for (index_t i = 0; i < n; ++i) {
            values(i) = dist(gen);
            keys(i) = idist(gen);
        }

        auto compute = [&]() {
            return std::make_pair(sum(values), stable_arg_sort(keys));
        };

        set_num_threads(0);
        const auto reference = compute();
        for (index_t threads: {1, 2, 8}) {
            INFO("threads: " << threads);
            set_num_threads(threads);
            const auto result = compute();
            REQUIRE(std::memcmp(&result.first, &reference.first, sizeof(double)) == 0);
            REQUIRE(result.second == reference.second);
        }
    }
}
