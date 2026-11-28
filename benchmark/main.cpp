/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include <benchmark/benchmark.h>
#include "higra/parallel.hpp"

namespace {
    /**
     * Parse the value of HG_NUM_THREADS: decimal digits, surrounding whitespace allowed, value between 1 and
     * the largest index_t. Return 0 if the value is invalid.
     */
    hg::index_t parse_num_threads(const std::string &value) {
        auto is_space = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
        std::size_t begin = 0;
        std::size_t end = value.size();
        while (begin < end && is_space(value[begin])) ++begin;
        while (end > begin && is_space(value[end - 1])) --end;
        if (begin == end) {
            return 0;
        }
        const hg::index_t max = std::numeric_limits<hg::index_t>::max();
        hg::index_t result = 0;
        for (std::size_t i = begin; i < end; ++i) {
            const char c = value[i];
            if (c < '0' || c > '9') {
                return 0;
            }
            const hg::index_t digit = c - '0';
            if (result > (max - digit) / 10) {
                return 0;
            }
            result = result * 10 + digit;
        }
        return result;
    }

    bool is_blank(const std::string &value) {
        for (char c: value) {
            if (!std::isspace(static_cast<unsigned char>(c))) {
                return false;
            }
        }
        return true;
    }
}

// Custom main function: the environment variable HG_NUM_THREADS (positive integer) limits the number of threads
int main(int argc, char **argv) {
    const char *env_p = std::getenv("HG_NUM_THREADS");
    if (env_p != nullptr && !is_blank(env_p)) {
        const hg::index_t num_threads = parse_num_threads(env_p);
        if (num_threads > 0) {
            hg::set_num_threads(num_threads);
        } else {
            std::cerr << "Ignoring HG_NUM_THREADS='" << env_p
                      << "': it must be a positive integer smaller than 2**63.\n";
        }
    }

    std::cout << "Higra parallelism constrained to: " << hg::get_num_threads() << " thread(s)\n\n";

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    benchmark::RunSpecifiedBenchmarks();
}
