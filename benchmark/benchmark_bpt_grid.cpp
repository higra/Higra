/***************************************************************************
* Copyright ESIEE Paris (2026)                                              *
* Distributed under the terms of the CECILL-B License.                      *
* The full license is in the file LICENSE, distributed with this software.  *
****************************************************************************/

#include "bpt_cases.hpp"
#include <benchmark/benchmark.h>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <tuple>
#include <sstream>
#ifdef HG_USE_TBB
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/version.h>
#endif

namespace grid_bench {
    namespace core = hg::hierarchy_core_internal;
    using hg::index_t;
    using hg::num_edges;
    using result_t = hg::node_weighted_tree_and_mst<hg::tree, hg::array_1d<double>>;
    using clock = std::chrono::steady_clock;

    // The three policies use the SAME frontend and a compile-time scalar
    // comparator. No method-name or rank/scalar dispatch runs per comparison.
    template<core::bpt_boruvka_numbering numbering, core::bpt_boruvka_tie_order ties>
    result_t boruvka(const hg::ugraph &graph, const hg::array_1d<double> &weights) {
        hg_assert_edge_weights(graph, weights);
        hg_assert_1d_array(weights);
        const index_t n = hg::num_vertices(graph);
        core::bpt_boruvka_check_sizes(n, weights.size());
        for (const auto w: weights) hg_assert(w == w, "NaN edge weights are not supported.");
        const auto &src = hg::sources(graph), &dst = hg::targets(graph);
        hg_assert(src.size() == weights.size(), "Endpoint and weight counts must match.");
        auto less = [&](index_t a, index_t b) {
            return core::bpt_boruvka_scalar_less(weights, a, b, ties);
        };
        auto result = core::bpt_boruvka_from_edge_order(src, dst, n, less, numbering);
        hg::array_1d<double> levels = xt::zeros<double>({result.first.size()});
        hg::parfor(0, n - 1, [&](index_t i) { levels(n + i) = weights(result.second(i)); });
        return hg::make_node_weighted_tree_and_mst(hg::tree(std::move(result.first)),
                                                  std::move(levels), std::move(result.second));
    }

    result_t call(const hg::ugraph &graph, const hg::array_1d<double> &weights,
                  const std::string &method) {
        if (method == "kruskal") return hg::bpt_canonical(graph, weights);
        if (method == "exact") return boruvka<core::bpt_boruvka_numbering::exact,
            core::bpt_boruvka_tie_order::edge_id>(graph, weights);
        if (method == "basin") return boruvka<core::bpt_boruvka_numbering::basin,
            core::bpt_boruvka_tie_order::edge_id>(graph, weights);
        hg_assert(method == "reverse", "Unknown method.");
        return boruvka<core::bpt_boruvka_numbering::basin,
            core::bpt_boruvka_tie_order::reverse_edge_id>(graph, weights);
    }

    result_t reference(const hg::ugraph &graph, const hg::array_1d<double> &weights,
                       bool reverse) {
        if (!reverse) return hg::bpt_canonical(graph, weights);
        hg::array_1d<index_t> order = xt::arange<index_t>(weights.size());
        hg::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
            return core::bpt_boruvka_scalar_less(weights, a, b, core::bpt_boruvka_tie_order::reverse_edge_id);
        });
        auto r = core::bpt_canonical_from_sorted_edges(hg::sources(graph), hg::targets(graph), order,
                                                     hg::num_vertices(graph));
        hg::array_1d<double> levels = xt::zeros<double>({r.first.size()});
        const index_t n = hg::num_vertices(graph);
        for (index_t i = 0; i < n - 1; ++i) levels(n + i) = weights(r.second(i));
        return hg::make_node_weighted_tree_and_mst(hg::tree(std::move(r.first)),
                                                  std::move(levels), std::move(r.second));
    }

    void validate(const hg::ugraph &graph, const hg::array_1d<double> &weights,
                  const result_t &got, const result_t &ref, bool exact) {
        const index_t n = hg::num_vertices(graph), count = 2 * n - 1;
        const auto &p = got.tree.parents(), &q = ref.tree.parents();
        hg_assert(p.size() == size_t(count) && got.mst_edge_map.size() == size_t(n - 1), "Wrong output size.");
        hg_assert(p(count - 1) == count - 1, "Wrong root.");
        std::vector<index_t> edge_to_node(weights.size(), hg::invalid_index), remap(count), degree(count, 0);
        for (index_t i = 0; i < n; ++i) remap[i] = i;
        for (index_t i = 0; i < n - 1; ++i) edge_to_node.at(ref.mst_edge_map(i)) = n + i;
        for (index_t i = 0; i < n - 1; ++i) {
            const auto node = edge_to_node.at(got.mst_edge_map(i));
            hg_assert(node != hg::invalid_index, "Different MST set.");
            remap[n + i] = node;
        }
        for (index_t i = 0; i < count; ++i) {
            hg_assert(p(i) >= i && p(i) < count && (i == count - 1 || p(i) > i), "Invalid topological parent.");
            if (i < count - 1) ++degree[p(i)];
            hg_assert(remap[p(i)] == q(remap[i]), "Edge-labeled parent mismatch.");
            hg_assert(got.altitudes(i) == ref.altitudes(remap[i]), "Altitude mismatch.");
            if (exact) hg_assert(remap[i] == i && p(i) == q(i), "Exact output mismatch.");
        }
        for (index_t i = n; i < count; ++i) hg_assert(degree[i] == 2, "Non-binary output.");
    }

    uint64_t mix(uint64_t hash, uint64_t word) {
        for (int byte = 0; byte < 8; ++byte) {
            hash ^= word & 255; hash *= 1099511628211ULL; word >>= 8;
        }
        return hash;
    }
    uint64_t bits(double value) {
        uint64_t result; std::memcpy(&result, &value, sizeof(result)); return result;
    }
    uint64_t digest_result(const result_t &result) {
        uint64_t hash = 14695981039346656037ULL;
        for (auto v: result.tree.parents()) hash = mix(hash, v);
        for (auto v: result.mst_edge_map) hash = mix(hash, v);
        for (auto v: result.altitudes) hash = mix(hash, bits(v));
        return hash;
    }
    std::string hex(uint64_t value) {
        std::ostringstream stream; stream << std::hex << std::setw(16) << std::setfill('0') << value;
        return stream.str();
    }

    void self_test() {
        for (const hg::array_1d<double> &weights: std::vector<hg::array_1d<double>>{
                {4, 3, 10, 1}, {1, 1, 1, 1}, {10, 1, 9, 2},
                {-std::numeric_limits<double>::infinity(), 0, 0, std::numeric_limits<double>::infinity()}}) {
            hg::ugraph graph(5);
            for (index_t i = 0; i < 4; ++i) hg::add_edge(i, i + 1, graph);
            for (const std::string method: {"exact", "basin", "reverse"}) {
                const auto got = call(graph, weights, method);
                const auto ref = reference(graph, weights, method == "reverse");
                validate(graph, weights, got, ref, method == "exact");
                if (method == "exact") {
                    const auto api = hg::bpt_canonical_boruvka(graph, weights);
                    validate(graph, weights, got, api, true);
                }
            }
        }
        // Tied cycle: reversing ties can change the MST, not just node IDs.
        hg::ugraph cycle(4);
        for (index_t i = 0; i < 4; ++i) hg::add_edge(i, (i + 1) % 4, cycle);
        hg::array_1d<double> tied = {1, 1, 1, 1};
        for (const std::string method: {"exact", "basin", "reverse"}) {
            const auto got = call(cycle, tied, method), ref = reference(cycle, tied, method == "reverse");
            validate(cycle, tied, got, ref, method == "exact");
        }
        hg::array_1d<double> nan = tied; nan(0) = std::numeric_limits<double>::quiet_NaN();
        for (const std::string method: {"exact", "basin", "reverse"}) {
            bool rejected = false;
            try { call(cycle, nan, method); } catch (const std::runtime_error &) { rejected = true; }
            hg_assert(rejected, "NaN was not rejected.");
        }
        std::cout << "{\"self_test\":\"passed\",\"fixtures\":5,\"policies\":3,\"nan_rejections\":3}\n";
    }
}

int main(int argc, char **argv) {
    using namespace grid_bench;
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string name = argv[i];
            if (name == "--self-test") { self_test(); return 0; }
            hg_assert(name.rfind("--", 0) == 0 && i + 1 < argc, "Expected --option value.");
            args[name] = argv[++i];
        }
        auto option = [&](const std::string &key, const std::string &fallback) {
            auto it = args.find(key); return it == args.end() ? fallback : it->second;
        };
        const index_t side = std::stoll(option("--side", "128"));
        const auto method = option("--method", "exact");
        const size_t threads = std::stoull(option("--threads", "1"));
        const uint64_t seed = std::stoull(option("--seed", "1729"));
        const int repetitions = std::stoi(option("--repetitions", "5"));
        const double min_time = std::stod(option("--min-time", "0.5"));
        const double warmup = std::stod(option("--warmup", "0.2"));
        const double max_mib = std::stod(option("--max-mib", "8192"));
        hg_assert(side >= 2 && side <= 8192 && threads > 0 && repetitions > 0 && min_time > 0 && warmup >= 0,
                  "Invalid size, threads or timing parameters.");
        hg_assert(method == "kruskal" || method == "exact" || method == "basin" || method == "reverse", "Unknown method.");
#ifdef HG_USE_TBB
        const size_t default_concurrency = oneapi::tbb::this_task_arena::max_concurrency();
        oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, threads);
        const auto active_limit = oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism);
#else
        hg_assert(threads == 1, "TBB-off runner requires --threads 1.");
        const size_t default_concurrency = 1, active_limit = 1;
#endif
        const index_t n = side * side, m = 2 * side * (side - 1);
        hg_assert(512.0 * n + 64.0 * m < max_mib * 1048576, "Case exceeds memory preflight bound.");
        // Grid vertex IDs are row-major. Append right edge, then down edge.
        hg::ugraph graph(n, m, 4);
        hg::array_1d<double> weights = xt::empty<double>({m});
        bpt_bench::rng random{seed ? seed : 1};
        uint64_t input_hash = mix(mix(14695981039346656037ULL, n), m);
        index_t edge = 0;
        auto add = [&](index_t a, index_t b) {
            hg::add_edge(a, b, graph);
            weights(edge) = double(random() >> 11) / 9007199254740992.0;
            input_hash = mix(mix(mix(input_hash, a), b), bits(weights(edge)));
            ++edge;
        };
        for (index_t v = 0; v < n; ++v) {
            if (v % side + 1 < side) add(v, v + 1);
            if (v / side + 1 < side) add(v, v + side);
        }
        hg_assert(edge == m, "Wrong grid edge count.");
        std::cerr << "Generated " << side << "x" << side << "; " << method << "; threads=" << active_limit << "\n";
        const auto validation_start = clock::now();
        uint64_t output_hash;
        { // Reference and validation arrays die before warmup/primary timings.
            const auto got = call(graph, weights, method);
            const auto ref = reference(graph, weights, method == "reverse");
            validate(graph, weights, got, ref, method == "kruskal" || method == "exact");
            output_hash = digest_result(got);
        }
        const auto validation_seconds = std::chrono::duration<double>(clock::now() - validation_start).count();
        std::cerr << "Validated in " << validation_seconds << " s\n";
        auto batch = [&](double duration) {
            const auto cpu_start = std::clock();
            const auto start = clock::now();
            size_t iterations = 0;
            double elapsed;
            do {
                { auto result = call(graph, weights, method);
                  benchmark::DoNotOptimize(result.tree.parents().data());
                  benchmark::DoNotOptimize(result.altitudes.data());
                  benchmark::DoNotOptimize(result.mst_edge_map.data());
                  benchmark::ClobberMemory(); }
                ++iterations;
                elapsed = std::chrono::duration<double>(clock::now() - start).count();
            } while (elapsed < duration);
            return std::make_tuple(elapsed, iterations, double(std::clock() - cpu_start) / CLOCKS_PER_SEC);
        };
        if (warmup > 0) batch(warmup);
        std::vector<double> seconds, cpu_seconds;
        std::vector<size_t> iterations;
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            const auto sample = batch(min_time);
            seconds.push_back(std::get<0>(sample) / std::get<1>(sample));
            iterations.push_back(std::get<1>(sample));
            cpu_seconds.push_back(std::get<2>(sample) / std::get<1>(sample));
            std::cerr << "Sample " << repetition + 1 << ": " << seconds.back() << " s/call\n";
        }
        std::ofstream file;
        auto *out = &std::cout;
        if (args.count("--out")) { file.open(args["--out"]); hg_assert(bool(file), "Cannot open output."); out = &file; }
        *out << std::setprecision(17) << "{\"schema\":1,\"side\":" << side << ",\"vertices\":" << n << ",\"edges\":" << m
             << ",\"seed\":" << seed << ",\"method\":\"" << method << "\",\"requested_threads\":" << threads
             << ",\"active_thread_limit\":" << active_limit << ",\"default_concurrency\":" << default_concurrency;
#ifdef HG_USE_TBB
        *out << ",\"tbb\":true,\"tbb_version\":\"" << TBB_VERSION_STRING << "\"";
#else
        *out << ",\"tbb\":false";
#endif
        *out << ",\"weights\":\"uniform [0,1), float64\",\"edge_storage\":\"row-major, right then down\""
             << ",\"input_fnv1a64\":\"" << hex(input_hash) << "\",\"output_fnv1a64\":\"" << hex(output_hash)
             << "\",\"validated\":true,\"validation_seconds\":" << validation_seconds
             << ",\"minimum_batch_seconds\":" << min_time << ",\"warmup_seconds\":" << warmup << ",\"seconds_per_call\":[";
        for (size_t i = 0; i < seconds.size(); ++i) { if (i) *out << ','; *out << seconds[i]; }
        *out << "],\"iterations\":[";
        for (size_t i = 0; i < iterations.size(); ++i) { if (i) *out << ','; *out << iterations[i]; }
        *out << "],\"cpu_seconds_per_call\":[";
        for (size_t i = 0; i < cpu_seconds.size(); ++i) { if (i) *out << ','; *out << cpu_seconds[i]; }
        *out << "]}\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
