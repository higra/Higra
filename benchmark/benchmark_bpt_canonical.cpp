/***************************************************************************
* Copyright ESIEE Paris (2026)                                              *
* Distributed under the terms of the CECILL-B License.                      *
* The full license is in the file LICENSE, distributed with this software.  *
****************************************************************************/

#include "bpt_cases.hpp"
#include <benchmark/benchmark.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#ifdef __unix__
#include <sys/resource.h>
#endif

namespace bpt_bench {
    namespace core = hg::hierarchy_core_internal;
    const std::vector<std::string> methods{"kruskal", "kruskal_comparator", "exact", "basin", "reverse"};

    template<bool kruskal = false, class T, class result_t>
    auto finalize(result_t result, const input<T> &data) {
#ifdef HG_BPT_BENCH_DIAGNOSTICS
        core::bpt_bench_phase finalization_timer("frontend_finalization");
#endif
        const auto n = hg::num_vertices(data.graph);
        hg::array_1d<T> levels = xt::zeros<T>({result.first.size()});
        if constexpr (kruskal) {
            xt::noalias(xt::view(levels, xt::range(n, levels.size()))) = xt::index_view(data.weights, result.second);
        } else {
            hg::parfor(0, n - 1, [&](index_t i) { levels(n + i) = data.weights(result.second(i)); });
        }
        return hg::make_node_weighted_tree_and_mst(hg::tree(std::move(result.first)),
            std::move(levels), std::move(result.second));
    }

    // Match the scalar public frontend for every policy. Compile-time policies
    // keep method-name and rank/scalar dispatch out of all edge comparisons.
    template<core::bpt_boruvka_numbering numbering, core::bpt_boruvka_tie_order ties, class T>
    auto scalar_boruvka(const input<T> &data) {
        const auto &weights = data.weights;
        hg_assert_edge_weights(data.graph, weights);
        hg_assert_1d_array(weights);
        const auto vertex_count = hg::num_vertices(data.graph);
        hg_assert(vertex_count > 0 &&
                  static_cast<uintmax_t>(vertex_count) <= static_cast<uintmax_t>((std::numeric_limits<index_t>::max)()),
                  "Vertex count exceeds index capacity or is empty.");
        const auto n = static_cast<index_t>(vertex_count);
        core::bpt_boruvka_check_sizes(n, weights.size());
        for (const auto w: weights) hg_assert(w == w, "NaN edge weights are not supported.");
        const auto &src = hg::sources(data.graph), &dst = hg::targets(data.graph);
        hg_assert(src.size() == weights.size(), "Endpoint and weight counts must match.");
        auto less = [&](index_t a, index_t b) {
            return core::bpt_boruvka_scalar_less(weights, a, b, ties);
        };
        return finalize(core::bpt_boruvka_from_edge_order(src, dst, n, less, numbering), data);
    }

    template<class T> auto call(const input<T> &data, const spec &s, const std::string &method) {
        const auto n = hg::num_vertices(data.graph);
        if (s.input != "order") {
            if (method == "kruskal") return hg::bpt_canonical(data.graph, data.weights);
            if (method == "exact") return scalar_boruvka<core::bpt_boruvka_numbering::exact,
                core::bpt_boruvka_tie_order::edge_id>(data);
            if (method == "basin") return scalar_boruvka<core::bpt_boruvka_numbering::basin,
                core::bpt_boruvka_tie_order::edge_id>(data);
            if (method == "reverse") return scalar_boruvka<core::bpt_boruvka_numbering::basin,
                core::bpt_boruvka_tie_order::reverse_edge_id>(data);
            hg_assert(method == "kruskal_comparator", "Unknown method.");
            // Match the current baseline's validation and finalization. Only
            // its stable index sort is replaced with a unique-key comparator sort.
            hg_assert_edge_weights(data.graph, data.weights);
            hg_assert_1d_array(data.weights);
            hg::array_1d<index_t> order = xt::arange<index_t>(data.weights.size());
            hg::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
                return core::bpt_boruvka_scalar_less(data.weights, a, b);
            });
            return finalize<true>(core::bpt_canonical_from_sorted_edges(
                hg::sources(data.graph), hg::targets(data.graph), order, n), data);
        }
        // Supplied-order Kruskal consumes the permutation directly. Both new
        // policies pay identical validation/inverse-rank costs inside the call.
        if (method == "kruskal") return finalize<true>(core::bpt_canonical_from_sorted_edges(
            hg::sources(data.graph), hg::targets(data.graph), data.order, n), data);
        hg_assert(method == "exact" || method == "basin", "Unknown order method.");
        hg_assert_edge_weights(data.graph, data.weights);
        hg_assert_1d_array(data.weights);
        for (auto w: data.weights) hg_assert(w == w, "NaN weight.");
        hg::array_1d<index_t> rank = xt::empty<index_t>({data.weights.size()});
        rank.fill(hg::invalid_index);
        hg_assert(data.order.size() == rank.size(), "Invalid permutation length.");
        for (size_t i = 0; i < data.order.size(); ++i) {
            auto e = data.order(i);
            hg_assert(e >= 0 && size_t(e) < rank.size() && rank(e) == hg::invalid_index, "Invalid permutation.");
            rank(e) = i;
        }
        auto less = [&](index_t a, index_t b) { return rank(a) < rank(b); };
        return finalize(core::bpt_boruvka_from_edge_order(hg::sources(data.graph), hg::targets(data.graph), n, less,
            method == "exact" ? core::bpt_boruvka_numbering::exact : core::bpt_boruvka_numbering::basin), data);
    }

    template<class A, class B> void validate(const A &got, const B &ref, index_t n, bool exact) {
        const auto &p = got.tree.parents(), &q = ref.tree.parents();
        hg_assert(p.size() == size_t(2 * n - 1) && p(p.size() - 1) == index_t(p.size() - 1), "Bad root/size.");
        std::vector<index_t> degree(p.size(), 0), remap(p.size()), inverse(n - 1);
        for (index_t i = 0; i < n - 1; ++i) inverse[i] = ref.mst_edge_map(i);
        std::map<index_t, index_t> edge_to_node;
        for (index_t i = 0; i < n - 1; ++i) edge_to_node[inverse[i]] = n + i;
        for (index_t v = 0; v < n; ++v) remap[v] = v;
        for (index_t i = 0; i < n - 1; ++i) {
            hg_assert(edge_to_node.count(got.mst_edge_map(i)), "Different MST set.");
            remap[n + i] = edge_to_node.at(got.mst_edge_map(i));
        }
        for (size_t i = 0; i < p.size(); ++i) {
            if (i + 1 < p.size()) { hg_assert(p(i) > index_t(i), "Backward parent."); ++degree.at(p(i)); }
            hg_assert(remap[p(i)] == q(remap[i]), "Edge-labeled tree mismatch.");
            hg_assert(got.altitudes(i) == ref.altitudes(remap[i]), "Altitude mismatch.");
            if (exact) hg_assert(p(i) == q(i) && remap[i] == index_t(i), "Exact mismatch.");
        }
        for (index_t i = n; i < index_t(p.size()); ++i) hg_assert(degree[i] == 2, "Not binary.");
    }
    template<class T> void check(const input<T> &data, const spec &s, const std::string &method) {
        auto got = call(data, s, method);
        if (method == "exact" && s.input != "order") {
            auto api = hg::bpt_canonical_boruvka(data.graph, data.weights);
            validate(got, api, s.n, true);
        }
        if (method != "reverse") {
            auto ref = call(data, s, "kruskal");
            validate(got, ref, s.n, method != "basin");
        } else {
            auto order = data.order;
            hg::sort(order.begin(), order.end(), [&](index_t a, index_t b) {
                return core::bpt_boruvka_scalar_less(data.weights, a, b, core::bpt_boruvka_tie_order::reverse_edge_id);
            });
            auto sorted = core::bpt_canonical_from_sorted_edges(hg::sources(data.graph), hg::targets(data.graph), order, s.n);
            hg::array_1d<T> levels = xt::zeros<T>({sorted.first.size()});
            for (index_t i = 0; i < s.n - 1; ++i) levels(s.n + i) = data.weights(sorted.second(i));
            auto ref = hg::make_node_weighted_tree_and_mst(hg::tree(std::move(sorted.first)), std::move(levels), std::move(sorted.second));
            validate(got, ref, s.n, false);
        }
    }

    template<class T> void run(benchmark::State &state, spec s, std::string method) {
        try {
            const auto data = generate<T>(s);
            check(data, s, method); // Before State's timed loop starts.
            for (auto _: state) {
                auto result = call(data, s, method);
                benchmark::DoNotOptimize(result.tree.parents().data());
                benchmark::DoNotOptimize(result.altitudes.data());
                benchmark::DoNotOptimize(result.mst_edge_map.data());
                benchmark::ClobberMemory();
            } // Result destruction is timed for every competitor.
            state.counters["vertices"] = s.n;
            state.counters["edges"] = hg::num_edges(data.graph);
            state.SetItemsProcessed(state.iterations() * hg::num_edges(data.graph));
            state.SetLabel("validated;fresh_allocations;destruction_included");
        } catch (const std::exception &e) { state.SkipWithError(e.what()); }
    }
    template<class F> auto dispatch(const spec &s, F &&f) {
        if (s.dtype == "f32") return f(float{});
        if (s.dtype == "i64") return f(int64_t{});
        if (s.dtype == "u64") return f(uint64_t{});
        return f(double{});
    }
    const bool registered = [] {
        for (const auto &s: cases()) for (const auto &method: methods) {
            if (s.input == "order" && (method == "reverse" || method == "kruskal_comparator")) continue;
            const auto name = "BM_bpt/" + s.name() + "/" + method;
            dispatch(s, [&](auto value) {
                benchmark::RegisterBenchmark(name.c_str(), run<decltype(value)>, s, method)->UseRealTime();
            });
        }
        return true;
    }();
    template<class E> void write_binary(const std::string &path, const E &values) {
        std::ofstream file(path, std::ios::binary);
        for (const auto &v: values) file.write(reinterpret_cast<const char *>(&v), sizeof(v));
        hg_assert(bool(file), "Cannot write input export.");
    }
    template<class T> void single(const spec &s, const std::string &mode, const std::string &method) {
        const auto data = generate<T>(s);
        if (mode == "inputs") {
            const auto prefix = env("BPT_INPUT_PREFIX", "/tmp/bpt-input");
            write_binary(prefix + ".sources", hg::sources(data.graph));
            write_binary(prefix + ".targets", hg::targets(data.graph));
            write_binary(prefix + ".weights", data.weights);
            write_binary(prefix + ".order", data.order);
            std::cout << "{\"n\":" << s.n << ",\"m\":" << data.weights.size() << ",\"dtype\":\"" << s.dtype << "\"}\n";
            return;
        }
        if (mode == "validate") {
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            core::bpt_boruvka_diagnostics validation_diagnostics;
            core::bpt_bench_diagnostics = &validation_diagnostics;
#endif
            check(data, s, method);
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            core::bpt_bench_diagnostics = nullptr;
#endif
            std::cout << "{\"validated\":true}\n"; return;
        }
#ifdef HG_BPT_BENCH_DIAGNOSTICS
        core::bpt_boruvka_diagnostics diagnostics;
        core::bpt_bench_diagnostics = &diagnostics;
#endif
        const auto start = std::chrono::steady_clock::now();
        { auto result = call(data, s, method); benchmark::DoNotOptimize(result.tree.parents().data()); }
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
#ifdef HG_BPT_BENCH_DIAGNOSTICS
        core::bpt_bench_diagnostics = nullptr;
#endif
        std::cout << "{\"schema\":1,\"seconds_including_destruction\":" << elapsed;
#ifdef __unix__
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
        std::cout << ",\"peak_rss_bytes\":" << usage.ru_maxrss;
#else
        std::cout << ",\"peak_rss_bytes\":" << usage.ru_maxrss * 1024;
#endif
#endif
#ifdef HG_BPT_BENCH_DIAGNOSTICS
        std::cout << ",\"tracked_scratch_high_water_bytes\":" << diagnostics.tracked_scratch_high_water_bytes;
        std::cout << ",\"rounds\":[";
        for (size_t i = 0; i < diagnostics.rounds.size(); ++i) {
            auto r = diagnostics.rounds[i];
            if (i) std::cout << ',';
            std::cout << "{\"components\":" << r.components << ",\"active\":" << r.active << ",\"accepted\":" << r.accepted
                << ",\"remaining\":" << r.remaining << ",\"jump_passes\":" << r.jump_passes << '}';
        }
        std::cout << "],\"phases\":[";
        double phase_sum = 0;
        for (size_t i = 0; i < diagnostics.phases.size(); ++i) {
            const auto &p = diagnostics.phases[i];
            if (i) std::cout << ',';
            std::cout << "{\"name\":\"" << p.first << "\",\"seconds\":" << p.second << '}';
            if (p.first == "waterfall_total" || p.first == "expansion" || p.first == "basin_sort" ||
                p.first == "rewiring" || p.first == "exact_numbering_sort" || p.first == "export_mapping" ||
                p.first == "frontend_finalization") phase_sum += p.second;
        }
        std::cout << "],\"frontend_and_unattributed_seconds\":" << elapsed - phase_sum << ",\"expansion_climbs\":[";
        for (size_t i = 0; i < diagnostics.climbs.size(); ++i) { if (i) std::cout << ','; std::cout << diagnostics.climbs[i]; }
        std::cout << ']';
#ifdef HG_BPT_BENCH_CAS
        std::cout << ",\"cas_attempts\":" << diagnostics.cas_attempts.load() << ",\"cas_failures\":" << diagnostics.cas_failures.load();
#endif
#endif
        std::cout << "}\n";
    }
}

bool bpt_special_run() {
    using namespace bpt_bench;
    std::cout << std::setprecision(17);
    const auto mode = env("BPT_RUN_MODE", "");
    if (mode.empty()) return false;
    if (mode == "self-test") {
        size_t checked = 0;
        for (const auto &s: cases()) {
            dispatch(s, [&](auto value) {
                const auto data = generate<decltype(value)>(s);
                for (const auto &method: methods) {
                    if (s.input == "order" && (method == "reverse" || method == "kruskal_comparator")) continue;
                    check(data, s, method);
                    ++checked;
                }
            });
        }
        std::cout << "{\"self_test\":\"passed\",\"method_cases\":" << checked << "}\n";
        return true;
    }
    hg_assert(mode == "inputs" || mode == "validate" || mode == "memory" || mode == "diagnostics", "Unknown BPT_RUN_MODE.");
    const auto selected = env("BPT_CASE", "");
    for (const auto &s: cases()) if (s.name() == selected) {
        const auto method = env("BPT_METHOD", "exact");
        hg_assert(std::find(methods.begin(), methods.end(), method) != methods.end(), "Unknown method.");
        dispatch(s, [&](auto value) { single<decltype(value)>(s, mode, method); });
        return true;
    }
    throw std::runtime_error("BPT_CASE must select a full case name.");
}
