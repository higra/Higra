/***************************************************************************
* Copyright ESIEE Paris (2026)                                              *
* Distributed under the terms of the CECILL-B License.                      *
* The full license is in the file LICENSE, distributed with this software.  *
****************************************************************************/
#include "higra/hierarchy/hierarchy_core.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace bpt_anchor {
using hg::index_t;
namespace core = hg::hierarchy_core_internal;
using result_t = hg::node_weighted_tree_and_mst<hg::tree, hg::array_1d<float>>;
const std::vector<std::string> cases = {
    "kruskal", "boruvka", "boruvka_compact", "boruvka_basin", "boruvka_basin_compact"};

struct input {
    hg::ugraph graph;
    hg::array_1d<float> weights;
    uint64_t fingerprint = 14695981039346656037ULL;

    explicit input(index_t side = 1024)
        : graph(side * side, 2 * side * (side - 1) + 2 * (side - 1) * (side - 1), 8),
          weights(xt::empty<float>({2 * side * (side - 1) + 2 * (side - 1) * (side - 1)})) {
        uint64_t state = 1729;
        index_t e = 0;
        auto mix = [&](uint64_t value) { fingerprint = (fingerprint ^ value) * 1099511628211ULL; };
        auto add = [&](index_t a, index_t b) {
            hg::add_edge(a, b, graph);
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            weights(e) = float(state >> 40) * 0x1p-24f;
            uint32_t bits;
            std::memcpy(&bits, &weights(e), sizeof(bits));
            mix(a); mix(b); mix(bits);
            ++e;
        };
        for (index_t v = 0; v < side * side; ++v) {
            if (v % side + 1 < side) add(v, v + 1);
            if (v / side + 1 < side) {
                add(v, v + side);
                if (v % side + 1 < side) add(v, v + side + 1);
                if (v % side) add(v, v + side - 1);
            }
        }
        hg_assert(e == hg::num_edges(graph), "Bad anchor edge count.");
        if (side == 1024) hg_assert(fingerprint == 7594927944782184811ULL, "Bad anchor fingerprint.");
    }
};

result_t call(const input &in, const std::string &method) {
    hg_assert(std::find(cases.begin(), cases.end(), method) != cases.end(), "Unknown BPT case.");
    if (method == "kruskal") return hg::bpt_canonical(in.graph, in.weights);
    const auto numbering = method.find("basin") == std::string::npos ?
        core::bpt_boruvka_numbering::exact : core::bpt_boruvka_numbering::basin;
    return hg::bpt_canonical_boruvka(in.graph, in.weights, numbering,
                                   method.find("compact") != std::string::npos);
}

void validate(const input &in, const result_t &got, const result_t &ref, bool exact) {
    const auto n = static_cast<index_t>(hg::num_vertices(in.graph));
    const auto m = static_cast<index_t>(hg::num_edges(in.graph));
    const auto &p = got.tree.parents(), &q = ref.tree.parents();
    hg_assert(p.size() == size_t(2 * n - 1) && got.altitudes.size() == p.size() &&
              got.mst_edge_map.size() == size_t(n - 1), "Bad output sizes.");
    const auto root = index_t(p.size()) - 1;
    hg_assert(p(root) == root && got.tree.category() == hg::tree_category::partition_tree, "Bad root/category.");
    std::vector<index_t> inverse(m, hg::invalid_index), remap(p.size()), degree(p.size(), 0);
    for (index_t i = 0; i < n - 1; ++i) inverse.at(ref.mst_edge_map(i)) = n + i;
    for (index_t i = 0; i < n; ++i) remap[i] = i;
    for (index_t i = 0; i < n - 1; ++i) {
        const auto e = got.mst_edge_map(i);
        hg_assert(e >= 0 && e < m && inverse.at(e) >= n, "Different or repeated building edge.");
        remap[n + i] = inverse[e];
        inverse[e] = hg::invalid_index;
        hg_assert(got.altitudes(n + i) == in.weights(e), "Bad building altitude.");
    }
    for (index_t i = 0; i <= root; ++i) {
        hg_assert(p(i) >= 0 && p(i) <= root && (i == root || p(i) > i), "Bad parent order.");
        if (i != root) {
            ++degree[p(i)];
            hg_assert(got.altitudes(i) <= got.altitudes(p(i)), "Nonmonotone altitude.");
        }
        hg_assert(remap[p(i)] == q(remap[i]) && got.altitudes(i) == ref.altitudes(remap[i]), "Tree mismatch.");
        if (exact) hg_assert(remap[i] == i && p(i) == q(i), "Exact numbering mismatch.");
    }
    for (index_t i = 0; i < n; ++i) hg_assert(degree[i] == 0 && got.altitudes(i) == 0, "Bad leaf.");
    for (index_t i = n; i <= root; ++i) hg_assert(degree[i] == 2, "Not binary.");
}

void consume(const result_t &r) {
    benchmark::DoNotOptimize(r.tree.parents().data());
    benchmark::DoNotOptimize(r.altitudes.data());
    benchmark::DoNotOptimize(r.mst_edge_map.data());
    benchmark::ClobberMemory();
}

void prepare(const input &in, const std::string &method) {
    { auto ref = hg::bpt_canonical(in.graph, in.weights);
      auto got = call(in, method);
      validate(in, got, ref, method.find("basin") == std::string::npos); }
    { auto warmup = call(in, method); consume(warmup); }
}

void run(benchmark::State &state, const std::string &method) {
    input in;
    prepare(in, method);
    for (auto _ : state) {
        auto result = call(in, method);
        consume(result);
    }
    state.SetItemsProcessed(state.iterations());
}

// A tiny untimed fixture checks the runner and all five adapters, never the large anchor.
void self_test() {
    input in(4);
    for (const auto &method : cases) prepare(in, method);
    bool rejected = false;
    try { call(in, "unknown"); } catch (const std::runtime_error &) { rejected = true; }
    hg_assert(rejected, "Unknown case accepted.");
    std::cout << "BPT runner: five adapters and invalid selector passed.\n";
}
} // namespace bpt_anchor

// Returns before Google Benchmark initialization only for explicit single-call/fixture commands.
bool bpt_single_run(int argc, char **argv, size_t thread_limit) {
    using namespace bpt_anchor;
    if (argc == 2 && std::string(argv[1]) == "--bpt-self-test") { self_test(); return true; }
    if (argc < 2 || std::string(argv[1]) != "--bpt-case") return false;
    hg_assert(argc == 3, "Usage: --bpt-case CASE");
    const std::string method = argv[2];
    hg_assert(std::find(cases.begin(), cases.end(), method) != cases.end(), "Unknown BPT case.");
    input in;
    prepare(in, method);
    const auto start = std::chrono::steady_clock::now();
    { auto result = call(in, method); consume(result); }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::setprecision(17) << "{\"case\":\"" << method << "\",\"seconds\":" << seconds
              << ",\"n\":1048576,\"m\":4188162,\"fingerprint\":\"" << in.fingerprint
              << "\",\"validated\":true,\"warmups\":1,\"timed_calls\":1,\"thread_limit\":" << thread_limit;
#ifdef HG_USE_TBB
    std::cout << ",\"tbb\":true";
#else
    std::cout << ",\"tbb\":false";
#endif
    std::cout << "}\n";
    return true;
}

namespace {
const bool registered = [] {
    for (const auto &method : bpt_anchor::cases)
        benchmark::RegisterBenchmark(("BPT/" + method).c_str(),
            [method](benchmark::State &state) { bpt_anchor::run(state, method); })->UseRealTime();
    return true;
}();
}
