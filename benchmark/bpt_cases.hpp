/***************************************************************************
* Copyright ESIEE Paris (2026)                                              *
* Distributed under the terms of the CECILL-B License.                      *
* The full license is in the file LICENSE, distributed with this software.  *
****************************************************************************/
#pragma once

#include "higra/structure/undirected_graph.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <random>
#include <string>

namespace bpt_bench {
    using hg::index_t;
    using hg::num_edges;
    inline std::string env(const char *key, const std::string &fallback) {
        auto value = std::getenv(key);
        return value ? value : fallback;
    }
    struct spec {
        std::string family, weights, dtype, input;
        index_t n;
        uint64_t seed;
        std::string name() const {
            return family + "/" + weights + "/" + dtype + "/" + input +
                "/n" + std::to_string(n) + "/seed" + std::to_string(seed);
        }
    };
    inline std::vector<spec> cases() {
        const auto profile = env("BPT_PROFILE", "smoke");
        hg_assert(profile == "smoke" || profile == "campaign", "Unknown BPT_PROFILE.");
        const index_t n = std::stoll(env("BPT_SIZE", profile == "smoke" ? "64" : "65536"));
        const uint64_t seed = std::stoull(env("BPT_SEED", "1729"));
        hg_assert(n >= 8 && n <= 10000000, "BPT_SIZE must be in [8, 10000000].");
        const index_t side = std::sqrt(n), cube = std::cbrt(n);
        const index_t square = side * side, volume = cube * cube * cube;
        return {
            {"grid4", "unique", "f64", "graph", square, seed},
            {"grid8", "low8", "f32", "graph", square, seed},
            {"grid4", "equal", "i64", "graph", square, seed},
            {"grid4", "gradient", "f64", "graph", square, seed},
            {"grid3", "float", "f32", "graph", volume, seed},
            {"path", "ordered", "u64", "graph", n, seed},
            {"star", "low8", "u64", "graph", n, seed},
            {"balanced", "unique", "i64", "graph", n, seed},
            {"irregular", "float", "f64", "graph", n, seed},
            {"sparse4", "unique", "f64", "graph", n, seed},
            {"sparse8", "low8", "i64", "graph", n, seed},
            {"sparse32", "float", "f32", "graph", n, seed},
            {"geometric", "gradient", "f64", "graph", n, seed},
            {"dense", "unique", "i64", "graph", (std::min)(n, index_t(512)), seed},
            {"hierarchical", "hierarchical", "i64", "graph", n, seed},
            {"grid4", "low8", "i64", "order", square, seed},
            {"sparse8", "low8", "i64", "mst", n, seed}
        };
    }
    // Fixed generator, independent of standard-library distribution algorithms.
    struct rng {
        uint64_t state;
        uint64_t operator()() {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            return state;
        }
    };
    template<class T> struct input {
        hg::ugraph graph;
        hg::array_1d<T> weights;
        hg::array_1d<index_t> order;
    };
    template<class T> auto read_binary(const std::string &path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        hg_assert(bool(file), "Cannot read exported input.");
        const auto bytes = file.tellg();
        hg_assert(bytes >= 0 && size_t(bytes) % sizeof(T) == 0, "Bad input byte count.");
        hg::array_1d<T> data = xt::empty<T>({size_t(bytes) / sizeof(T)});
        file.seekg(0); file.read(reinterpret_cast<char *>(data.data()), bytes);
        hg_assert(bool(file), "Cannot read exported input payload.");
        return data;
    }
    template<class T> auto generate(const spec &s) {
        const auto prefix = env("BPT_LOAD_PREFIX", "");
        if (!prefix.empty()) {
            auto src = read_binary<index_t>(prefix + ".sources");
            auto dst = read_binary<index_t>(prefix + ".targets");
            auto weights = read_binary<T>(prefix + ".weights");
            auto order = read_binary<index_t>(prefix + ".order");
            hg_assert(src.size() == dst.size() && src.size() == weights.size(), "Bad input sizes.");
            hg::ugraph graph(s.n);
            for (size_t e = 0; e < src.size(); ++e) hg::add_edge(src(e), dst(e), graph);
            return input<T>{std::move(graph), std::move(weights), std::move(order)};
        }

        // Conservative preflight bound, not a promise about allocator/RSS peaks.
        const double m_bound = s.family == "dense" ? double(s.n) * (s.n - 1) / 2 :
            (s.family == "sparse32" ? 17.0 : 6.0) * s.n;
        const double budget = std::stod(env("BPT_MAX_MIB", "1024")) * 1048576;
        hg_assert(160 * m_bound + 512.0 * s.n < budget, "Case exceeds BPT_MAX_MIB preflight bound.");
        std::vector<std::pair<index_t, index_t>> edges;
        auto add = [&](index_t a, index_t b) { edges.emplace_back(a, b); };
        rng random{s.seed ? s.seed : 1};
        if (s.family == "grid4" || s.family == "grid8" || s.family == "grid3") {
            const index_t side = s.family == "grid3" ? std::llround(std::cbrt(s.n)) : std::llround(std::sqrt(s.n));
            for (index_t v = 0; v < s.n; ++v) {
                if (v % side + 1 < side) add(v, v + 1);
                if ((v / side) % side + 1 < side) add(v, v + side);
                if (s.family == "grid3" && v + side * side < s.n) add(v, v + side * side);
                if (s.family == "grid8" && v / side + 1 < side) {
                    if (v % side + 1 < side) add(v, v + side + 1);
                    if (v % side) add(v, v + side - 1);
                }
            }
        } else if (s.family == "dense") {
            for (index_t a = 0; a < s.n; ++a) for (index_t b = a + 1; b < s.n; ++b) add(a, b);
        } else {
            for (index_t v = 1; v < s.n; ++v) {
                auto parent = v - 1;
                if (s.family == "star") parent = 0;
                if (s.family == "balanced") parent = (v - 1) / 2;
                if (s.family == "irregular") parent = random() % v;
                add(parent, v);
            }
            index_t degree = s.family == "sparse4" ? 4 : s.family == "sparse8" ? 8 :
                s.family == "sparse32" ? 32 : s.family == "geometric" ? 8 : 0;
            while (degree && edges.size() < size_t(s.n * degree / 2)) {
                const index_t a = random() % s.n;
                const index_t b = s.family == "geometric" ? (a + 1 + random() % 16) % s.n : random() % s.n;
                if (a != b) add(a, b);
            }
        }
        if (s.weights != "ordered" && s.weights != "hierarchical") {
            for (size_t i = edges.size(); i > 1; --i) std::swap(edges[i - 1], edges[random() % i]);
        }
        hg::ugraph graph(s.n);
        hg::array_1d<T> weights = xt::empty<T>({edges.size()});
        std::vector<index_t> ranks(edges.size());
        std::iota(ranks.begin(), ranks.end(), 0);
        for (size_t i = ranks.size(); i > 1; --i) std::swap(ranks[i - 1], ranks[random() % i]);
        for (size_t i = 0; i < edges.size(); ++i) {
            const auto a = edges[i].first, b = edges[i].second;
            hg::add_edge(a, b, graph);
            double w = ranks[i];
            if (s.weights == "low8") w = random() % 8;
            if (s.weights == "equal") w = 1;
            if (s.weights == "float") w = double(random() >> 11) / 9007199254740992.0;
            if (s.weights == "gradient") w = std::abs(std::sin(a * .03) - std::sin(b * .03));
            if (s.weights == "ordered") w = i;
            if (s.weights == "hierarchical") {
                index_t v = i + 1, level = 0;
                while (!(v & 1)) { ++level; v >>= 1; }
                w = level;
            }
            weights(i) = static_cast<T>(w);
            // Preserve integer precision above the exact double range.
            if (s.dtype == "u64") weights(i) += T(uint64_t(1) << 63);
            if (s.dtype == "i64") weights(i) += T(int64_t(1) << 54);
        }
        if (s.input == "mst") {
            auto reference = hg::bpt_canonical(graph, weights);
            hg::ugraph mst(s.n);
            hg::array_1d<T> mw = xt::empty<T>({size_t(s.n - 1)});
            auto src = hg::sources(graph), dst = hg::targets(graph);
            for (index_t i = 0; i < s.n - 1; ++i) {
                auto edge = reference.mst_edge_map(i);
                hg::add_edge(src(edge), dst(edge), mst); mw(i) = weights(edge);
            }
            graph = std::move(mst); weights = std::move(mw);
        }
        hg::array_1d<index_t> order = hg::stable_arg_sort(weights);
        if (s.input == "order") {
            // Authoritative supplied order, deliberately independent of weights.
            for (size_t i = order.size(); i > 1; --i) std::swap(order(i - 1), order(random() % i));
        }
        return input<T>{std::move(graph), std::move(weights), std::move(order)};
    }
}
