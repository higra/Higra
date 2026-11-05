/***************************************************************************
* Copyright ESIEE Paris (2018)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once


#include "../graph.hpp"
#include "../structure/array.hpp"
#include "higra/structure/unionfind.hpp"
#include "higra/hierarchy/hierarchy_core.hpp"
#include "higra/algo/graph_core.hpp"
#include "higra/sorting.hpp"
#include <vector>
#include <algorithm>
#include <stack>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#ifdef HG_USE_TBB
#include <oneapi/tbb/task_arena.h>
#endif

namespace hg {

    /**
     * Linear time watershed cut algorithm.
     *
     * Jean Cousty, Gilles Bertrand, Laurent Najman, Michel Couprie. Watershed Cuts: Minimum Spanning
     * Forests and the Drop of Water Principle. IEEE Transactions on Pattern Analysis and Machine
     * Intelligence, Institute of Electrical and Electronics Engineers, 2009, 31 (8), pp.1362-1374.
     * @tparam graph_t
     * @tparam T
     * @param graph
     * @param xedge_weights
     * @return array of labels on graph vertices, numbered from 1 to n with n the number of minima
     */
    template<typename graph_t, typename T>
    auto
    labelisation_watershed(const graph_t &graph, const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        using value_type = typename T::value_type;
        using vertex_t = typename graph_traits<graph_t>::vertex_descriptor;

        auto fminus = array_1d<value_type>::from_shape({graph.num_vertices()});

        for (auto v: vertex_iterator(graph)) {
            auto minValue = (std::numeric_limits<value_type>::max)();
            for (auto e: out_edge_iterator(v, graph)) {
                minValue = (std::min)(minValue, edge_weights(e));
            }
            fminus[v] = minValue;
        }


        auto no_label = (std::numeric_limits<index_t>::max)();
        auto labels = array_1d<index_t>::from_shape({graph.num_vertices()});
        std::fill(labels.begin(), labels.end(), no_label);

        auto notInL = array_1d<bool>::from_shape({graph.num_vertices()});
        std::fill(notInL.begin(), notInL.end(), true);

        std::vector<vertex_t> L;
        std::vector<vertex_t> LL;

        auto stream = [&L, &LL, &graph, &edge_weights, &fminus, &notInL, &labels, no_label](vertex_t x) {
            L.clear();
            LL.clear();
            L.push_back(x);
            LL.push_back(x);
            notInL[x] = false;

            while (!LL.empty()) {
                auto y = LL[LL.size() - 1];
                LL.pop_back();

                for (auto e: out_edge_iterator(y, graph)) {
                    auto adjacent_vertex = target(e, graph);
                    if (notInL[adjacent_vertex] && edge_weights(e) == fminus[y]) {
                        if (labels[adjacent_vertex] != no_label) {
                            return labels[adjacent_vertex];
                        } else if (fminus[adjacent_vertex] < fminus[y]) {
                            L.push_back(adjacent_vertex);
                            notInL[adjacent_vertex] = false;
                            LL.clear();
                            LL.push_back(adjacent_vertex);
                            break; // stop breadth_first
                        } else {
                            L.push_back(adjacent_vertex);
                            notInL[adjacent_vertex] = false;
                            LL.push_back(adjacent_vertex);
                        }
                    }
                }
            }
            return no_label;
        };

        index_t num_labs = 0;

        for (auto v: vertex_iterator(graph)) {
            if (labels[v] == no_label) {
                auto res = stream(v);
                if (res == no_label) {
                    num_labs++;
                    for (auto x: L) {
                        labels[x] = num_labs;
                        notInL[x] = true;
                    }
                } else {
                    for (auto x: L) {
                        labels[x] = res;
                        notInL[x] = true;
                    }
                }
            }
        }
        return labels;
    };


    template<typename graph_t, typename T1, typename T2>
    auto labelisation_seeded_watershed(
            const graph_t &graph,
            const xt::xexpression<T1> &xedge_weights,
            const xt::xexpression<T2> &xvertex_seeds,
            const typename T2::value_type background_label = 0) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        auto &vertex_seeds = xvertex_seeds.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_node_weights(graph, vertex_seeds);
        hg_assert_1d_array(edge_weights);
        hg_assert_1d_array(vertex_seeds);

        using label_type = typename T2::value_type;

        array_1d<index_t> sorted_edges_indices = stable_arg_sort(edge_weights);

        index_t num_nodes = num_vertices(graph);
        index_t num_edges = sorted_edges_indices.size();

        union_find uf(num_nodes);

        array_1d<label_type> labels = vertex_seeds;

        for (index_t i = 0; i < num_edges; i++) {
            auto ei = sorted_edges_indices[i];
            auto e = edge_from_index(ei, graph);
            auto c1 = uf.find(source(e, graph));
            auto c2 = uf.find(target(e, graph));

            if (c1 != c2 && (labels(c1) == background_label || labels(c2) == background_label)) {
                if (labels(c1) == background_label) {
                    labels(c1) = labels(c2);
                } else {
                    labels(c2) = labels(c1);
                }
                uf.link(c1, c2);
            }

        }

        for (index_t i = 0; i < num_nodes; i++) {
            if (labels(i) == background_label) {
                labels(i) = labels(uf.find(i));
            }
        }

        return labels;
    };


    /**
     * Incremental seeded watershed cut based on the binary partition tree.
     *
     * This class provides an efficient way to compute seeded watershed cuts
     * in an interactive segmentation setting, where seeds are added and removed
     * incrementally.
     *
     * The algorithm maintains a canonical BPT and a visitCount array to identify
     * watershed edges without recomputing from scratch at each interaction.
     * The labeling is cached and updated locally when seeds change.
     *
     * Complexity:
     *   Construction:    O(n) with n the number of elements in the BPT
     *   add_seeds(K):    O(K * d + S) where d is the height of the BPT
     *                    (O(log N) for balanced trees, O(N) worst case, with
     *                    N the number of vertices) and S is the total size of
     *                    the regions relabeled.
     *   remove_seeds(K): O(K * d + S) same as add_seeds.
     *   get_labeling:    O(1) (the labeling is maintained incrementally).
     *
     * Reference:
     * Quentin Lebon, Josselin Lefevre, Jean Cousty, Benjamin Perret.
     * Interactive Segmentation With Incremental Watershed Cuts.
     * CIARP 2023.
     *
     * Example:
     *   auto iws = hg::make_incremental_watershed_cut(graph, edge_weights);
     *   hg::array_1d<hg::index_t> sv{0, 5};
     *   hg::array_1d<hg::index_t> sl{1, 2};
     *   iws.add_seeds(sv, sl);
     *   auto labels = iws.get_labeling();
     *   iws.remove_seeds(sv);
     */
    class incremental_watershed_cut {

    public:

        /**
         * Create an incremental watershed cut object from a BPT and its
         * associated MST.
         *
         * The MST edges must be ordered consistently with the BPT internal
         * nodes: MST edge i corresponds to BPT internal node (num_leaves + i).
         * This ordering is guaranteed when the MST is built from bpt_canonical.
         *
         * @param bpt canonical binary partition tree
         * @param mst minimum spanning tree of the original graph (as ugraph)
         */
        incremental_watershed_cut(
                                tree bpt, 
                                ugraph mst
                            ) :
            m_bpt(std::move(bpt)),
            m_mst(std::move(mst)),
            m_num_leaves((index_t)num_leaves(m_bpt)),
            m_root((index_t)hg::num_vertices(m_bpt) - 1),
            m_seed_count(hg::num_vertices(m_bpt), 0),
            m_is_cut(m_num_leaves - 1, false),
            #ifdef HG_USE_TBB
            m_minimum_parallel_front_size(165),
            m_num_parrallel_hops(12),
            m_maximum_concurrency(oneapi::tbb::info::default_concurrency()),
            #endif
            m_visited(m_num_leaves, 0) {
                HG_TRACE();
                hg_assert((index_t)num_vertices(m_mst) == m_num_leaves,
                        "MST must have the same number of vertices as leaves in the BPT.");
                hg_assert((index_t)num_edges(m_mst) == m_num_leaves - 1,
                        "MST must have exactly num_leaves - 1 edges.");
                m_labels = xt::zeros<index_t>({(size_t)m_num_leaves});
                m_seed_map = array_1d<index_t>::from_shape({(size_t)m_num_leaves});
                m_seed_map.fill(-1);
            }

        /**
         * Set the value of m_minimum_parallel_front_size
         * 
         * @param minimum_parallel_front_size minimum breadth for the parallel algorithm (index_t)
         */
        void set_minimum_parallel_front_size(index_t minimum_parallel_front_size) {
            #ifndef HG_USE_TBB
                HG_LOG_WARNING("set_minimum_parallel_front_size has no effect because TBB is not enabled.");
            #else
                m_minimum_parallel_front_size = std::clamp(minimum_parallel_front_size, index_t(1), std::numeric_limits<index_t>::infinity());
            #endif
        }

        /**
         * Set the value of m_num_parrallel_hops
         * 
         * @param num_parrallel_hops parallel depth for the parallel algorithm (index_t)
         */
        void set_num_parrallel_hops(index_t num_parrallel_hops) {
            #ifndef HG_USE_TBB
                HG_LOG_WARNING("set_num_parrallel_hops has no effect because TBB is not enabled.");
            #else
                m_num_parrallel_hops = std::clamp(num_parrallel_hops, index_t(1), std::numeric_limits<index_t>::infinity());
            #endif
        }

        /**
         * Set the value of m_maximum_concurrency
         * 
         * @param maximum_concurrency maximum number of processors to use (index_t)
         */
        void set_max_concurrency(index_t maximum_concurrency) {
            #ifndef HG_USE_TBB
                HG_LOG_WARNING("set_max_concurrency has no effect because TBB is not enabled.");
            #else
                m_maximum_concurrency = std::clamp(maximum_concurrency, index_t(1), static_cast<index_t>(oneapi::tbb::info::default_concurrency()));
            #endif
        }

        /**
         * Add seeds to the current watershed cut.
         *
         * Each seed is defined by a vertex and a label. Two seeds cannot share
         * the same vertex but can share the same label (resulting in merged regions
         * in the output labeling). Labels must be non-zero (0 is reserved for
         * unlabeled/background vertices).
         *
         * @tparam T1 integral type for seed vertices
         * @tparam T2 integral type for seed labels
         * @param xseed_vertices 1d array of seed vertex indices
         * @param xseed_labels 1d array of seed labels (same size as seed_vertices)
         */
        template<typename T1, typename T2>
        void add_seeds(const xt::xexpression<T1> &xseed_vertices,
                       const xt::xexpression<T2> &xseed_labels) {
            HG_TRACE();
            auto &seed_vertices = xseed_vertices.derived_cast();
            auto &seed_labels = xseed_labels.derived_cast();
            hg_assert_1d_array(seed_vertices);
            hg_assert_1d_array(seed_labels);
            hg_assert(seed_vertices.size() == seed_labels.size(),
                      "seed_vertices and seed_labels must have the same size.");

            m_visited_generation++;

            for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                auto v = (index_t)seed_vertices(i);

                if(m_seed_map[v] == v) // v is already a seed
                    continue;

                index_t n = v;
                while (n != m_root && m_seed_count[n] != 2) {
                    n = m_bpt.parent(n);
                    m_seed_count[n] += 1;
                    if (m_seed_count[n] == 2) {
                        m_is_cut[n - m_num_leaves] = true;
                    }
                }
            }

            #ifdef HG_USE_TBB
            if (m_maximum_concurrency > 1) {
                std::vector<index_t> roots;
                std::vector<index_t> root_labels;
                std::vector<index_t> root_seeds;

                for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                    auto v = (index_t)seed_vertices(i);
                    auto l = (index_t)seed_labels(i);

                    if (m_seed_map[v] == v && m_labels[v] == l)
                        continue;

                    roots.push_back(v);
                    root_labels.push_back(l);
                    root_seeds.push_back(v);
                }

                relabel_component_from_seeds(roots, root_labels, root_seeds);
            } else
            #endif
            {
                for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                    auto v = (index_t)seed_vertices(i);
                    auto l = (index_t)seed_labels(i);

                    if (m_seed_map[v] == v && m_labels[v] == l)
                        continue;

                    relabel_component_from_seed(v, l, v);
                }
            }
        }

        /**
         * Remove seeds from the current watershed cut.
         *
         * @tparam T1 integral type for seed vertices
         * @param xseed_vertices 1d array of seed vertex indices to remove
         */
        template<typename T1>
        void remove_seeds(const xt::xexpression<T1> &xseed_vertices) {
            m_visited_generation++;
            HG_TRACE();
            auto &seed_vertices = xseed_vertices.derived_cast();
            hg_assert_1d_array(seed_vertices);

            std::vector<index_t> decut_edges;   // MST edge indices that just got un-cut
            std::vector<index_t> start_vertices; // vertices from which to start BFS
            std::vector<index_t> labels_to_propagate; // labels to propagate
            std::vector<index_t> reference_seeds; // seeds that define the labels to propagate
            std::vector<index_t> removed_ws_cut_edges; // MST edge indices of the watershed edges to remove

            for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                auto v = (index_t)seed_vertices(i);

                if(m_seed_map[v] != v) // this vertex is not a seed...
                    continue;

                m_seed_map[v] = -1; // mark seed as undefined for the moment

                index_t n = v;
                while (n != m_root) {
                    n = m_bpt.parent(n);
                    m_seed_count[n] -= 1;
                    if (m_seed_count[n] == 1) {
                        decut_edges.push_back(n - m_num_leaves);
                        break;
                    }
                }

            }

            for (auto k : decut_edges) {
                const auto &e = edge_from_index(k, m_mst);
                auto u = source(e, m_mst);
                auto w = target(e, m_mst);

                auto u_seed = m_seed_map[u];
                if (u_seed !=-1 ) u_seed = m_seed_map[u_seed]; // if u was not a seed itself, follow the seed map to find the seed of its component
                auto w_seed = m_seed_map[w];
                if (w_seed != -1) w_seed = m_seed_map[w_seed]; // if w was not a seed itself, follow the seed map to find the seed of its component

                bool u_removed = (u_seed == -1);
                bool w_removed = (w_seed == -1);

                if (u_removed && w_removed) { // no seed on either side of the de-cut edge
                    m_is_cut[k] = false; // remove the cut immediately, relabeling will come from somewhere else
                } else if (w_removed) { // relabel starting from w with u data
                    removed_ws_cut_edges.push_back(k);
                    start_vertices.push_back(w);
                    labels_to_propagate.push_back(m_labels(u_seed));
                    reference_seeds.push_back(u_seed);
                } else if (u_removed) { // relabel starting from u with w data
                    removed_ws_cut_edges.push_back(k);
                    start_vertices.push_back(u);
                    labels_to_propagate.push_back(m_labels(w_seed));
                    reference_seeds.push_back(w_seed);
                } else { // should not be possible
                    throw std::logic_error("Invariant violation: at least one of u_seed or w_seed should be != -1");
                }
            }

            if (start_vertices.empty()){ // no seed left, reset everything
                m_labels.fill(0);
                m_seed_map.fill(-1);
                return;
            }
            
            #ifdef HG_USE_TBB
            if (m_maximum_concurrency > 1) {
                relabel_component_from_seeds(start_vertices, labels_to_propagate, reference_seeds);
            } else
            #endif
            {
                for (index_t i = 0; i < (index_t)start_vertices.size(); i++) {
                    auto start = start_vertices[i];
                    auto label = labels_to_propagate[i];
                    auto reference_seed = reference_seeds[i];

                    relabel_component_from_seed(start, label, reference_seed);
                }
            }

            for (auto k : removed_ws_cut_edges)
                m_is_cut[k] = false;
        }

        /**
         * Return the current vertex labeling.
         *
         * The labeling is maintained incrementally by add_seeds and remove_seeds.
         * Vertices with no seed in their component are labeled 0 (background).
         *
         * @return 1d array of labels on graph vertices
         */
        const array_1d<index_t> &get_labeling() const {
            return m_labels;
        }

    private:

        /**
         * BFS from seed vertex start, labeling all reachable vertices (not crossing
         * cut edges) with the given label and seed, and updating the seed_map accordingly.
         */
        void relabel_component_from_seed(index_t start, index_t label, index_t seed) {
            m_labels[start] = label;
            m_seed_map[start] = seed;
            m_visited[start] = m_visited_generation;
            m_queue.push(start);
            while (!m_queue.empty()) {
                auto u = m_queue.front();
                m_queue.pop();
                for (auto e : out_edge_iterator(u, m_mst)) {
                    auto neighbor = target(e, m_mst);
                    auto edge_idx = index(e, m_mst);
                    if (!m_is_cut[edge_idx] && m_visited[neighbor] != m_visited_generation) {
                        m_visited[neighbor] = m_visited_generation;
                        m_labels[neighbor] = label;
                        m_seed_map[neighbor] = seed;
                        m_queue.push(neighbor);
                    }
                }
            }
        }

        #ifdef HG_USE_TBB

        /**
         * Partition a vector of values into a specified number of parts.
         *
         * @param values The vector of values to partition.
         * @param nb_processor The number of parts to create.
         * @param parts The vector to store the resulting parts.
         */
        void partition(const std::vector<index_t>& values, index_t nb_processor, std::vector<std::vector<index_t>>& parts) {

            parts.resize(nb_processor);

            const index_t nb_values = values.size();
            const index_t quotient = nb_values / nb_processor;
            const index_t rest = nb_values % nb_processor;
            
            tbb::parallel_for(index_t(0), nb_processor, index_t(1), [&](index_t i) {
                index_t size;
                index_t begin;
                if (i < rest) {
                    size = quotient + 1;
                    begin = i * size;
                } else {
                    size = quotient;
                    begin = i * size + rest;
                }
                parts[i].assign(values.begin() + begin,
                                values.begin() + begin + size);
            });
        }

        /**
         * BFS from seed vertex start, labeling all reachable vertices (not crossing
         * cut edges) with the given label and seed, and updating the seed_map accordingly.
         */
        void relabel_component_from_seeds(std::vector<index_t>& roots, std::vector<index_t>& root_labels, std::vector<index_t>& root_seeds) {
            hg_assert(roots.size() == root_labels.size(), "Each root must have a label.");
            hg_assert(roots.size() == root_seeds.size(), "Each root must have an origin seed.");
            
            std::vector<index_t> E = roots;

            for (size_t i = 0; i < roots.size(); ++i) {
                auto r = roots[i];
                m_labels(r) = root_labels[i];
                m_seed_map(r) = root_seeds[i];
                m_visited[r] = m_visited_generation;
            }

            const auto thread_count = static_cast<index_t>(m_maximum_concurrency);
            std::vector<std::vector<index_t>> S_parts(thread_count);
            std::vector<std::vector<index_t>> E_i(thread_count);
            std::vector<std::size_t> offsets(thread_count + 1);

            tbb::task_arena arena(static_cast<int>(thread_count));
            while (!E.empty()) {
                if (static_cast<index_t>(E.size()) > m_minimum_parallel_front_size) {
                    arena.execute([&]{
                        partition(E, thread_count, E_i);
                        tbb::parallel_for(index_t(0), thread_count, index_t(1), [&](index_t i) {
                            for (index_t l = 0; l <= m_num_parrallel_hops; l++) {
                                successor_labeling(E_i[i], S_parts[i]);
                                E_i[i].swap(S_parts[i]);
                            }
                        });
                        offsets[0] = 0;
                        for (index_t i = 0; i < thread_count; i++) offsets[i + 1] = offsets[i] + E_i[i].size();
                        E.resize(offsets[thread_count]);
                        tbb::parallel_for(index_t(0), thread_count, index_t(1), [&](index_t i) {
                            std::copy(E_i[i].begin(), E_i[i].end(), E.begin() + offsets[i]);
                        });
                    });
                }
                else {
                    successor_labeling(E, S_parts[0]);
                    E.swap(S_parts[0]);
                }
           }
        }

        void successor_labeling(const std::vector<index_t>& E, std::vector<index_t>& S) {
            S.clear();
            for (auto v : E) {
                const auto label = m_labels(v);
                const auto seed = m_seed_map(v);
                for (auto w : out_edge_iterator(v, m_mst)) {
                    auto neighbor = target(w, m_mst);
                    auto edge_idx = index(w, m_mst);
                    if (!m_is_cut[edge_idx] && m_visited[neighbor] != m_visited_generation) {
                        m_visited[neighbor] = m_visited_generation;
                        m_labels(neighbor) = label;
                        m_seed_map(neighbor) = seed;
                        S.push_back(neighbor);
                    }
                }
            }
        }
        #endif
        
        tree m_bpt;
        ugraph m_mst;
        index_t m_num_leaves;
        index_t m_root;

        // visitCount array on BPT nodes (Algorithm 1 & 2)
        std::vector<index_t> m_seed_count;

        // cut state of each MST edge
        std::vector<bool> m_is_cut;

        // cached queue for BFS calls in relabel_component_from_seed (pass 2a) to avoid repeated allocations
        std::queue<index_t> m_queue;

        // cached vertex labeling, updated locally by add_seeds/remove_seeds
        array_1d<index_t> m_labels;
        // seed that originated the label of each vertex (-1 if no seed, i.e. background)
        array_1d<index_t> m_seed_map;

        // BFS visited buffer with generation counter (zero-cost reset pattern)
        index_t m_visited_generation = 0;
        std::vector<index_t> m_visited;

        index_t m_minimum_parallel_front_size;
        index_t m_num_parrallel_hops;
        index_t m_maximum_concurrency;
    };

    /**
     * Create an incremental watershed cut object from an edge-weighted graph.
     *
     * Builds the canonical BPT and MST, then constructs an
     * incremental_watershed_cut object.
     *
     * @tparam graph_t
     * @tparam T
     * @param graph input graph (must be connected)
     * @param xedge_weights edge weights
     * @return an incremental_watershed_cut object
     */
    template<typename graph_t, typename T>
    auto make_incremental_watershed_cut(const graph_t &graph,
                                        const xt::xexpression<T> &xedge_weights) {
        HG_TRACE();
        auto &edge_weights = xedge_weights.derived_cast();
        hg_assert_edge_weights(graph, edge_weights);
        hg_assert_1d_array(edge_weights);

        auto bpt_res = bpt_canonical(graph, edge_weights);
        auto mst = subgraph_spanning(graph, bpt_res.mst_edge_map);
        return incremental_watershed_cut(std::move(bpt_res.tree), std::move(mst));
    }
}
