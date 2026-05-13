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
#include <stack>
#include <queue>
#include <unordered_map>

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
     *   Construction:    O(n log n) with n the number of edges in the input
     *                    graph (dominated by the sort in bpt_canonical).
     *   add_seeds(K):    O(K * d + S) where d is the height of the BPT
     *                    (O(log N) for balanced trees, O(N) worst case, with
     *                    N the number of vertices) and S is the total size of
     *                    the K MST-forest components relabeled in pass 2.
     *                    By the visit_count==2 invariant these K components
     *                    are pairwise disjoint, hence S <= N.
     *   remove_seeds(K): O(K * d + S') where d is as above and S' is the
     *                    total work performed by the BFS calls of pass 2a
     *                    and 2b. When the de-cuts of the batch touch mostly
     *                    disjoint regions, S' <= N; in the worst case of
     *                    cascading merges within the same batch (each removal
     *                    extends a growing super-component), S' can grow to
     *                    O(K * N).
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
         * This ordering is guaranteed when the MST is built from bpt_canonical
         * via subgraph_spanning.
         *
         * @param bpt canonical binary partition tree
         * @param mst minimum spanning tree of the original graph (as ugraph)
         */
        incremental_watershed_cut(tree bpt, ugraph mst) :
                m_bpt(std::move(bpt)),
                m_mst(std::move(mst)),
                m_num_leaves((index_t)num_leaves(m_bpt)),
                m_root((index_t)hg::num_vertices(m_bpt) - 1),
                m_visit_count(hg::num_vertices(m_bpt), 0),
                m_is_cut(m_num_leaves - 1, false),
                m_labels(xt::zeros<index_t>({(size_t)m_num_leaves})),
                m_visited(m_num_leaves, 0) {
            HG_TRACE();
            hg_assert((index_t)num_vertices(m_mst) == m_num_leaves,
                      "MST must have the same number of vertices as leaves in the BPT.");
            hg_assert((index_t)num_edges(m_mst) == m_num_leaves - 1,
                      "MST must have exactly num_leaves - 1 edges.");
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

            // Pass 1: register seeds and update BPT cut state (Algorithm 1, Lebon et al.).
            for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                auto v = (index_t)seed_vertices(i);
                auto l = (index_t)seed_labels(i);
                hg_assert(v >= 0 && v < m_num_leaves, "Seed vertex out of range.");
                hg_assert(l != 0, "Seed label must be non-zero (0 is reserved for background).");
                hg_assert(m_seed_labels.find(v) == m_seed_labels.end(),
                          "Vertex is already a seed.");

                m_seed_labels[v] = l;

                index_t n = v;
                while (n != m_root && m_visit_count[n] != 2) {
                    n = m_bpt.parent(n);
                    m_visit_count[n] += 1;
                    if (m_visit_count[n] == 2) {
                        m_is_cut[n - m_num_leaves] = true;
                    }
                }
            }
            // Pass 2: relabel each seed's component once all cuts of the batch are stable.
            for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                auto v = (index_t)seed_vertices(i);
                auto l = (index_t)seed_labels(i);
                relabel_component_from_seed(v, l);
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
            HG_TRACE();
            auto &seed_vertices = xseed_vertices.derived_cast();
            hg_assert_1d_array(seed_vertices);

            std::vector<index_t> decut_edges;   // MST edge indices that just got un-cut
            std::vector<index_t> lone_seeds;    // seeds whose walk-up reached the root without a 2->1 transition

            // Pass 1: walk up the BPT for each removed seed, decrement visitCount,
            // collect the de-cut MST edge indices in batch insertion order (i.e.
            // the order seeds appear in the input array; each seed produces at
            // most one de-cut, at the first 2->1 transition along its walk-up).
            // The cut state m_is_cut is NOT updated here; the update is deferred
            // to Pass 2 so that each de-cut can be processed in isolation (the
            // BFS for de-cut k sees only the de-cuts processed before k, which
            // bounds the relabeling to the freshly merged region). The
            // correctness of the resulting labeling does not depend on the
            // relative BPT depth of the de-cuts within the batch.
            for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                auto v = (index_t)seed_vertices(i);
                hg_assert(v >= 0 && v < m_num_leaves, "Seed vertex out of range.");
                hg_assert(m_seed_labels.find(v) != m_seed_labels.end(),
                          "Vertex is not a seed.");

                m_seed_labels.erase(v);

                index_t n = v;
                bool produced_decut = false;
                while (n != m_root) {
                    n = m_bpt.parent(n);
                    m_visit_count[n] -= 1;
                    if (m_visit_count[n] == 1) {
                        decut_edges.push_back(n - m_num_leaves);
                        produced_decut = true;
                        break;
                    }
                }
                if (!produced_decut) {
                    lone_seeds.push_back(v);
                }
            }

            // Pass 2a: for each de-cut MST edge (in collection order), reactivate
            // the edge in the forest and propagate the surviving label across the
            // newly merged region. The surviving side is identified from seed
            // presence in each side component (before edge reactivation), not from
            // endpoint label values alone.
            for (auto k : decut_edges) {
                const auto &e = edge_from_index(k, m_mst);
                auto u = source(e, m_mst);
                auto w = target(e, m_mst);

                auto lu_seed = component_seed_label(u);
                auto lw_seed = component_seed_label(w);

                m_is_cut[k] = false;

                index_t target_label;
                index_t start_vertex;
                if (lu_seed != 0 && lw_seed == 0) {
                    target_label = lu_seed;
                    start_vertex = w;
                } else if (lw_seed != 0 && lu_seed == 0) {
                    target_label = lw_seed;
                    start_vertex = u;
                } else if (lu_seed == 0 && lw_seed == 0) {
                    target_label = 0;
                    start_vertex = u;
                } else if (lu_seed == lw_seed) {
                    target_label = lu_seed;
                    start_vertex = w;
                } else {
                    // Both sides have surviving seeds with different labels.
                    // This branch is unreachable: a de-cut at BPT node n corresponds
                    // to a visitCount(n): 2 -> 1 transition, so exactly one of the
                    // two BPT subtrees of n lost its last active seed; the other
                    // still has at least one. Earlier de-cuts in this batch can
                    // extend a side's MST-forest component beyond a single BPT
                    // subtree, but every active seed reachable from a given side
                    // of edge k shares one label by induction on the 2->1
                    // invariant (Lebon et al., Algorithm 2). The release-mode
                    // fallback below avoids leaving the labeling in an
                    // inconsistent state if the invariant ever breaks.
                    hg_assert(false, "Both sides of de-cut edge have different seed labels");
                    target_label = 0;
                    start_vertex = u;
                }
                relabel_component_from_seed(start_vertex, target_label);
            }

            // Pass 2b: a lone seed had no 2->1 transition during its walk-up,
            // meaning it was the only seed contributing to visitCount along its
            // entire path to the root. Its component therefore had no other seed
            // and must now revert to background (0).
            for (auto v : lone_seeds) {
                relabel_component_from_seed(v, 0);
            }
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
         * Return a seed label present in the connected component of start,
         * considering current cut state. Returns 0 if no seed is found.
         */
        index_t component_seed_label(index_t start) {
            m_visited_generation++;
            std::queue<index_t> queue;
            queue.push(start);
            m_visited[(size_t)start] = m_visited_generation;

            while (!queue.empty()) {
                auto u = queue.front();
                queue.pop();

                auto it = m_seed_labels.find(u);
                if (it != m_seed_labels.end()) {
                    return it->second;
                }

                for (auto e : out_edge_iterator(u, m_mst)) {
                    auto neighbor = target(e, m_mst);
                    auto edge_idx = index(e, m_mst);
                    if (!m_is_cut[edge_idx] && m_visited[(size_t)neighbor] != m_visited_generation) {
                        m_visited[(size_t)neighbor] = m_visited_generation;
                        queue.push(neighbor);
                    }
                }
            }
            return 0;
        }

        /**
         * BFS from seed vertex v, labeling all reachable vertices (not crossing
         * cut edges) with the given label.
         *
         * After add_seeds creates new cuts, v's component is guaranteed to
         * contain no other seed (the binary tree structure of the BPT ensures
         * that visitCount == 2 at the LCA of v and any existing seed).
         */
        void relabel_component_from_seed(index_t v, index_t label) {
            std::queue<index_t> queue;
            m_labels(v) = label;
            queue.push(v);
            while (!queue.empty()) {
                auto u = queue.front();
                queue.pop();
                for (auto e : out_edge_iterator(u, m_mst)) {
                    auto neighbor = target(e, m_mst);
                    auto edge_idx = index(e, m_mst);
                    if (!m_is_cut[edge_idx] && m_labels(neighbor) != label) {
                        m_labels(neighbor) = label;
                        queue.push(neighbor);
                    }
                }
            }
        }

        tree m_bpt;
        ugraph m_mst;
        index_t m_num_leaves;
        index_t m_root;

        // visitCount array on BPT nodes (Algorithm 1 & 2)
        std::vector<index_t> m_visit_count;

        // cut state of each MST edge
        std::vector<bool> m_is_cut;

        // current seeds: vertex -> label
        std::unordered_map<index_t, index_t> m_seed_labels;

        // cached vertex labeling, updated locally by add_seeds/remove_seeds
        array_1d<index_t> m_labels;

        // BFS visited buffer with generation counter (zero-cost reset pattern)
        index_t m_visited_generation = 0;
        std::vector<index_t> m_visited;
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
