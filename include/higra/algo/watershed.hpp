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
     * Reference:
     * Quentin Lebon, Josselin Lefevre, Jean Cousty, Benjamin Perret.
     * Interactive Segmentation With Incremental Watershed Cuts.
     * CIARP 2023.
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
                m_labels(xt::zeros<index_t>({(size_t)m_num_leaves})) {
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

            for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                auto v = (index_t)seed_vertices(i);
                auto l = (index_t)seed_labels(i);
                hg_assert(v >= 0 && v < m_num_leaves, "Seed vertex out of range.");
                hg_assert(l != 0, "Seed label must be non-zero (0 is reserved for background).");
                hg_assert(m_seed_labels.find(v) == m_seed_labels.end(),
                          "Vertex is already a seed.");

                m_seed_labels[v] = l;

                // Algorithm 1 (Lebon et al.): walk up BPT, increment visitCount
                index_t n = v;
                while (n != m_root && m_visit_count[n] != 2) {
                    n = m_bpt.parent(n);
                    m_visit_count[n] += 1;
                    if (m_visit_count[n] == 2) {
                        m_is_cut[n - m_num_leaves] = true;
                    }
                }

                // Local relabeling: BFS from v in its component
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

            for (index_t i = 0; i < (index_t)seed_vertices.size(); i++) {
                auto v = (index_t)seed_vertices(i);
                hg_assert(v >= 0 && v < m_num_leaves, "Seed vertex out of range.");
                hg_assert(m_seed_labels.find(v) != m_seed_labels.end(),
                          "Vertex is not a seed.");

                m_seed_labels.erase(v);

                // Algorithm 2 (Lebon et al.): walk up BPT, decrement visitCount
                index_t n = v;
                while (n != m_root && m_visit_count[n] != 1) {
                    n = m_bpt.parent(n);
                    m_visit_count[n] -= 1;
                    if (m_visit_count[n] == 1) {
                        m_is_cut[n - m_num_leaves] = false;
                    }
                }

                // Local relabeling: find merged component and relabel from remaining seeds
                relabel_merged_component(v);
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

        /**
         * After removing a seed at vertex v: find the merged component
         * (BFS from v respecting current cuts), reset labels to 0, then
         * relabel from all remaining seeds in the component.
         */
        void relabel_merged_component(index_t v) {
            // Step 1: BFS from v to find the merged component and collect seeds
            std::queue<index_t> queue;
            std::vector<index_t> component;
            std::vector<std::pair<index_t, index_t>> seeds_in_component;

            m_labels(v) = -1; // temporary marker
            queue.push(v);
            while (!queue.empty()) {
                auto u = queue.front();
                queue.pop();
                component.push_back(u);
                auto it = m_seed_labels.find(u);
                if (it != m_seed_labels.end()) {
                    seeds_in_component.push_back({u, it->second});
                }
                for (auto e : out_edge_iterator(u, m_mst)) {
                    auto neighbor = target(e, m_mst);
                    auto edge_idx = index(e, m_mst);
                    if (!m_is_cut[edge_idx] && m_labels(neighbor) != -1) {
                        m_labels(neighbor) = -1; // temporary marker
                        queue.push(neighbor);
                    }
                }
            }

            // Step 2: reset component labels to 0
            for (auto u : component) {
                m_labels(u) = 0;
            }

            // Step 3: relabel from remaining seeds in the component
            for (const auto &seed : seeds_in_component) {
                auto sv = seed.first;
                auto sl = seed.second;
                if (m_labels(sv) != 0) continue;
                m_labels(sv) = sl;
                queue.push(sv);
                while (!queue.empty()) {
                    auto u = queue.front();
                    queue.pop();
                    for (auto e : out_edge_iterator(u, m_mst)) {
                        auto neighbor = target(e, m_mst);
                        auto edge_idx = index(e, m_mst);
                        if (!m_is_cut[edge_idx] && m_labels(neighbor) == 0) {
                            m_labels(neighbor) = sl;
                            queue.push(neighbor);
                        }
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
