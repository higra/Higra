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

#include "tree.hpp"
#include "graph_core.hpp"

namespace hg {

    template<typename value_t>
    struct horizontal_cut_nodes {

        horizontal_cut_nodes(array_1d<index_t> &&_nodes,
                             value_t _altitude) :
                nodes(std::forward<array_1d<index_t> >(_nodes)),
                altitude(_altitude) {
        }

        template<typename tree_t>
        auto labelisation_leaves(const tree_t &tree) const {
            array_1d<bool> deleted({num_vertices(tree)}, true);
            xt::index_view(deleted, nodes) = false;
            return hg::reconstruct_leaf_data(tree,
                                             xt::arange<index_t>(num_vertices(tree)),
                                             deleted);
        }

        template<typename tree_t, typename T>
        auto reconstruct_leaf_data(const tree_t &tree,
                                   const xt::xexpression<T> &altitudes) const {
            array_1d<bool> deleted({num_vertices(tree)}, true);
            xt::index_view(deleted, nodes) = false;
            return hg::reconstruct_leaf_data(tree,
                                             altitudes,
                                             deleted);
        }

        template<typename tree_t, typename graph_t>
        auto graph_cut(const tree_t &tree,
                       const graph_t &leaf_graph) const {
            return hg::labelisation_2_graph_cut(leaf_graph, labelisation_leaves(tree));
        }

        array_1d<index_t> nodes;
        value_t altitude;
    };

    template<typename value_t>
    decltype(auto) make_horizontal_cut_nodes(array_1d<index_t> &&nodes,
                                             value_t altitude) {
        return horizontal_cut_nodes<value_t>(
                std::forward<array_1d<index_t> >(nodes),
                altitude);
    }

    template<typename tree_t, typename value_t>
    class horizontal_cut_explorer {
    public:
        using tree_type = tree_t;
        using value_type = value_t;

        template<typename T>
        horizontal_cut_explorer(const tree_t &tree, const xt::xexpression<T> &xaltitudes):
                m_original_tree(tree) {
            auto &altitudes = xaltitudes.derived_cast();
            hg_assert_node_weights(tree, altitudes);

            hg_assert(xt::count_nonzero(xt::view(altitudes, xt::range(0, num_leaves(tree))))() == 0,
                      "The altitude of the leaf nodes must be equal to 0.");

            hg_assert(xt::all(xt::view(altitudes, xt::range(num_leaves(tree), num_vertices(tree))) >=
                              static_cast<typename T::value_type>(0)),
                      "The altitude of the nodes must be greater than or equal to 0.");

            if (!is_sorted(altitudes)) {
                m_use_node_map = true;
                auto res = sort_hierarchy_with_altitudes(tree, altitudes);
                m_sorted_tree = std::move(res.tree);
                m_node_map = std::move(res.node_map);
                m_altitudes = xt::index_view(altitudes, m_node_map);
                init(m_sorted_tree, m_altitudes);
            } else {
                m_use_node_map = false;
                m_altitudes = altitudes;
                init(m_original_tree, m_altitudes);
            }
        }

        auto num_cuts() const {
            return m_num_regions_cuts.size();
        }

        auto num_regions_cut(index_t i) const {
            return m_num_regions_cuts[i];
        }

        const auto &num_regions_cuts() const {
            return m_num_regions_cuts;
        }

        auto altitude_cut(index_t i) const {
            return m_altitudes_cuts[i];
        }

        const auto &altitude_cuts() const {
            return m_altitudes_cuts;
        }

        inline
        auto horizontal_cut_from_index(index_t cut_index) const {
            auto num_regions = m_num_regions_cuts[cut_index];
            array_1d<index_t> nodes = array_1d<index_t>::from_shape({(size_t) num_regions});
            const tree &ct = (m_use_node_map) ? m_sorted_tree : m_original_tree;
            ct.compute_children();
            if (cut_index == 0) { // special case for single region partition
                nodes(0) = root(ct);
            } else {
                auto altitude = m_altitudes_cuts[cut_index];
                auto &range = m_range_nodes_cuts[cut_index];
                for (index_t i = range.first, j = 0; i <= range.second; i++) {
                    for (auto c: children_iterator(i, ct)) {
                        if (m_altitudes(c) <= altitude) {
                            nodes(j++) = c;
                        }
                    }
                }
            }

            if (m_use_node_map) {
                nodes = xt::index_view(m_node_map, nodes);
            }
            return make_horizontal_cut_nodes(std::move(nodes), m_altitudes_cuts[cut_index]);
        }

        auto horizontal_cut_from_altitude(value_t threshold) const {
            index_t cut_index;
            auto pos = std::upper_bound(m_altitudes_cuts.rbegin(),
                                        m_altitudes_cuts.rend(),
                                        threshold);
            if (pos == m_altitudes_cuts.rbegin()) {
                cut_index = m_altitudes_cuts.size() - 1;
            } else {
                cut_index = std::distance(pos, m_altitudes_cuts.rend());
            }
            return horizontal_cut_from_index(cut_index);
        }

        auto horizontal_cut_from_num_regions(index_t num_regions, bool at_least = true) const {
            index_t cut_index;
            auto pos = std::lower_bound(m_num_regions_cuts.begin(),
                                        m_num_regions_cuts.end(),
                                        num_regions);
            if (pos == m_num_regions_cuts.end()) {
                cut_index = m_num_regions_cuts.size() - 1;
            } else {
                cut_index = std::distance(m_num_regions_cuts.begin(), pos);
            }
            if (m_num_regions_cuts[cut_index] > num_regions && !at_least) {
                if (cut_index > 0) {
                    cut_index--;
                }
            }
            return horizontal_cut_from_index(cut_index);
        }

    private:

        template<typename T, typename E>
        void init(const T &t, const E &a) {
            auto min_alt_children = accumulate_parallel(t, a, accumulator_min());
            t.compute_children();

            // single region partition... edge case
            m_num_regions_cuts.push_back(1);
            m_altitudes_cuts.push_back(a(root(t)));
            m_range_nodes_cuts.push_back({invalid_index, invalid_index});
            index_t range_start = root(t);
            index_t range_end = root(t);
            index_t num_regions = num_children(root(t), t);
            auto current_threshold = a(range_start);

            while (current_threshold != 0 && range_start >= (index_t) num_leaves(t)) {

                while (min_alt_children(range_end) >= current_threshold) {
                    range_end--;
                }
                while (a(range_start - 1) >= current_threshold) {
                    range_start--;
                    num_regions += num_children(range_start, t) - 1;
                }

                current_threshold = a(range_start - 1);

                m_num_regions_cuts.push_back(num_regions);
                m_altitudes_cuts.push_back(current_threshold);
                m_range_nodes_cuts.push_back({range_start, range_end});
            }
        }

        template<typename T>
        static bool is_sorted(const T &a) {
            for (index_t i = 1; i < (index_t) a.size(); i++) {
                if (a[i - 1] > a[i]) {
                    return false;
                }
            }
            return true;
        }

        bool m_use_node_map;
        const tree_t &m_original_tree;
        hg::tree m_sorted_tree;
        array_1d<index_t> m_node_map;
        array_1d<value_t> m_altitudes;
        std::vector<index_t> m_num_regions_cuts;
        std::vector<value_t> m_altitudes_cuts;
        std::vector<std::pair<index_t, index_t>> m_range_nodes_cuts;
    };

    template<typename tree_t, typename T>
    decltype(auto) make_horizontal_cut_explorer(tree_t &&tree, T &&altitudes) {
        return horizontal_cut_explorer<tree_t, typename std::decay_t<T>::value_type>(
                std::forward<tree_t>(tree),
                std::forward<T>(altitudes));
    }


}
