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

#include "higra/structure/array.hpp"
#include "higra/structure/unionfind.hpp"
#include "higra/sorting.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace hg {
    namespace hierarchy_core_internal {

        inline void bpt_boruvka_check_sizes(index_t num_vertices, size_t num_edges) {
            hg_assert(num_vertices > 0, "Input graph must have at least one vertex.");
            hg_assert(num_vertices <= (std::numeric_limits<index_t>::max)() / 2 + 1,
                      "BPT node count exceeds index capacity.");
            const auto num_nodes = num_vertices + (num_vertices - 1);
            const auto capacity = std::vector<index_t>().max_size();
            hg_assert(static_cast<uintmax_t>(num_nodes) <= capacity &&
                      num_edges <= capacity &&
                      num_edges <= static_cast<uintmax_t>((std::numeric_limits<index_t>::max)()),
                      "BPT array size exceeds allocation/index capacity.");
        }

        inline auto bpt_boruvka_invalid_indices(index_t count) {
            array_1d<index_t> result = xt::empty<index_t>({count});
            result.fill(invalid_index);
            return result;
        }

        // All history indices are selected slots, except selected_edges itself
        // (slot -> original edge). Basins are globally unique chronological IDs.
        struct bpt_boruvka_history {
            array_1d<index_t> selected_edges;
            array_1d<index_t> leaf_parent;
            array_1d<index_t> raw_parent;
            array_1d<index_t> raw_basin;
        };

        template<typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_minima(const E1 &sources, const E2 &targets,
                               const array_1d<index_t> &labels, index_t component_count,
                               std::vector<index_t> &active, const edge_less_t &less) {
            array_1d<index_t> saddle = bpt_boruvka_invalid_indices(component_count);
            size_t retained = 0;
            for (auto edge: active) {
                const auto a = labels(sources(edge)), b = labels(targets(edge));
                // Undecided cycle edges can become internal after contraction.
                if (a == b) {
                    continue;
                }
                active[retained++] = edge;
                for (auto c: {a, b}) {
                    if (saddle(c) == invalid_index || less(edge, saddle(c))) {
                        saddle(c) = edge;
                    }
                }
            }
            active.resize(retained);
            for (auto edge: saddle) {
                hg_assert(edge != invalid_index, "Input graph must be connected.");
            }
            return saddle;
        }

        // Replaceable serial forest labeler. Input: frozen old vertex labels and
        // one outgoing saddle (original edge ID) per old component. Output:
        // old-component -> dense new component, and the new component count.
        template<typename E1, typename E2>
        auto bpt_boruvka_label_components(const E1 &sources, const E2 &targets,
                                          const array_1d<index_t> &labels,
                                          const array_1d<index_t> &saddle) {
            const index_t count = saddle.size();
            union_find forest(count);
            for (auto edge: saddle) {
                auto a = forest.find(labels(sources(edge)));
                auto b = forest.find(labels(targets(edge)));
                if (a != b) {
                    forest.link(a, b);
                }
            }
            array_1d<index_t> root_label = bpt_boruvka_invalid_indices(count);
            array_1d<index_t> component_to_new = xt::empty<index_t>({count});
            index_t new_count = 0;
            for (index_t c = 0; c < count; ++c) {
                auto root = forest.find(c);
                if (root_label(root) == invalid_index) {
                    root_label(root) = new_count++;
                }
                component_to_new(c) = root_label(root);
            }
            return std::make_pair(std::move(component_to_new), new_count);
        }

        // Preconditions: validated 1-D integral endpoints and n >= 1; less is
        // an immutable strict total order on original edge IDs for this call.
        template<typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_waterfall(const E1 &sources, const E2 &targets,
                                  index_t n, const edge_less_t &less) {
            const index_t m = sources.size();
            bpt_boruvka_history history{xt::empty<index_t>({n - 1}),
                                        bpt_boruvka_invalid_indices(n),
                                        bpt_boruvka_invalid_indices(n - 1),
                                        xt::empty<index_t>({n - 1})};
            if (n == 1) {
                return history;
            }
            array_1d<index_t> edge_to_slot = bpt_boruvka_invalid_indices(m);
            array_1d<index_t> labels = xt::arange<index_t>(n);
            std::vector<index_t> active, next_active;
            active.reserve(m);
            next_active.reserve(m);
            for (index_t edge = 0; edge < m; ++edge) {
                active.push_back(edge);
            }
            index_t component_count = n, selected_count = 0, basin_offset = 0;
            auto saddle = bpt_boruvka_minima(sources, targets, labels, component_count, active, less);
            history.leaf_parent = saddle;
            while (component_count > 1) {
                const auto first_slot = selected_count;
                next_active.clear();
                for (auto edge: active) {
                    auto a = labels(sources(edge)), b = labels(targets(edge));
                    if (edge == saddle(a) || edge == saddle(b)) {
                        hg_assert(selected_count < n - 1, "Too many selected forest edges.");
                        history.selected_edges(selected_count) = edge;
                        edge_to_slot(edge) = selected_count++;
                    } else {
                        next_active.push_back(edge);
                    }
                }
                auto labeling = bpt_boruvka_label_components(sources, targets, labels, saddle);
                const auto new_count = labeling.second;
                hg_assert(new_count <= component_count / 2, "Selected forest must halve component count.");
                hg_assert(selected_count - first_slot == component_count - new_count,
                          "Selected edges must form a forest.");
                for (auto &label: labels) {
                    label = labeling.first(label);
                }
                active.swap(next_active);
                if (new_count > 1) {
                    saddle = bpt_boruvka_minima(sources, targets, labels, new_count, active, less);
                }
                hg_assert(basin_offset <= n - 1 - new_count, "Basin count exceeds capacity.");
                for (auto slot = first_slot; slot < selected_count; ++slot) {
                    auto basin = labels(sources(history.selected_edges(slot)));
                    history.raw_basin(slot) = basin_offset + basin;
                    history.raw_parent(slot) = new_count == 1 ? invalid_index : saddle(basin);
                }
                basin_offset += new_count;
                component_count = new_count;
            }
            hg_assert(selected_count == n - 1, "Input graph must be connected.");
            // Original-edge references are converted only after every saddle has
            // received its selected slot. No inverse edge map survives this stage.
            for (auto &edge: history.leaf_parent) {
                hg_assert(edge_to_slot(edge) != invalid_index, "Leaf saddle must be selected.");
                edge = edge_to_slot(edge);
            }
            for (auto &edge: history.raw_parent) {
                if (edge != invalid_index) {
                    hg_assert(edge_to_slot(edge) != invalid_index, "Parent saddle must be selected.");
                    edge = edge_to_slot(edge);
                }
            }
            return history;
        }

        template<typename edge_less_t>
        auto bpt_boruvka_expand(const bpt_boruvka_history &history, const edge_less_t &less) {
            const index_t count = history.selected_edges.size();
            array_1d<index_t> parents = xt::empty<index_t>({count});
            array_1d<index_t> basins = xt::empty<index_t>({count});
            for (index_t slot = 0; slot < count; ++slot) {
                auto node = slot;
                auto ancestor = history.raw_parent(node);
                // Read the immutable raw snapshot. Compare every ancestor with
                // the fixed starting edge, and move its basin together with it.
                while (ancestor != invalid_index &&
                       less(history.selected_edges(ancestor), history.selected_edges(slot))) {
                    node = ancestor;
                    ancestor = history.raw_parent(node);
                }
                parents(slot) = ancestor;
                basins(slot) = history.raw_basin(node);
            }
            return std::make_pair(std::move(parents), std::move(basins));
        }

        template<typename edge_less_t>
        auto bpt_boruvka_export_exact(const bpt_boruvka_history &history, const edge_less_t &less) {
            const index_t n = history.leaf_parent.size(), count = history.selected_edges.size();
            auto expanded = bpt_boruvka_expand(history, less);
            auto &edge_parent = expanded.first;
            const auto &basins = expanded.second;
            auto slot_less = [&](index_t a, index_t b) {
                return less(history.selected_edges(a), history.selected_edges(b));
            };
            array_1d<index_t> group_order = xt::arange<index_t>(count);
            hg::sort(group_order.begin(), group_order.end(), [&](index_t a, index_t b) {
                return basins(a) < basins(b) || (basins(a) == basins(b) && slot_less(a, b));
            });
            for (index_t i = 0; i + 1 < count; ++i) {
                auto a = group_order(i), b = group_order(i + 1);
                if (basins(a) == basins(b)) {
                    edge_parent(a) = b;
                }
            }
            array_1d<index_t> global_order = xt::arange<index_t>(count);
            hg::sort(global_order.begin(), global_order.end(), slot_less);
            array_1d<index_t> slot_to_node = xt::empty<index_t>({count});
            array_1d<index_t> mst_edge_map = xt::empty<index_t>({count});
            for (index_t i = 0; i < count; ++i) {
                slot_to_node(global_order(i)) = n + i;
                mst_edge_map(i) = history.selected_edges(global_order(i));
            }
            array_1d<index_t> parents = xt::empty<index_t>({n + count});
            if (n == 1) {
                parents(0) = 0;
            } else {
                for (index_t v = 0; v < n; ++v) {
                    parents(v) = slot_to_node(history.leaf_parent(v));
                }
                for (index_t slot = 0; slot < count; ++slot) {
                    auto node = slot_to_node(slot), p = edge_parent(slot);
                    hg_assert(p != invalid_index || node == n + count - 1, "Only the root can lack a parent.");
                    parents(node) = p == invalid_index ? node : slot_to_node(p);
                    hg_assert(parents(node) > node || node == n + count - 1,
                              "Export must put parents after children.");
                }
            }
            return std::make_pair(std::move(parents), std::move(mst_edge_map));
        }

        // Endpoint/comparator entry point shared by scalar and future frontends.
        // Returns owning (parents, MST edge map), with exact stable-Kruskal IDs.
        template<typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_from_edge_order(const xt::xexpression<E1> &xsources,
                                        const xt::xexpression<E2> &xtargets,
                                        index_t n, const edge_less_t &less) {
            const auto &sources = xsources.derived_cast();
            const auto &targets = xtargets.derived_cast();
            hg_assert_1d_array(sources);
            hg_assert_same_shape(sources, targets);
            hg_assert_integral_value_type(sources);
            hg_assert_integral_value_type(targets);
            bpt_boruvka_check_sizes(n, sources.size());
            for (size_t e = 0; e < sources.size(); ++e) {
                hg_assert(sources(e) >= 0 && targets(e) >= 0 &&
                          static_cast<uintmax_t>(sources(e)) < static_cast<uintmax_t>(n) &&
                          static_cast<uintmax_t>(targets(e)) < static_cast<uintmax_t>(n),
                          "Edge endpoints must be valid vertex indices.");
            }
            auto history = bpt_boruvka_waterfall(sources, targets, n, less);
            return bpt_boruvka_export_exact(history, less);
        }
    }
}
