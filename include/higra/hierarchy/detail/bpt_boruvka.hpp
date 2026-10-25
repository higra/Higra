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
#include "higra/sorting.hpp"
#include "xtensor/generators/xbuilder.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <limits>
#include <utility>
#include <vector>

// Benchmark-only instrumentation is absent from ordinary library builds.
#ifdef HG_BPT_BENCH_DIAGNOSTICS
#include <chrono>
#include <string>
#endif

#ifdef HG_USE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#endif

namespace hg {
    namespace hierarchy_core_internal {

#ifdef HG_BPT_BENCH_DIAGNOSTICS
        struct bpt_boruvka_diagnostics {
            struct round { index_t components, active, accepted, remaining, jump_passes; };
            std::vector<round> rounds;
            std::vector<std::pair<std::string, double>> phases;
            std::vector<index_t> climbs;
            size_t tracked_scratch_high_water_bytes = 0;
            index_t jump_passes = 0;
#ifdef HG_BPT_BENCH_CAS
            std::atomic<size_t> cas_attempts{0}, cas_failures{0};
#endif
        };
        // Set only by the single-call diagnostic executable. Parallel workers
        // read this frozen pointer; each recorded climb has a unique writer.
        inline bpt_boruvka_diagnostics *bpt_bench_diagnostics = nullptr;
        struct bpt_bench_phase {
            const char *name;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            explicit bpt_bench_phase(const char *value) : name(value) {}
            ~bpt_bench_phase() {
                if (bpt_bench_diagnostics) bpt_bench_diagnostics->phases.emplace_back(name,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            }
        };
#define HG_BPT_PHASE(name) bpt_bench_phase bpt_phase_timer(name)
#else
#define HG_BPT_PHASE(name)
#endif

        // Experimental policies: numbering never changes the construction's
        // comparator. Frontends supplying explicit ranks pass their comparator
        // directly, bypassing scalar tie refinement altogether.
        enum class bpt_boruvka_numbering { exact, basin };
        enum class bpt_boruvka_tie_order { edge_id, reverse_edge_id };

        // Preconditions: validated scalar weights without NaNs. Infinities and
        // signed zeros use ordinary numeric comparison; IDs make ties total.
        template<typename weights_t>
        bool bpt_boruvka_scalar_less(const weights_t &weights, index_t a, index_t b,
                                     bpt_boruvka_tie_order ties = bpt_boruvka_tie_order::edge_id) {
            return weights(a) < weights(b) ||
                   (!(weights(b) < weights(a)) &&
                    (ties == bpt_boruvka_tie_order::edge_id ? a < b : a > b));
        }

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

        struct bpt_boruvka_counts {
            index_t retained = 0, selected = 0;

            bpt_boruvka_counts operator+(const bpt_boruvka_counts &other) const {
                return {retained + other.retained, selected + other.selected};
            }
        };

        // Reused across rounds. Allocate/resize only outside parallel bodies.
        struct bpt_boruvka_scratch {
            static constexpr index_t block_size = 1024;
            std::vector<uint8_t> edge_class;
            std::vector<bpt_boruvka_counts> counts, offsets;
#ifdef HG_USE_TBB
            std::unique_ptr<std::atomic<index_t>[]> saddle_atomic;
#endif

            bpt_boruvka_scratch(index_t n, index_t m)
                : edge_class(m), counts(m / block_size + (m % block_size != 0)), offsets(counts.size())
#ifdef HG_USE_TBB
                , saddle_atomic(new std::atomic<index_t>[n])
#endif
            {
                (void)n;
            }
        };

        // Classify (0=drop, 1=retain, 2=select), count, scan block counts, then
        // scatter. Only final scans write offsets; no scan writes edge outputs.
        // Input order is preserved, independently of scheduling or partitioning.
        template<typename classify_t, typename scatter_t>
        auto bpt_boruvka_compact(const std::vector<index_t> &active,
                                 bpt_boruvka_scratch &scratch, index_t selected_capacity,
                                 const classify_t &classify, const scatter_t &scatter) {
            const index_t count = active.size(), width = bpt_boruvka_scratch::block_size;
            const index_t blocks = count / width + (count % width != 0);
            parfor(0, blocks, [&](index_t block) {
                bpt_boruvka_counts counts;
                for (auto i = block * width; i < (std::min)(count, (block + 1) * width); ++i) {
                    auto kind = classify(active[i]);
                    scratch.edge_class[i] = kind;
                    counts.retained += kind == 1;
                    counts.selected += kind == 2;
                }
                scratch.counts[block] = counts;
            });
            bpt_boruvka_counts total;
#ifdef HG_USE_TBB
            total = tbb::parallel_scan(tbb::blocked_range<index_t>(0, blocks), total,
                [&](const auto &range, bpt_boruvka_counts prefix, bool final_scan) {
                    for (auto b = range.begin(); b != range.end(); ++b) {
                        if (final_scan) {
                            scratch.offsets[b] = prefix;
                        }
                        prefix = prefix + scratch.counts[b];
                    }
                    return prefix;
                }, std::plus<bpt_boruvka_counts>());
#else
            for (index_t b = 0; b < blocks; ++b) {
                scratch.offsets[b] = total;
                total = total + scratch.counts[b];
            }
#endif
            hg_assert(total.selected <= selected_capacity, "Too many selected forest edges.");
            parfor(0, blocks, [&](index_t block) {
                auto offset = scratch.offsets[block];
                for (auto i = block * width; i < (std::min)(count, (block + 1) * width); ++i) {
                    const auto kind = scratch.edge_class[i];
                    if (kind == 1) {
                        scatter(active[i], kind, offset.retained++);
                    } else if (kind == 2) {
                        scatter(active[i], kind, offset.selected++);
                    }
                }
            });
            return total;
        }

        template<typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_minima(const E1 &sources, const E2 &targets,
                               const array_1d<index_t> &labels, index_t component_count,
                               std::vector<index_t> &active, std::vector<index_t> &next_active,
                               bpt_boruvka_scratch &scratch, const edge_less_t &less) {
            HG_BPT_PHASE("minima_compaction");
            array_1d<index_t> saddle = xt::empty<index_t>({component_count});
            parfor(0, component_count, [&](index_t c) {
#ifdef HG_USE_TBB
                scratch.saddle_atomic[c].store(invalid_index, std::memory_order_relaxed);
#else
                saddle(c) = invalid_index;
#endif
            });
            // Immutable endpoints/labels/keys precede the joined initialization.
            // CAS publishes only an edge ID, not other memory. Relaxed atomics
            // suffice: modification order chooses the least candidate, and the
            // reduction joins before freezing ordinary saddles for consumers.
            // Correctness does not require lock-free atomics or bounded retries.
            auto minimize = [&](index_t c, index_t edge) {
#ifdef HG_USE_TBB
                auto &candidate = scratch.saddle_atomic[c];
                auto current = candidate.load(std::memory_order_relaxed);
                while (current == invalid_index || less(edge, current)) {
#ifdef HG_BPT_BENCH_CAS
                    if (bpt_bench_diagnostics) ++bpt_bench_diagnostics->cas_attempts;
#endif
                    if (candidate.compare_exchange_weak(current, edge, std::memory_order_relaxed)) {
                        break;
                    }
#ifdef HG_BPT_BENCH_CAS
                    if (bpt_bench_diagnostics) ++bpt_bench_diagnostics->cas_failures;
#endif
                }
#else
                if (saddle(c) == invalid_index || less(edge, saddle(c))) {
                    saddle(c) = edge;
                }
#endif
            };
            next_active.resize(active.size());
            auto counts = bpt_boruvka_compact(active, scratch, 0, [&](index_t edge) -> uint8_t {
                const auto a = labels(sources(edge)), b = labels(targets(edge));
                // Drop newly internal undecided edges BEFORE comparing keys.
                if (a == b) {
                    return 0;
                }
                minimize(a, edge);
                minimize(b, edge);
                return 1;
            }, [&](index_t edge, uint8_t, index_t destination) {
                next_active[destination] = edge;
            });
            next_active.resize(counts.retained);
            active.swap(next_active);
            parfor(0, component_count, [&](index_t c) {
#ifdef HG_USE_TBB
                saddle(c) = scratch.saddle_atomic[c].load(std::memory_order_relaxed);
#endif
                hg_assert(saddle(c) != invalid_index, "Input graph must be connected.");
            });
            return saddle;
        }

        // Minimum-edge successor graphs have only mutual-pair cycles: in a
        // longer cycle the largest edge cannot be minimal at its tail. Break
        // each pair at its smaller endpoint ID, then jump from frozen buffers.
        // Output: old-component -> dense new label, and new component count.
        template<typename E1, typename E2>
        auto bpt_boruvka_label_components(const E1 &sources, const E2 &targets,
                                          const array_1d<index_t> &labels,
                                          const array_1d<index_t> &saddle) {
            HG_BPT_PHASE("labeling");
            const index_t count = saddle.size();
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            index_t diagnostic_passes = 0;
#endif
            array_1d<index_t> successor = xt::empty<index_t>({count});
            array_1d<index_t> jump_a = xt::empty<index_t>({count});
            array_1d<index_t> jump_b = xt::empty<index_t>({count});
            array_1d<index_t> root_rank = xt::empty<index_t>({count});
            array_1d<index_t> component_to_new = xt::empty<index_t>({count});
            parfor(0, count, [&](index_t c) {
                auto edge = saddle(c), a = labels(sources(edge)), b = labels(targets(edge));
                hg_assert(a != b && (a == c || b == c), "Saddle must leave its component.");
                successor(c) = a == c ? b : a;
            });
            parfor(0, count, [&](index_t c) {
                auto next = successor(c);
                jump_a(c) = successor(next) == c && c < next ? c : next;
            });
            bool changed;
            do {
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                ++diagnostic_passes;
#endif
                auto jump = [&](index_t c) {
                    jump_b(c) = jump_a(jump_a(c));
                    return jump_b(c) != jump_a(c);
                };
#ifdef HG_USE_TBB
                changed = tbb::parallel_reduce(tbb::blocked_range<index_t>(0, count), false,
                    [&](const auto &range, bool any) {
                        for (auto c = range.begin(); c != range.end(); ++c) {
                            any = jump(c) || any;
                        }
                        return any;
                    }, std::logical_or<bool>());
#else
                changed = false;
                for (index_t c = 0; c < count; ++c) {
                    changed = jump(c) || changed;
                }
#endif
                // Every writer has joined. Neither pass modifies its input.
                std::swap(jump_a, jump_b);
            } while (changed);
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            if (bpt_bench_diagnostics) bpt_bench_diagnostics->jump_passes = diagnostic_passes;
#endif
            index_t new_count = 0;
#ifdef HG_USE_TBB
            new_count = tbb::parallel_scan(tbb::blocked_range<index_t>(0, count), index_t(0),
                [&](const auto &range, index_t prefix, bool final_scan) {
                    for (auto c = range.begin(); c != range.end(); ++c) {
                        if (final_scan) {
                            root_rank(c) = prefix;
                        }
                        prefix += jump_a(c) == c;
                    }
                    return prefix;
                }, std::plus<index_t>());
#else
            for (index_t c = 0; c < count; ++c) {
                root_rank(c) = new_count;
                new_count += jump_a(c) == c;
            }
#endif
            hg_assert(new_count <= count / 2, "Selected forest must halve component count.");
            parfor(0, count, [&](index_t c) {
                component_to_new(c) = root_rank(jump_a(c));
            });
            return std::make_pair(std::move(component_to_new), new_count);
        }

        // Preconditions: validated 1-D integral endpoints and n >= 1; less is
        // an immutable strict total order on original edge IDs for this call.
        // TBB invokes less concurrently; it must support simultaneous reads.
        template<typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_waterfall(const E1 &sources, const E2 &targets,
                                  index_t n, const edge_less_t &less) {
            HG_BPT_PHASE("waterfall_total");
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
            bpt_boruvka_scratch scratch(n, m);
            std::vector<index_t> active(m), next_active(m);
            parfor(0, m, [&](index_t edge) { active[edge] = edge; });
            index_t component_count = n, selected_count = 0, basin_offset = 0;
            auto saddle = bpt_boruvka_minima(sources, targets, labels, component_count, active, next_active, scratch, less);
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            auto tracked_bytes = [&]() {
                size_t bytes = sizeof(index_t) * (history.selected_edges.size() + history.leaf_parent.size() +
                    history.raw_parent.size() + history.raw_basin.size() + edge_to_slot.size() + labels.size() +
                    active.capacity() + next_active.capacity() + saddle.size()) + scratch.edge_class.capacity() +
                    sizeof(bpt_boruvka_counts) * (scratch.counts.capacity() + scratch.offsets.capacity());
#ifdef HG_USE_TBB
                bytes += sizeof(std::atomic<index_t>) * size_t(n);
#endif
                return bytes;
            };
            auto note_scratch = [&](size_t bytes) {
                if (bpt_bench_diagnostics) bpt_bench_diagnostics->tracked_scratch_high_water_bytes =
                    (std::max)(bpt_bench_diagnostics->tracked_scratch_high_water_bytes, bytes);
            };
            note_scratch(tracked_bytes());
#endif
            parfor(0, n, [&](index_t v) { history.leaf_parent(v) = saddle(v); });
            while (component_count > 1) {
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                const auto diagnostic_active = active.size();
                const auto diagnostic_start = std::chrono::steady_clock::now();
#endif
                const auto first_slot = selected_count;
                next_active.resize(active.size());
                const auto counts = bpt_boruvka_compact(active, scratch, n - 1 - first_slot,
                    [&](index_t edge) -> uint8_t {
                        auto a = labels(sources(edge)), b = labels(targets(edge));
                        return edge == saddle(a) || edge == saddle(b) ? 2 : 1;
                    }, [&](index_t edge, uint8_t kind, index_t destination) {
                        if (kind == 2) {
                            auto slot = first_slot + destination;
                            history.selected_edges(slot) = edge;
                            edge_to_slot(edge) = slot;
                        } else {
                            next_active[destination] = edge;
                        }
                    });
                selected_count += counts.selected;
                next_active.resize(counts.retained);
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                if (bpt_bench_diagnostics) bpt_bench_diagnostics->phases.emplace_back("selection_compaction",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - diagnostic_start).count());
#endif
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                // Five simultaneously live labeler arrays, including its result.
                note_scratch(tracked_bytes() + 5 * sizeof(index_t) * size_t(component_count));
#endif
                auto labeling = bpt_boruvka_label_components(sources, targets, labels, saddle);
                const auto new_count = labeling.second;
                hg_assert(new_count <= component_count / 2, "Selected forest must halve component count.");
                hg_assert(selected_count - first_slot == component_count - new_count,
                          "Selected edges must form a forest.");
                {
                    HG_BPT_PHASE("vertex_relabeling");
                    parfor(0, n, [&](index_t v) {
                        labels(v) = labeling.first(labels(v));
                    });
                    active.swap(next_active);
                }
                if (new_count > 1) {
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                    // Old saddle survives until the new return value is assigned.
                    note_scratch(tracked_bytes() + sizeof(index_t) * (labeling.first.size() + size_t(new_count)));
#endif
                    saddle = bpt_boruvka_minima(sources, targets, labels, new_count, active, next_active, scratch, less);
                }
                {
                    HG_BPT_PHASE("history_assignment");
                    hg_assert(basin_offset <= n - 1 - new_count, "Basin count exceeds capacity.");
                    parfor(first_slot, selected_count, [&](index_t slot) {
                        auto basin = labels(sources(history.selected_edges(slot)));
                        history.raw_basin(slot) = basin_offset + basin;
                        history.raw_parent(slot) = new_count == 1 ? invalid_index : saddle(basin);
                    });
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                    if (bpt_bench_diagnostics) bpt_bench_diagnostics->rounds.push_back({component_count,
                        static_cast<index_t>(diagnostic_active), counts.selected, new_count, bpt_bench_diagnostics->jump_passes});
#endif
                    basin_offset += new_count;
                    component_count = new_count;
                }
            }
            hg_assert(selected_count == n - 1, "Input graph must be connected.");
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            bpt_bench_phase slot_conversion_timer("history_slot_conversion");
#endif
            // Original-edge references are converted only after every saddle has
            // received its selected slot. No inverse edge map survives this stage.
            parfor(0, n, [&](index_t v) {
                auto &edge = history.leaf_parent(v);
                hg_assert(edge_to_slot(edge) != invalid_index, "Leaf saddle must be selected.");
                edge = edge_to_slot(edge);
            });
            parfor(0, n - 1, [&](index_t slot) {
                auto &edge = history.raw_parent(slot);
                if (edge != invalid_index) {
                    hg_assert(edge_to_slot(edge) != invalid_index, "Parent saddle must be selected.");
                    edge = edge_to_slot(edge);
                }
            });
            return history;
        }

        template<typename edge_less_t>
        auto bpt_boruvka_expand(const bpt_boruvka_history &history, const edge_less_t &less) {
            HG_BPT_PHASE("expansion");
            const index_t count = history.selected_edges.size();
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            if (bpt_bench_diagnostics) bpt_bench_diagnostics->climbs.resize(count);
#endif
            array_1d<index_t> parents = xt::empty<index_t>({count});
            array_1d<index_t> basins = xt::empty<index_t>({count});
            parfor(0, count, [&](index_t slot) {
                auto node = slot;
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                index_t diagnostic_climbs = 0;
#endif
                auto ancestor = history.raw_parent(node);
                // Read the immutable raw snapshot. Compare every ancestor with
                // the fixed starting edge, and move its basin together with it.
                while (ancestor != invalid_index &&
                       less(history.selected_edges(ancestor), history.selected_edges(slot))) {
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                    ++diagnostic_climbs;
#endif
                    node = ancestor;
                    ancestor = history.raw_parent(node);
                }
#ifdef HG_BPT_BENCH_DIAGNOSTICS
                if (bpt_bench_diagnostics) bpt_bench_diagnostics->climbs[slot] = diagnostic_climbs;
#endif
                parents(slot) = ancestor;
                basins(slot) = history.raw_basin(node);
            });
            return std::make_pair(std::move(parents), std::move(basins));
        }

        template<typename edge_less_t>
        auto bpt_boruvka_export(const bpt_boruvka_history &history, const edge_less_t &less,
                                bpt_boruvka_numbering numbering = bpt_boruvka_numbering::exact) {
            const index_t n = history.leaf_parent.size(), count = history.selected_edges.size();
            auto expanded = bpt_boruvka_expand(history, less);
            auto &edge_parent = expanded.first;
            const auto &basins = expanded.second;
            auto slot_less = [&](index_t a, index_t b) {
                return less(history.selected_edges(a), history.selected_edges(b));
            };
            array_1d<index_t> group_order = xt::arange<index_t>(count);
            {
                HG_BPT_PHASE("basin_sort");
                hg::sort(group_order.begin(), group_order.end(), [&](index_t a, index_t b) {
                    return basins(a) < basins(b) || (basins(a) == basins(b) && slot_less(a, b));
                });
            }
            {
                HG_BPT_PHASE("rewiring");
                parfor(0, (std::max)(index_t(0), count - 1), [&](index_t i) {
                    auto a = group_order(i), b = group_order(i + 1);
                    if (basins(a) == basins(b)) {
                        edge_parent(a) = b;
                    }
                });
            }
            // Basin order is topological even after non-monotone relocation:
            // every raw parent was selected in a later round, hence has a
            // strictly larger raw basin ID. A climb ends at r with parent p;
            // expanded_basin[u] = raw_basin[r] < raw_basin[p] <=
            // expanded_basin[p]. Retained cross-basin parents therefore occur
            // later. Rewired same-basin parents occur later under the same
            // total edge order. Undefined parents all relocate to the final
            // (largest) terminal basin; its chain has a sole, last root.
            // Leaves precede all selected slots. No topological sort is needed.
            array_1d<index_t> node_order = std::move(group_order);
            if (numbering == bpt_boruvka_numbering::exact) {
                HG_BPT_PHASE("exact_numbering_sort");
                hg::sort(node_order.begin(), node_order.end(), slot_less);
            }
            HG_BPT_PHASE("export_mapping");
#ifdef HG_BPT_BENCH_DIAGNOSTICS
            if (bpt_bench_diagnostics) {
                // History + expanded arrays + permutation + inverse node map.
                // Final parents/MST map are outputs, not tracked scratch.
                const size_t bytes = sizeof(index_t) * (history.leaf_parent.size() + 7 * size_t(count));
                bpt_bench_diagnostics->tracked_scratch_high_water_bytes =
                    (std::max)(bpt_bench_diagnostics->tracked_scratch_high_water_bytes, bytes);
            }
#endif
            array_1d<index_t> slot_to_node = xt::empty<index_t>({count});
            array_1d<index_t> mst_edge_map = xt::empty<index_t>({count});
            parfor(0, count, [&](index_t i) {
                slot_to_node(node_order(i)) = n + i;
                mst_edge_map(i) = history.selected_edges(node_order(i));
            });
            array_1d<index_t> parents = xt::empty<index_t>({n + count});
            if (n == 1) {
                parents(0) = 0;
            } else {
                parfor(0, n, [&](index_t v) {
                    parents(v) = slot_to_node(history.leaf_parent(v));
                });
                parfor(0, count, [&](index_t slot) {
                    auto node = slot_to_node(slot), p = edge_parent(slot);
                    hg_assert(p != invalid_index || node == n + count - 1, "Only the root can lack a parent.");
                    parents(node) = p == invalid_index ? node : slot_to_node(p);
                    hg_assert(p == invalid_index || parents(node) > node,
                              "Export must put parents after children.");
                });
            }
            return std::make_pair(std::move(parents), std::move(mst_edge_map));
        }

        // Exact-numbering convenience helper.
        template<typename edge_less_t>
        auto bpt_boruvka_export_exact(const bpt_boruvka_history &history, const edge_less_t &less) {
            return bpt_boruvka_export(history, less);
        }

        // Endpoint/comparator entry point shared by scalar and future frontends.
        // Returns owning (parents, MST edge map), with exact stable-Kruskal IDs
        // by default. Basin numbering preserves the edge-labeled tree and leaf
        // IDs, but internal altitudes need not be globally sorted. The supplied
        // comparator is authoritative in both modes, including explicit ranks.
        template<typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_from_edge_order(const xt::xexpression<E1> &xsources,
                                        const xt::xexpression<E2> &xtargets,
                                        index_t n, const edge_less_t &less,
                                        bpt_boruvka_numbering numbering = bpt_boruvka_numbering::exact) {
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
            return bpt_boruvka_export(history, less, numbering);
        }
    }
}

#undef HG_BPT_PHASE
