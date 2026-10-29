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
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <limits>
#include <utility>
#include <vector>

#ifdef HG_USE_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#endif

namespace hg {
    namespace hierarchy_core_internal {

        // Numbering never changes the construction's
        // comparator. Frontends supplying explicit ranks pass their comparator
        // directly, bypassing scalar tie refinement altogether.
        enum class bpt_boruvka_numbering { exact, basin };
        enum class bpt_boruvka_traversal { full_scan, active };

        // Preconditions: validated scalar weights without NaNs. Infinities and
        // signed zeros use ordinary numeric comparison; IDs make ties total.
        template<typename weights_t>
        bool bpt_boruvka_scalar_less(const weights_t &weights, index_t a, index_t b) {
            return weights(a) < weights(b) ||
                   (!(weights(b) < weights(a)) && a < b);
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

        // Fixed per-call capacity with a separately tracked initialized prefix.
        // new T[count] starts scalar lifetimes without value-initializing them.
        // Compaction writes every live destination before publishing its size.
        class bpt_boruvka_active_edges {
            std::unique_ptr<index_t[]> m_data;
            size_t m_capacity, m_size;
        public:
            explicit bpt_boruvka_active_edges(size_t count)
                : m_data(new index_t[count]), m_capacity(count), m_size(count) {}
            size_t size() const { return m_size; }
            size_t capacity() const { return m_capacity; }
            index_t &operator[](size_t i) { return m_data[i]; }
            const index_t &operator[](size_t i) const { return m_data[i]; }
            void resize(size_t count) {
                assert(count <= m_capacity);
                m_size = count;
            }
            void swap(bpt_boruvka_active_edges &other) {
                m_data.swap(other.m_data);
                std::swap(m_capacity, other.m_capacity);
                std::swap(m_size, other.m_size);
            }
        };

        // Reused across rounds. Allocate/resize only outside parallel bodies.
        struct bpt_boruvka_scratch {
            static constexpr index_t block_size = 1024;
            xt::uvector<uint8_t> edge_class;
            std::vector<bpt_boruvka_counts> counts, offsets;
#ifdef HG_USE_TBB
            std::unique_ptr<std::atomic<index_t>[]> saddle_atomic;
#endif

            bpt_boruvka_scratch(index_t n, index_t m)
                : edge_class((std::max)(n, m)),
                  counts((std::max)(n, m) / block_size + ((std::max)(n, m) % block_size != 0)),
                  offsets(counts.size())
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
        template<typename active_t, typename classify_t, typename scatter_t>
        auto bpt_boruvka_compact(const active_t &active,
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
            assert(total.selected <= selected_capacity);
            (void)selected_capacity;
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

        // Frozen component saddles enumerate the selected set without an edge scan.
        // A sole chooser emits its edge; two choosers emit only at the smaller
        // component. Compare original edge IDs, including for parallel edges.
        // Prefix/scatter follows component order and gives each inverse-map entry
        // one writer. Forest bounds guarantee enough selected-slot capacity.
        template<typename E1, typename E2, typename emit_t>
        index_t bpt_boruvka_enumerate_saddles(const E1 &sources, const E2 &targets,
                                             const array_1d<index_t> &labels,
                                             const array_1d<index_t> &saddle,
                                             bpt_boruvka_scratch &scratch,
                                             index_t capacity, const emit_t &emit) {
            auto components = xt::arange<index_t>(saddle.size());
            auto counts = bpt_boruvka_compact(components, scratch, capacity,
                [&](index_t c) -> uint8_t {
                    const auto edge = saddle(c);
                    const auto a = labels(sources(edge)), b = labels(targets(edge));
                    assert(a != b && (c == a || c == b));
                    const auto d = c == a ? b : a;
                    return saddle(d) != edge || c < d ? 2 : 0;
                }, [&](index_t c, uint8_t, index_t destination) {
                    emit(saddle(c), destination);
                });
            return counts.selected;
        }

        enum bpt_boruvka_edge_state : uint8_t {
            bpt_boruvka_undecided = 0, bpt_boruvka_selected = 1, bpt_boruvka_out = 2
        };

        // Full scans read state before endpoints/keys. Each joined pass has one
        // writer per original edge, so states need no atomics.
        struct bpt_boruvka_full_scan {
            bpt_boruvka_scratch scratch;
            xt::uvector<uint8_t> states;
            bpt_boruvka_full_scan(index_t n, index_t m)
                : scratch(n, 0), states(m, bpt_boruvka_undecided) {}
            void selected(index_t edge) {
                assert(states[edge] == bpt_boruvka_undecided);
                states[edge] = bpt_boruvka_selected;
            }
            template<typename E1, typename E2, typename minimize_t>
            void scan(const E1 &sources, const E2 &targets,
                      const array_1d<index_t> &labels, const minimize_t &minimize) {
                parfor(0, sources.size(), [&](index_t edge) {
                    if (states[edge] != bpt_boruvka_undecided) return;
                    const auto a = labels(sources(edge)), b = labels(targets(edge));
                    if (a == b) {
                        states[edge] = bpt_boruvka_out;
                        return;
                    }
                    minimize(a, edge);
                    minimize(b, edge);
                });
            }
        };

        // Active IDs are written before their initialized prefix is published.
        // Next minima drops selected and newly internal edges in the same pass.
        struct bpt_boruvka_active_scan {
            bpt_boruvka_scratch scratch;
            bpt_boruvka_active_edges active, next;
            bpt_boruvka_active_scan(index_t n, index_t m)
                : scratch(n, m), active(m), next(m) {
                parfor(0, m, [&](index_t edge) { active[edge] = edge; });
            }
            void selected(index_t) {}
            template<typename E1, typename E2, typename minimize_t>
            void scan(const E1 &sources, const E2 &targets,
                      const array_1d<index_t> &labels, const minimize_t &minimize) {
                next.resize(active.size());
                auto counts = bpt_boruvka_compact(active, scratch, 0,
                    [&](index_t edge) -> uint8_t {
                        const auto a = labels(sources(edge)), b = labels(targets(edge));
                        if (a == b) return 0;
                        minimize(a, edge);
                        minimize(b, edge);
                        return 1;
                    }, [&](index_t edge, uint8_t, index_t destination) {
                        next[destination] = edge;
                    });
                next.resize(counts.retained);
                active.swap(next);
            }
        };

        template<typename E1, typename E2, typename traversal_t, typename edge_less_t>
        auto bpt_boruvka_minima(const E1 &sources, const E2 &targets,
                               const array_1d<index_t> &labels, index_t component_count,
                               traversal_t &traversal, const edge_less_t &less) {
#ifdef HG_USE_TBB
            auto &scratch = traversal.scratch;
#endif
            array_1d<index_t> saddle = xt::empty<index_t>({component_count});
            parfor(0, component_count, [&](index_t c) {
#ifdef HG_USE_TBB
                scratch.saddle_atomic[c].store(invalid_index, std::memory_order_relaxed);
#else
                saddle(c) = invalid_index;
#endif
            });
            // Only edge IDs are published. Immutable labels/keys and joined
            // initialization precede CAS; reduction joins before consumers.
            auto minimize = [&](index_t c, index_t edge) {
#ifdef HG_USE_TBB
                auto &candidate = scratch.saddle_atomic[c];
                auto current = candidate.load(std::memory_order_relaxed);
                while (current == invalid_index || less(edge, current)) {
                    if (candidate.compare_exchange_weak(current, edge, std::memory_order_relaxed)) break;
                }
#else
                if (saddle(c) == invalid_index || less(edge, saddle(c))) saddle(c) = edge;
#endif
            };
            traversal.scan(sources, targets, labels, minimize);
            auto missing = [&](index_t c) {
#ifdef HG_USE_TBB
                saddle(c) = scratch.saddle_atomic[c].load(std::memory_order_relaxed);
#endif
                return saddle(c) == invalid_index;
            };
            bool disconnected = false;
#ifdef HG_USE_TBB
            disconnected = tbb::parallel_reduce(tbb::blocked_range<index_t>(0, component_count), false,
                [&](const auto &range, bool any) {
                    for (auto c = range.begin(); c != range.end(); ++c) any = missing(c) || any;
                    return any;
                }, std::logical_or<bool>());
#else
            for (index_t c = 0; c < component_count; ++c) disconnected = missing(c) || disconnected;
#endif
            // Required input error, reported after the joined scan and before
            // any invalid saddle can reach successor or endpoint indexing.
            hg_assert(!disconnected, "Input graph must be connected.");
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
            const index_t count = saddle.size();

            array_1d<index_t> successor = xt::empty<index_t>({count});
            array_1d<index_t> jump_a = xt::empty<index_t>({count});
            array_1d<index_t> jump_b = xt::empty<index_t>({count});
            array_1d<index_t> root_rank = xt::empty<index_t>({count});
            array_1d<index_t> component_to_new = xt::empty<index_t>({count});
            parfor(0, count, [&](index_t c) {
                auto edge = saddle(c), a = labels(sources(edge)), b = labels(targets(edge));
                assert(a != b && (a == c || b == c));
                successor(c) = a == c ? b : a;
            });
            parfor(0, count, [&](index_t c) {
                auto next = successor(c);
                jump_a(c) = successor(next) == c && c < next ? c : next;
            });
            bool changed;
            do {
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
            assert(new_count <= count / 2);
            parfor(0, count, [&](index_t c) {
                component_to_new(c) = root_rank(jump_a(c));
            });

            return std::make_pair(std::move(component_to_new), new_count);
        }

        template<typename E1>
        void bpt_boruvka_record_history(const E1 &sources, const array_1d<index_t> &labels,
                                        const array_1d<index_t> &saddle, bpt_boruvka_history &history,
                                        index_t first, index_t last, index_t basin_offset,
                                        index_t component_count) {
            assert(basin_offset <= static_cast<index_t>(history.selected_edges.size()) - component_count);
            parfor(first, last, [&](index_t slot) {
                auto basin = labels(sources(history.selected_edges(slot)));
                history.raw_basin(slot) = basin_offset + basin;
                history.raw_parent(slot) = component_count == 1 ? invalid_index : saddle(basin);
            });
        }

        // Convert original-edge references only after all selected slots exist.
        // The inverse map is discarded before immutable expansion.
        inline void bpt_boruvka_history_to_slots(bpt_boruvka_history &history,
                                                const array_1d<index_t> &edge_to_slot) {
            auto convert = [&](index_t &edge) {
                if (edge != invalid_index) {
                    assert(edge_to_slot(edge) != invalid_index);
                    edge = edge_to_slot(edge);
                }
            };
            parfor(0, history.leaf_parent.size(), [&](index_t v) {
                assert(history.leaf_parent(v) != invalid_index);
                convert(history.leaf_parent(v));
            });
            parfor(0, history.raw_parent.size(), [&](index_t slot) { convert(history.raw_parent(slot)); });
        }

        // Preconditions: validated 1-D integral endpoints and n >= 1; less is
        // an immutable strict total order on original edge IDs for this call.
        // TBB invokes less concurrently; it must support simultaneous reads.
        template<typename traversal_t, typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_waterfall_impl(const E1 &sources, const E2 &targets,
                                  index_t n, const edge_less_t &less) {
            const index_t m = sources.size();
            bpt_boruvka_history history{xt::empty<index_t>({n - 1}),
                                        xt::empty<index_t>({n}),
                                        xt::empty<index_t>({n - 1}),
                                        xt::empty<index_t>({n - 1})};
            if (n == 1) {
                history.leaf_parent(0) = invalid_index;
                return history;
            }

            array_1d<index_t> edge_to_slot = bpt_boruvka_invalid_indices(m);
            array_1d<index_t> labels = array_1d<index_t>(xt::arange<index_t>(n));
            traversal_t traversal(n, m);
            auto &scratch = traversal.scratch;
            index_t component_count = n, selected_count = 0, basin_offset = 0;
            auto saddle = bpt_boruvka_minima(sources, targets, labels, component_count, traversal, less);
            parfor(0, n, [&](index_t v) { history.leaf_parent(v) = saddle(v); });
            while (component_count > 1) {
                const auto first_slot = selected_count;
                const auto emitted = bpt_boruvka_enumerate_saddles(sources, targets, labels, saddle,
                    scratch, n - 1 - first_slot, [&](index_t edge, index_t destination) {
                        const auto slot = first_slot + destination;
                        history.selected_edges(slot) = edge;
                        edge_to_slot(edge) = slot;
                        traversal.selected(edge);
                    });
                selected_count += emitted;

                auto labeling = bpt_boruvka_label_components(sources, targets, labels, saddle);
                const auto new_count = labeling.second;
                assert(new_count <= component_count / 2);
                assert(selected_count - first_slot == component_count - new_count);
                parfor(0, n, [&](index_t v) {
                    labels(v) = labeling.first(labels(v));
                });
                // Join before either policy scans the frozen new labels.
                if (new_count > 1) {
                    saddle = bpt_boruvka_minima(sources, targets, labels, new_count, traversal, less);
                }
                bpt_boruvka_record_history(sources, labels, saddle, history,
                                           first_slot, selected_count, basin_offset, new_count);
                basin_offset += new_count;
                component_count = new_count;
            }
            assert(selected_count == n - 1);
            bpt_boruvka_history_to_slots(history, edge_to_slot);
            return history;
        }

        template<typename E1, typename E2, typename edge_less_t>
        auto bpt_boruvka_waterfall(const E1 &sources, const E2 &targets,
                                  index_t n, const edge_less_t &less,
                                  bpt_boruvka_traversal traversal = bpt_boruvka_traversal::full_scan) {
            switch (traversal) {
                case bpt_boruvka_traversal::full_scan:
                    return bpt_boruvka_waterfall_impl<bpt_boruvka_full_scan>(sources, targets, n, less);
                case bpt_boruvka_traversal::active:
                    return bpt_boruvka_waterfall_impl<bpt_boruvka_active_scan>(sources, targets, n, less);
                default:
                    throw std::runtime_error("Unknown Boruvka traversal policy.");
            }
        }

        template<typename edge_less_t>
        auto bpt_boruvka_expand(const bpt_boruvka_history &history, const edge_less_t &less) {
            const index_t count = history.selected_edges.size();
            array_1d<index_t> parents = xt::empty<index_t>({count});
            array_1d<index_t> basins = xt::empty<index_t>({count});
            parfor(0, count, [&](index_t slot) {
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
            });
            return std::make_pair(std::move(parents), std::move(basins));
        }

        // Stable LSD radix grouping of slots by immutable expanded basin ID.
        // Each block owns a contiguous input interval. Bucket offsets enumerate
        // blocks in input order; each block scatters in input order. Thus equal
        // digits preserve order regardless of task scheduling. Induction over
        // low-to-high digit passes gives ascending basin order with stability.
        // At most 64 * 256 counters, independent of the number of basins/threads.
        // global_order is copied, never aliased or modified by buffer swaps.
        inline array_1d<index_t> bpt_boruvka_group_basins(
                const array_1d<index_t> &global_order, const array_1d<index_t> &basins) {
            const index_t count = global_order.size();
            array_1d<index_t> order = global_order;
            if (count < 2) return order;
            array_1d<index_t> scratch = xt::empty<index_t>({count});
            constexpr index_t buckets = 256;
            const index_t blocks = (std::min)(index_t(64), count / 16384 + (count % 16384 != 0));
            std::vector<index_t> offsets(size_t(blocks) * buckets);
            // Balanced ranges avoid count*block overflow. IDs are in [0,count).
            auto boundary = [&](index_t b) { return (count / blocks) * b + (std::min)(b, count % blocks); };
            for (unsigned shift = 0; (size_t(count - 1) >> shift) != 0; shift += 8) {
                {
                    parfor(0, blocks, [&](index_t block) {
                        auto row = offsets.data() + block * buckets;
                        std::fill(row, row + buckets, 0);
                        for (auto i = boundary(block); i < boundary(block + 1); ++i)
                            ++row[(size_t(basins(order(i))) >> shift) & 255];
                    });
                }
                {
                    index_t prefix = 0;
                    for (index_t bucket = 0; bucket < buckets; ++bucket) {
                        for (index_t block = 0; block < blocks; ++block) {
                            auto &offset = offsets[block * buckets + bucket];
                            auto size = offset;
                            offset = prefix;
                            prefix += size;
                        }
                    }
                }
                {
                    parfor(0, blocks, [&](index_t block) {
                        auto row = offsets.data() + block * buckets;
                        for (auto i = boundary(block); i < boundary(block + 1); ++i) {
                            auto slot = order(i);
                            scratch(row[(size_t(basins(slot)) >> shift) & 255]++) = slot;
                        }
                    });
                }
                std::swap(order, scratch);
                // Avoid a shift by the word width on the largest valid input.
                if (shift + 8 >= std::numeric_limits<size_t>::digits) break;
            }
            return order;
        }

        template<typename edge_less_t>
        auto bpt_boruvka_export(const bpt_boruvka_history &history, const edge_less_t &less,
                                bpt_boruvka_numbering numbering = bpt_boruvka_numbering::exact) {
            hg_assert(numbering == bpt_boruvka_numbering::exact || numbering == bpt_boruvka_numbering::basin,
                      "Unknown Boruvka numbering policy.");
            const index_t n = history.leaf_parent.size(), count = history.selected_edges.size();
            auto expanded = bpt_boruvka_expand(history, less);
            auto &edge_parent = expanded.first;
            const auto &basins = expanded.second;
            auto slot_less = [&](index_t a, index_t b) {
                return less(history.selected_edges(a), history.selected_edges(b));
            };
            array_1d<index_t> node_order, group_order;
            if (numbering == bpt_boruvka_numbering::exact) {
                node_order = xt::arange<index_t>(count);
                {
                    hg::sort(node_order.begin(), node_order.end(), slot_less);
                }
                // Stability preserves global K order inside every basin, so
                // this is exactly the old lexicographic (B,K) permutation.
                // Rewiring agrees; cross-basin parents and final IDs stay intact.
                group_order = bpt_boruvka_group_basins(node_order, basins);
            } else {
                group_order = xt::arange<index_t>(count);

                hg::sort(group_order.begin(), group_order.end(), [&](index_t a, index_t b) {
                    return basins(a) < basins(b) || (basins(a) == basins(b) && slot_less(a, b));
                });
            }
            {
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
            if (numbering == bpt_boruvka_numbering::basin) node_order = std::move(group_order);
            // Release grouping before allocating inverse maps and outputs.
            group_order = array_1d<index_t>();

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
                    assert(p != invalid_index || node == n + count - 1);
                    parents(node) = p == invalid_index ? node : slot_to_node(p);
                    assert(p == invalid_index || parents(node) > node);
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
                                        bpt_boruvka_numbering numbering = bpt_boruvka_numbering::exact,
                                        bpt_boruvka_traversal traversal = bpt_boruvka_traversal::full_scan) {
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
            auto history = bpt_boruvka_waterfall(sources, targets, n, less, traversal);
            return bpt_boruvka_export(history, less, numbering);
        }
    }
}
