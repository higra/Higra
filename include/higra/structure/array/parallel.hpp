/***************************************************************************
* Copyright ESIEE Paris (2026)                                             *
*                                                                          *
* Contributor(s) : Benjamin Perret                                         *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#pragma once

#include <utility>
#include "../../utils.hpp"

#ifdef HG_USE_TBB

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

#endif

/**
 * Minimum number of elements for which built-in array loops run in parallel (only when
 * ``HG_USE_TBB`` is defined). Must be defined identically in every translation unit.
 * Fills (``fill``, ``full``, ``zeros``, ``ones``, ``*_like``) and ``arange`` are always sequential.
 */
#ifndef HG_ARRAY_PARALLEL_THRESHOLD
#define HG_ARRAY_PARALLEL_THRESHOLD (1 << 16)
#endif

/**
 * Number of elements of the fixed chunks used by array reductions. Results of floating point
 * sums depend on this value and on nothing else (neither the build nor the thread count).
 */
#ifndef HG_ARRAY_REDUCE_CHUNK
#define HG_ARRAY_REDUCE_CHUNK (1 << 12)
#endif

namespace hg {

    /**
     * Execution policies of ``hg::map`` and ``hg::map_into``.
     *
     * - ``exec::seq`` (default): the function is called sequentially, on the calling thread.
     * - ``exec::par``: the function may be called concurrently from several threads (with
     *   ``HG_USE_TBB``, above ``HG_ARRAY_PARALLEL_THRESHOLD`` elements); the caller guarantees
     *   that this is safe.
     */
    namespace exec {
        struct seq_t {
        };
        struct par_t {
        };
        inline constexpr seq_t seq{};
        inline constexpr par_t par{};
    }

    namespace detail {

        /**
         * True if a built-in loop over ``n`` elements runs in parallel.
         */
        inline constexpr bool use_parallel(index_t n) noexcept {
#ifdef HG_USE_TBB
            return n >= static_cast<index_t>(HG_ARRAY_PARALLEL_THRESHOLD);
#else
            (void) n;
            return false;
#endif
        }

        /**
         * Calls ``body(begin, end)`` on blocks covering ``[0, n)``: concurrently on blocks of at least
         * ``grain`` items (``tbb::parallel_for``, static partitioner) if ``parallel`` is true and
         * ``HG_USE_TBB`` is defined, otherwise once as ``body(0, n)`` on the calling thread (not at
         * all if ``n == 0``).
         */
        template<class Body>
        void parallel_blocks_if(bool parallel, index_t n, index_t grain, Body &&body) {
#ifdef HG_USE_TBB
            if (parallel && n > 1) {
                tbb::parallel_for(tbb::blocked_range<index_t>(0, n, grain > 0 ? grain : 1),
                                  [&body](const tbb::blocked_range<index_t> &r) {
                                      body(r.begin(), r.end());
                                  },
                                  tbb::static_partitioner());
                return;
            }
#else
            (void) parallel;
            (void) grain;
#endif
            if (n > 0) {
                body(index_t(0), n);
            }
        }

        /**
         * Calls ``body(begin, end)`` on blocks covering ``[0, n)``.
         *
         * With ``HG_USE_TBB`` and ``n >= HG_ARRAY_PARALLEL_THRESHOLD``, blocks of at least ``grain``
         * elements are processed concurrently (``tbb::parallel_for``, static partitioner);
         * otherwise ``body(0, n)`` is called once on the calling thread (not at all if ``n == 0``).
         */
        template<class Body>
        void parallel_blocks(index_t n, index_t grain, Body &&body) {
            parallel_blocks_if(use_parallel(n), n, grain, std::forward<Body>(body));
        }

        /**
         * Calls ``body(c, begin, end)`` for each chunk ``c`` of ``HG_ARRAY_REDUCE_CHUNK`` consecutive
         * elements of ``[0, n)`` (the last chunk may be shorter). Chunks run concurrently under
         * the same conditions as ``parallel_blocks``. Returns the number of chunks.
         */
        template<class Body>
        index_t for_each_chunk(index_t n, Body &&body) {
            constexpr index_t chunk = HG_ARRAY_REDUCE_CHUNK;
            const index_t num_chunks = (n + chunk - 1) / chunk;
            auto run = [&body, n](index_t cb, index_t ce) {
                for (index_t c = cb; c < ce; ++c) {
                    body(c, c * chunk, (c + 1) * chunk < n ? (c + 1) * chunk : n);
                }
            };
#ifdef HG_USE_TBB
            if (use_parallel(n)) {
                tbb::parallel_for(tbb::blocked_range<index_t>(0, num_chunks, 1),
                                  [&run](const tbb::blocked_range<index_t> &r) {
                                      run(r.begin(), r.end());
                                  },
                                  tbb::static_partitioner());
                return num_chunks;
            }
#endif
            run(0, num_chunks);
            return num_chunks;
        }
    }
}
