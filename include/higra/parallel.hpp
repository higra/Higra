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

#include "utils.hpp"

#ifdef HG_USE_TBB
#include <memory>
#include <oneapi/tbb/global_control.h>
#endif

namespace hg {

#ifdef HG_USE_TBB
    namespace parallel_internal {
        inline std::unique_ptr<oneapi::tbb::global_control> &thread_limit() {
            static std::unique_ptr<oneapi::tbb::global_control> limit;
            return limit;
        }
    }
#endif

    /**
     * Set the maximum number of threads used by parallel computations.
     *
     * If ``num_threads`` is equal to 0, the limit set by a previous call is removed and the number of threads returns
     * to its default value (usually the number of logical cores available to the process).
     *
     * The limit applies to the whole process. It must not be changed while Higra functions are running on other
     * threads.
     *
     * Parallel computations are only available when Higra is compiled with Intel TBB (``HG_USE_TBB``). Otherwise,
     * computations always run on the calling thread and calling this function with a value different from 0 and 1
     * logs a warning.
     *
     * @param num_threads maximum number of threads (positive), or 0 to restore the default
     * @throws std::runtime_error if ``num_threads`` is negative
     */
    inline void set_num_threads(index_t num_threads) {
        hg_assert(num_threads >= 0, "The number of threads must be positive or 0.");
#ifdef HG_USE_TBB
        auto &limit = parallel_internal::thread_limit();
        limit.reset();
        if (num_threads > 0) {
            limit = std::make_unique<oneapi::tbb::global_control>(
                    oneapi::tbb::global_control::max_allowed_parallelism,
                    static_cast<std::size_t>(num_threads));
        }
#else
        if (num_threads > 1) {
            HG_LOG_WARNING("Warning: trying to set maximum number of threads but Higra was compiled without multi-threading!");
        }
#endif
    }

    /**
     * Maximum number of threads used by parallel computations: the value given to ``set_num_threads``,
     * or the default value if no limit is set.
     *
     * Without Intel TBB (``HG_USE_TBB``), the result is always 1.
     *
     * @return maximum number of threads
     */
    inline index_t get_num_threads() {
#ifdef HG_USE_TBB
        return static_cast<index_t>(oneapi::tbb::global_control::active_value(
                oneapi::tbb::global_control::max_allowed_parallelism));
#else
        return 1;
#endif
    }
}
