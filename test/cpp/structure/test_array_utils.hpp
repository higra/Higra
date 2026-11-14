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

// Helpers for the tests of the array library (higra/structure/array/).

#include "higra/structure/array/all.hpp"

#include <cstddef>
#include <sstream>
#include <string>

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_tostring.hpp"
#include "../test_utils.hpp"

// Catch::StringMaker specializations for arrays, views and shapes are in test_utils.hpp.

namespace array_test_utils {

    /**
     * CPU allocator counting its calls and live bytes, to test allocation behavior.
     */
    struct counting_allocator final : hg::allocator {
        std::size_t allocations = 0;
        std::size_t deallocations = 0;
        std::size_t live_bytes = 0;

        void *allocate(std::size_t bytes) override {
            ++allocations;
            live_bytes += bytes;
            return hg::cpu_allocator().allocate(bytes);
        }

        void deallocate(void *p, std::size_t bytes) noexcept override {
            ++deallocations;
            live_bytes -= bytes;
            hg::cpu_allocator().deallocate(p, bytes);
        }

        hg::device device() const noexcept override {
            return hg::cpu_device;
        }
    };
}
