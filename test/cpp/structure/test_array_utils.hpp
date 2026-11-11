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
// This header does not include xtensor.

#include "higra/structure/array/all.hpp"

#include <cstddef>
#include <sstream>
#include <string>

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_tostring.hpp"

namespace Catch {

    template<class T>
    struct StringMaker<hg::array<T>> {
        static std::string convert(const hg::array<T> &a) {
            std::ostringstream s;
            s << a;
            return s.str();
        }
    };

    template<class T>
    struct StringMaker<hg::array_view<T>> {
        static std::string convert(const hg::array_view<T> &a) {
            std::ostringstream s;
            s << hg::array_view<const std::remove_const_t<T>>(a);
            return s.str();
        }
    };

    template<>
    struct StringMaker<hg::array_shape> {
        static std::string convert(const hg::array_shape &s) {
            std::string r = "(";
            for (hg::index_t k = 0; k < s.size(); ++k) {
                r += (k > 0 ? ", " : "") + std::to_string(s[k]);
            }
            return r + ")";
        }
    };
}

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
