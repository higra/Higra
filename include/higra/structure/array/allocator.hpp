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

#include <cstddef>
#include <new>
#include "device.hpp"

namespace hg {

    /**
     * Storage provider of owning arrays.
     *
     * Allocators are process-lifetime singletons: arrays keep a plain pointer to the allocator
     * that created their buffer and use it to release the buffer.
     */
    struct allocator {
        /**
         * Allocate ``bytes`` bytes; throws ``std::bad_alloc`` on failure.
         */
        virtual void *allocate(std::size_t bytes) = 0;

        /**
         * Release a buffer returned by ``allocate(bytes)``.
         */
        virtual void deallocate(void *p, std::size_t bytes) noexcept = 0;

        /**
         * Device of the buffers returned by this allocator.
         */
        virtual hg::device device() const noexcept = 0;

        virtual ~allocator() = default;
    };

    namespace array_internal {

        struct cpu_allocator_impl final : allocator {
            static constexpr std::size_t alignment = 64;

            void *allocate(std::size_t bytes) override {
                if (bytes == 0) {
                    return nullptr;
                }
                return ::operator new(bytes, std::align_val_t{alignment});
            }

            void deallocate(void *p, std::size_t /*bytes*/) noexcept override {
                if (p != nullptr) {
                    ::operator delete(p, std::align_val_t{alignment});
                }
            }

            hg::device device() const noexcept override {
                return cpu_device;
            }
        };
    }

    /**
     * Allocator of host memory, aligned on 64 bytes.
     */
    inline allocator &cpu_allocator() {
        static array_internal::cpu_allocator_impl instance;
        return instance;
    }
}
