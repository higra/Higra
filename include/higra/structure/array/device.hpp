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

#include <cstdint>
#include <type_traits>

/**
 * Marks functions callable from both host and device code when compiled by
 * a CUDA or HIP compiler; expands to nothing otherwise.
 */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HG_HOST_DEVICE __host__ __device__
#else
#define HG_HOST_DEVICE
#endif

namespace hg {

    /**
     * Kind of memory holding array elements. Values are the DLPack ``DLDeviceType`` codes.
     */
    enum class device_type : int32_t {
        cpu = 1,
        cuda = 2,
        cuda_host = 3,
        rocm = 10,
        cuda_managed = 13
    };

    /**
     * Memory location of array elements: a device type and a device index (DLPack ``DLDevice``).
     */
    struct device {
        device_type type = device_type::cpu;
        int32_t id = 0;
    };

    HG_HOST_DEVICE constexpr bool operator==(const device &a, const device &b) noexcept {
        return a.type == b.type && a.id == b.id;
    }

    HG_HOST_DEVICE constexpr bool operator!=(const device &a, const device &b) noexcept {
        return !(a == b);
    }

    /**
     * The host CPU device.
     */
    inline constexpr device cpu_device{};

    /**
     * True if elements stored on the given device can be read and written directly by host code.
     */
    HG_HOST_DEVICE constexpr bool is_cpu_accessible(const device &d) noexcept {
        return d.type == device_type::cpu || d.type == device_type::cuda_host ||
               d.type == device_type::cuda_managed;
    }

    /**
     * Element type category. Values are the DLPack ``DLDataTypeCode`` codes.
     */
    enum class dtype_code : uint8_t {
        int_ = 0,
        uint = 1,
        float_ = 2,
        bool_ = 6
    };

    /**
     * Runtime description of an element type: category and size in bits (DLPack ``DLDataType``
     * with one lane).
     */
    struct dtype {
        dtype_code code;
        uint8_t bits;
    };

    HG_HOST_DEVICE constexpr bool operator==(const dtype &a, const dtype &b) noexcept {
        return a.code == b.code && a.bits == b.bits;
    }

    HG_HOST_DEVICE constexpr bool operator!=(const dtype &a, const dtype &b) noexcept {
        return !(a == b);
    }

    /**
     * Runtime dtype of the C++ type ``T``.
     *
     * The result depends only on the properties of ``T`` (``bool``, integral, signedness, floating
     * point, size), never on type identity: ``long``, ``long long``, ``size_t`` and ``char`` map to
     * the integer dtype of their size and signedness on the current platform.
     * ``bool`` maps to ``{bool_, 8}``, as in DLPack and NumPy.
     */
    template<class T>
    constexpr dtype dtype_of() noexcept {
        using U = std::remove_cv_t<T>;
        static_assert(std::is_arithmetic_v<U>, "dtype_of: T must be bool, an integral type or a floating point type.");
        if constexpr (std::is_same_v<U, bool>) {
            return {dtype_code::bool_, 8};
        } else if constexpr (std::is_integral_v<U>) {
            return {std::is_signed_v<U> ? dtype_code::int_ : dtype_code::uint, static_cast<uint8_t>(sizeof(U) * 8)};
        } else {
            return {dtype_code::float_, static_cast<uint8_t>(sizeof(U) * 8)};
        }
    }
}
