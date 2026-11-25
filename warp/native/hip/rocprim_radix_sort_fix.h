// SPDX-License-Identifier: Apache-2.0

// ROCm build: rocPRIM's radix sort finishes small inputs with a merge sort whose comparator masks the keys to the
// sorted bit range [begin_bit, end_bit). It builds the mask as (T(1) << end_bit) - 1, a shift by the full key
// width when the range ends at the top bit, so a sort over [begin_bit, 32) or [begin_bit, 64) with begin_bit > 0
// compares the wrong bits and returns keys out of order (ROCm/rocm-libraries#16299). The specializations below
// replace that comparator for the integer key types, with the mask built without the full-width shift.
//
// Seen in rocPRIM 4.4 through 5.1; re-check the range when rocPRIM ships the fix.

#pragma once

#include <type_traits>

#include <rocprim/device/detail/device_radix_sort.hpp>
#include <rocprim/rocprim_version.hpp>

#if ROCPRIM_VERSION >= 400400 && ROCPRIM_VERSION < 500200

BEGIN_ROCPRIM_NAMESPACE
namespace detail {

template <bool Descending, class T> struct wp_radix_merge_compare_masked {
    using bits_type = typename std::make_unsigned<T>::type;
    static constexpr unsigned int key_bits = sizeof(T) * 8;

    T radix_mask;

    ROCPRIM_HOST_DEVICE
    wp_radix_merge_compare_masked(const unsigned int start_bit, const unsigned int current_radix_bits)
    {
        const unsigned int end_bit = start_bit + current_radix_bits;
        const bits_type upper
            = end_bit >= key_bits ? bits_type(~bits_type(0)) : bits_type((bits_type(1) << end_bit) - 1);
        const bits_type lower
            = start_bit >= key_bits ? bits_type(~bits_type(0)) : bits_type((bits_type(1) << start_bit) - 1);
        radix_mask = static_cast<T>(upper ^ lower);
    }

    ROCPRIM_DEVICE bool operator()(const T& a, const T& b) const
    {
        const T masked_key_a = a & radix_mask;
        const T masked_key_b = b & radix_mask;
        return Descending ? masked_key_a > masked_key_b : masked_key_b > masked_key_a;
    }
};

#define WP_ROCPRIM_RADIX_MERGE_COMPARE(DESCENDING, T) \
    template <> \
    struct radix_merge_compare<DESCENDING, true, T, identity_decomposer> \
        : wp_radix_merge_compare_masked<DESCENDING, T> { \
        ROCPRIM_HOST_DEVICE radix_merge_compare( \
            const unsigned int start_bit, const unsigned int current_radix_bits, identity_decomposer = {} \
        ) \
            : wp_radix_merge_compare_masked<DESCENDING, T>(start_bit, current_radix_bits) \
        { \
        } \
    };

#define WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH(T) \
    WP_ROCPRIM_RADIX_MERGE_COMPARE(false, T) \
    WP_ROCPRIM_RADIX_MERGE_COMPARE(true, T)

WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH(int)
WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH(unsigned int)
WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH(long)
WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH(unsigned long)
WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH(long long)
WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH(unsigned long long)

#undef WP_ROCPRIM_RADIX_MERGE_COMPARE_BOTH
#undef WP_ROCPRIM_RADIX_MERGE_COMPARE

}  // namespace detail
END_ROCPRIM_NAMESPACE

#endif
