#pragma once

#include "arc/simd/arch/riscv/detect.h"
#include "arc/simd/arch/riscv/registers.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace arc
{

#if defined(ARC_SIMD_RISCV)

template <class T, std::size_t N>
struct simd_op<riscv_vec<T, N>>
{
    using register_type = riscv_vec<T, N>;
    using mask_type = riscv_vec<uint32_t, N>;

    static inline register_type load_aligned(const T* ptr) noexcept
    {
        register_type result{};
        std::memcpy(result.lanes, ptr, sizeof(T) * N);
        return result;
    }

    static inline void store_aligned(T* ptr, register_type value) noexcept
    {
        std::memcpy(ptr, value.lanes, sizeof(T) * N);
    }

    static inline register_type load_unaligned(const T* ptr) noexcept
    {
        return load_aligned(ptr);
    }

    static inline void store_unaligned(T* ptr, register_type value) noexcept
    {
        store_aligned(ptr, value);
    }

    static inline register_type masked_load(const T* ptr, mask_type mask, register_type default_value) noexcept
    {
        register_type result{};
        for (std::size_t i = 0; i < N; ++i)
            result.lanes[i] = mask.lanes[i] != 0 ? ptr[i] : default_value.lanes[i];
        return result;
    }

    static inline void masked_store(T* ptr, register_type value, mask_type mask) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
            if (mask.lanes[i] != 0)
                ptr[i] = value.lanes[i];
    }

    static inline T extract(register_type value, std::size_t index) noexcept
    {
        return value.lanes[index];
    }

    template <std::size_t I>
    static inline T extract(register_type value) noexcept
    {
        static_assert(I < N, "lane index out of bounds");
        return value.lanes[I];
    }

    static inline register_type insert(register_type value, T element, std::size_t index) noexcept
    {
        value.lanes[index] = element;
        return value;
    }

    template <std::size_t I>
    static inline register_type insert(register_type value, T element) noexcept
    {
        static_assert(I < N, "lane index out of bounds");
        value.lanes[I] = element;
        return value;
    }

    static inline register_type fill(T value) noexcept
    {
        register_type result{};
        for (auto& lane : result.lanes)
            lane = value;
        return result;
    }

    static inline register_type add(register_type a, register_type b) noexcept
    {
        return map(a, b, [](T x, T y) { return static_cast<T>(x + y); });
    }

    static inline register_type sub(register_type a, register_type b) noexcept
    {
        return map(a, b, [](T x, T y) { return static_cast<T>(x - y); });
    }

    static inline register_type mul(register_type a, register_type b) noexcept
    {
        return map(a, b, [](T x, T y) { return static_cast<T>(x * y); });
    }

    static inline register_type div(register_type a, register_type b) noexcept
    {
        return map(a, b, [](T x, T y) { return static_cast<T>(x / y); });
    }

    static inline register_type neg(register_type a) noexcept
    {
        return map(a, [](T x) { return static_cast<T>(-x); });
    }

    static inline register_type min(register_type a, register_type b) noexcept
    {
        return map(a, b, [](T x, T y) { return x < y ? x : y; });
    }

    static inline register_type max(register_type a, register_type b) noexcept
    {
        return map(a, b, [](T x, T y) { return x > y ? x : y; });
    }

    static inline register_type bitwise_and(register_type a, register_type b) noexcept
    {
        return bitwise_map(a, b, [](auto x, auto y) { return x & y; });
    }

    static inline register_type bitwise_or(register_type a, register_type b) noexcept
    {
        return bitwise_map(a, b, [](auto x, auto y) { return x | y; });
    }

    static inline register_type bitwise_xor(register_type a, register_type b) noexcept
    {
        return bitwise_map(a, b, [](auto x, auto y) { return x ^ y; });
    }

    static inline register_type bitwise_not(register_type a) noexcept
    {
        return bitwise_map(a, [](auto x) { return ~x; });
    }

    static inline mask_type cmp_eq(register_type a, register_type b) noexcept
    {
        return compare(a, b, [](T x, T y) { return x == y; });
    }

    static inline mask_type cmp_ne(register_type a, register_type b) noexcept
    {
        return compare(a, b, [](T x, T y) { return x != y; });
    }

    static inline mask_type cmp_lt(register_type a, register_type b) noexcept
    {
        return compare(a, b, [](T x, T y) { return x < y; });
    }

    static inline mask_type cmp_le(register_type a, register_type b) noexcept
    {
        return compare(a, b, [](T x, T y) { return x <= y; });
    }

    static inline mask_type cmp_gt(register_type a, register_type b) noexcept
    {
        return compare(a, b, [](T x, T y) { return x > y; });
    }

    static inline mask_type cmp_ge(register_type a, register_type b) noexcept
    {
        return compare(a, b, [](T x, T y) { return x >= y; });
    }

    static inline register_type blend(register_type a, register_type b, mask_type mask) noexcept
    {
        register_type result{};
        for (std::size_t i = 0; i < N; ++i)
            result.lanes[i] = mask.lanes[i] != 0 ? b.lanes[i] : a.lanes[i];
        return result;
    }

    static inline bool any(mask_type mask) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
            if (mask.lanes[i] != 0)
                return true;
        return false;
    }

    static inline bool all(mask_type mask) noexcept
    {
        for (std::size_t i = 0; i < N; ++i)
            if (mask.lanes[i] == 0)
                return false;
        return true;
    }

    static inline register_type sqrt(register_type a) noexcept
    {
        return map_math(a, [](auto x) { return std::sqrt(x); });
    }

    static inline register_type rsqrt(register_type a) noexcept
    {
        return map_math(a, [](auto x) { return decltype(x){ 1 } / std::sqrt(x); });
    }

    static inline register_type reciprocal(register_type a) noexcept
    {
        return map(a, [](T x) { return static_cast<T>(T{ 1 } / x); });
    }

    static inline register_type fma(register_type a, register_type b, register_type c) noexcept
    {
        register_type result{};
        for (std::size_t i = 0; i < N; ++i)
        {
            if constexpr (std::is_floating_point_v<T>)
                result.lanes[i] = std::fma(a.lanes[i], b.lanes[i], c.lanes[i]);
            else
                result.lanes[i] = static_cast<T>(a.lanes[i] * b.lanes[i] + c.lanes[i]);
        }
        return result;
    }

    static inline register_type round(register_type a) noexcept
    {
        return map_math(a, [](auto x) { return std::round(x); });
    }

    static inline register_type floor(register_type a) noexcept
    {
        return map_math(a, [](auto x) { return std::floor(x); });
    }

    static inline register_type ceil(register_type a) noexcept
    {
        return map_math(a, [](auto x) { return std::ceil(x); });
    }

    static inline register_type trunc(register_type a) noexcept
    {
        return map_math(a, [](auto x) { return std::trunc(x); });
    }

    static inline T sum(register_type a) noexcept
    {
        T result{};
        for (std::size_t i = 0; i < N; ++i)
            result = static_cast<T>(result + a.lanes[i]);
        return result;
    }

    static inline T dot(register_type a, register_type b) noexcept
    {
        T result{};
        for (std::size_t i = 0; i < N; ++i)
            result = static_cast<T>(result + a.lanes[i] * b.lanes[i]);
        return result;
    }

    static inline T min_element(register_type a) noexcept
    {
        T result = a.lanes[0];
        for (std::size_t i = 1; i < N; ++i)
            result = a.lanes[i] < result ? a.lanes[i] : result;
        return result;
    }

    static inline T max_element(register_type a) noexcept
    {
        T result = a.lanes[0];
        for (std::size_t i = 1; i < N; ++i)
            result = a.lanes[i] > result ? a.lanes[i] : result;
        return result;
    }

private:
    template <class Op>
    static inline register_type map(register_type a, Op op) noexcept
    {
        register_type result{};
        for (std::size_t i = 0; i < N; ++i)
            result.lanes[i] = static_cast<T>(op(a.lanes[i]));
        return result;
    }

    template <class Op>
    static inline register_type map(register_type a, register_type b, Op op) noexcept
    {
        register_type result{};
        for (std::size_t i = 0; i < N; ++i)
            result.lanes[i] = static_cast<T>(op(a.lanes[i], b.lanes[i]));
        return result;
    }

    template <class Op>
    static inline register_type map_math(register_type a, Op op) noexcept
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            return map(a, op);
        }
        else
        {
            return a;
        }
    }

    template <class U>
    using bit_type = std::conditional_t<sizeof(U) == 8, uint64_t,
        std::conditional_t<sizeof(U) == 4, uint32_t,
        std::conditional_t<sizeof(U) == 2, uint16_t, uint8_t>>>;

    template <class U>
    static inline bit_type<U> to_bits(U value) noexcept
    {
        if constexpr (std::is_integral_v<U>)
        {
            return static_cast<bit_type<U>>(value);
        }
        else
        {
            bit_type<U> bits{};
            std::memcpy(&bits, &value, sizeof(U));
            return bits;
        }
    }

    static inline T from_bits(bit_type<T> bits) noexcept
    {
        if constexpr (std::is_integral_v<T>)
        {
            return static_cast<T>(bits);
        }
        else
        {
            T value{};
            std::memcpy(&value, &bits, sizeof(T));
            return value;
        }
    }

    template <class Op>
    static inline register_type bitwise_map(register_type a, Op op) noexcept
    {
        register_type result{};
        for (std::size_t i = 0; i < N; ++i)
            result.lanes[i] = from_bits(static_cast<bit_type<T>>(op(to_bits(a.lanes[i]))));
        return result;
    }

    template <class Op>
    static inline register_type bitwise_map(register_type a, register_type b, Op op) noexcept
    {
        register_type result{};
        for (std::size_t i = 0; i < N; ++i)
            result.lanes[i] = from_bits(static_cast<bit_type<T>>(op(to_bits(a.lanes[i]), to_bits(b.lanes[i]))));
        return result;
    }

    template <class Op>
    static inline mask_type compare(register_type a, register_type b, Op op) noexcept
    {
        mask_type result{};
        for (std::size_t i = 0; i < N; ++i)
            result.lanes[i] = op(a.lanes[i], b.lanes[i]) ? 0xFFFFFFFFu : 0u;
        return result;
    }
};

#endif

} // namespace arc
