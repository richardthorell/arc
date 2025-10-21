#pragma once

#include <arc/simd/storage.h>

namespace arc
{

template <class T, std::size_t N>
struct simd;

template <class T, std::size_t N>
struct simd_mask;

template <class T, std::size_t N>
struct simd
{
    using value_type = T;
    using storage_type = typename simd_storage<T, N>::storage_type;
    using op_type = simd_op<typename simd_storage<T, N>::register_type>;

    constexpr simd() noexcept = default;

    constexpr explicit simd(value_type value)
    {
        for (auto& block : data)
        {
            block = op_type::fill(value);
        }
    }

    static constexpr std::size_t size() noexcept
    {
        return N;
    }

private:
    template <class T, std::size_t N, std::size_t... Index, class Op>
    friend constexpr simd<T, N> apply(const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>, Op op) noexcept;

    template <class... Blocks>
    constexpr explicit simd(Blocks... blocks)
        : data{ blocks... }
    {
    }

    storage_type data;
};

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd<T, N> apply(std::index_sequence<Index...>, Op op) noexcept
{
    return simd<T, N>(op(Index)...);
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> apply(Op op) noexcept
{
    return apply<T, N>(std::make_index_sequence<simd_storage<T, N>::blocks>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd<T, N> apply(const simd<T, N>& a, std::index_sequence<Index...>, Op op) noexcept
{
    return simd<T, N>(op(a.data[Index])...);
}

template <class T, std::size_t N>
constexpr simd<T, N> apply(const simd<T, N>& a, Op op) noexcept
{
    return apply(a, std::make_index_sequence<simd_storage<T, N>::blocks>{}, op);
}

template <class T, std::size_t N, std::size_t... Index, class Op>
constexpr simd<T, N> apply(const simd<T, N>& a, const simd<T, N>& b, std::index_sequence<Index...>, Op op) noexcept
{
    return simd<T, N>(op(a.data[Index], b.data[Index])...);
}

template <class T, std::size_t N, class Op>
constexpr simd<T, N> apply(const simd<T, N>& a, const simd<T, N>& b, Op op) noexcept
{
    return apply(a, b, std::make_index_sequence<simd_storage<T, N>::blocks>{}, op);
}


// Load/Store operations
template <class T, std::size_t N>
constexpr simd<T, N> load(const T* ptr) noexcept
{
    return apply<T, N>(
        [&](std::size_t block_index)
        {
            return simd_op<typename simd_storage<T, N>::register_type>::load(ptr + block_index * simd_storage<T, N>::lanes);
        }
    );
}

template <class T, std::size_t N>
constexpr void store(T* ptr, const simd<T, N>& value) noexcept
{
    apply<T, N>(
        [&](std::size_t block_index)
        {
            simd_op<typename simd_storage<T, N>::register_type>::store(ptr + block_index * simd_storage<T, N>::lanes, value.data[block_index]);
        }
    );
}

template <class T, std::size_t N>
constexpr simd<T, N> fill(T value) noexcept
{
    return apply<T, N>(
        [&](std::size_t)
        {
            return simd_op<typename simd_storage<T, N>::register_type>::fill(value);
        }
    );
}


// Arithmetic operations
template <class T, std::size_t N>
constexpr simd<T, N> add(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::add);
}

template <class T, std::size_t N>
constexpr simd<T, N> sub(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::sub);
}

template <class T, std::size_t N>
constexpr simd<T, N> mul(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::mul);
}

template <class T, std::size_t N>
constexpr simd<T, N> div(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::div);
}

template <class T, std::size_t N>
constexpr simd<T, N> neg(const simd<T, N>& a) noexcept
{
    return apply(a, simd_op<typename simd_storage<T, N>::register_type>::neg);
}

// Min/Max operations
template <class T, std::size_t N>
constexpr simd<T, N> min(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::min);
}

template <class T, std::size_t N>
constexpr simd<T, N> max(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::max);
}

// Bitwise operations
template <class T, std::size_t N>
constexpr simd<T, N> bitwise_and(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::bitwise_and);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_or(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::bitwise_or);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_xor(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return apply(a, b, simd_op<typename simd_storage<T, N>::register_type>::bitwise_xor);
}

template <class T, std::size_t N>
constexpr simd<T, N> bitwise_not(const simd<T, N>& a) noexcept
{
    return apply(a, simd_op<typename simd_storage<T, N>::register_type>::bitwise_not);
}


// Operator overloads
template <class T, std::size_t N>
constexpr simd<T, N> operator+(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return add(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator-(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return sub(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator*(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return mul(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator/(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return div(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator-(const simd<T, N>& a) noexcept
{
    return neg(a);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator&(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_and(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator|(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_or(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator^(const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return bitwise_xor(a, b);
}

template <class T, std::size_t N>
constexpr simd<T, N> operator~(const simd<T, N>& a) noexcept
{
    return bitwise_not(a);
}

}