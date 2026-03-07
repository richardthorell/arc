#pragma once

#include <arc/simd/core/block.h>

#include <ranges>

namespace arc
{

/**
 * @brief Helper alias to get the operation type for a given SIMD or mask type.
 */
template <typename SimdOrMask>
using ops_for = typename SimdOrMask::op_type;


template <std::size_t N>
struct simd_mask;


/**
 * @brief SIMD vector type.
 * 
 * @tparam T The base data type (e.g., float, int32_t).
 * @tparam N The total number of lanes in the SIMD vector.
 */
template <class T, std::size_t N>
struct simd
{
    /**
     * @brief The base data type.
     */
    using value_type = T;

    /**
     * @brief Mask type for the SIMD vector.
     */
    using mask_type = simd_mask<N>;

    /**
     * @brief SIMD operation type.
     */
    using op_type = simd_op<register_type>;

    /**
     * @brief Default constructor.
     */
    constexpr simd() noexcept = default;

    /**
     * @brief Construct a SIMD vector with all lanes set to the given value.
     * 
     * @param value The value to set in all lanes.
     */
    constexpr explicit simd(value_type value) noexcept
        : simd(std::make_index_sequence<blocks()>{}, [&](std::size_t) { return op_type::fill(value); })
    {
    }

    /**
     * @brief Get the total number of lanes in the SIMD vector.
     * 
     * @return The number of lanes.
     */
    static constexpr std::size_t size() noexcept
    {
        return N;
    }


    /**
     * @brief Get the number of SIMD register blocks in the SIMD vector.
     * 
     * @return The number of blocks.
     */
    static constexpr std::size_t blocks() noexcept
    {
        return block_type::blocks;
    }

private:
    /**
     * @brief Storage block type.
     */
    using block_type = simd_block<T, N>;

    /**
     * @brief Data storage type.
     */
    using data_type = typename block_type::data_type;

    /**
     * @brief SIMD register type.
     */
    using register_type = typename block_type::register_type;

    template <std::size_t M>
    friend struct simd_mask;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd<U, M> apply(std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd<U, M> apply(const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd<U, M> apply(const simd<U, M>&, const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd_mask<M> compare(const simd<U, M>&, const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd<U, M> masked(const simd_mask<M>&, const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd<U, M> masked(const simd_mask<M>&, const simd<U, M>&, const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M>
    friend constexpr simd<U, M> load(const U*) noexcept;

    template <class U, std::size_t M>
    friend constexpr void store(U*, const simd<U, M>&) noexcept;

    template <std::size_t... Index, class Op>
    constexpr explicit simd(std::index_sequence<Index...>, Op op) noexcept
        : data{ op(Index)... }
    {
    }

    template <class... Blocks>
    constexpr explicit simd(Blocks... blocks) noexcept
        : data{ blocks... }
    {
    }

    data_type data;
};


template <typename T, std::size_t N, std::input_iterator Iter, std::sentinel_for<Iter> Sent>
    requires std::convertible_to<std::iter_value_t<Iter>, T>
inline simd<T, N> make_simd(Iter first, Sent last)
{
    using block = simd_block<T, N>;
    using ops   = ops_for<simd<T, N>>;

    simd<T, N> result;

    const std::size_t range_size  = static_cast<std::size_t>(std::ranges::distance(first, last));
    const std::size_t full_blocks = range_size / block::lanes;
    const std::size_t remainder   = range_size % block::lanes;

    // Full contiguous blocks
    if constexpr (std::contiguous_iterator<Iter>)
    {
        for (std::size_t b = 0; b < full_blocks; ++b)
        {
            result.data[b] = ops::load(std::to_address(first) + b * block::lanes);
        }
    }
    else
    {
        Iter it = first;
        for (std::size_t b = 0; b < full_blocks; ++b)
        {
            std::array<T, block::lanes> tmp{};

            for (std::size_t l = 0; l < block::lanes && it != last; ++l, ++it)
                tmp[l] = *it;

            result.data[b] = ops::load(tmp.data());
        }
    }

    // Partial block (remainder)
    if (remainder > 0)
    {
        std::array<T, block::lanes> tmp{};
        for (std::size_t i = 0; i < remainder && first != last; ++i, ++first)
            tmp[i] = *first;

        result.data[full_blocks] = ops::load(tmp.data());
    }

    return result;
}

} // namespace arc
