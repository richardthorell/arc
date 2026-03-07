#pragma once

#include <arc/simd/core/simd.h>

namespace arc
{

/**
 * @brief SIMD mask type.
 * 
 * @tparam N The total number of lanes in the SIMD mask.
 */
template <std::size_t N>
struct simd_mask
{
    /**
     * @brief The base data type.
     */
    using value_type = uint32_t;

    /**
     * @brief SIMD type for the mask.
     */
    using simd_type = simd<value_type, N>;

    /**
     * @brief SIMD operation type.
     */
    using op_type = simd_op<typename simd_block<value_type, N>::register_type>;

    /**
     * @brief Construct a SIMD mask from a SIMD vector.
     * 
     * @param value The SIMD vector to convert to a mask.
     */
    constexpr explicit simd_mask(const simd_type& value)
        : data{value.data}
    {
    }

    /**
     * @brief Construct a SIMD mask with all lanes set to the given value.
     * 
     * @param value The value to set in all lanes (true or false).
     */
    constexpr explicit simd_mask(bool value) noexcept
        : simd_mask(std::make_index_sequence<blocks()>{}, [lanes = value ? 0xFFFFFFFF : 0](std::size_t) { return op_type::fill(lanes); })
    {

    }

    /**
     * @brief Get the total number of lanes in the SIMD mask.
     * 
     * @return The number of lanes.
     */
    static constexpr std::size_t size() noexcept
    {
        return N;
    }

    /**
     * @brief Get the number of SIMD register blocks in the SIMD mask.
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
    using block_type = simd_block<value_type, N>;

    /**
     * @brief Data storage type.
     */
    using data_type = typename block_type::data_type;

    /**
     * @brief SIMD register type.
     */
    using register_type = typename block_type::register_type;

    template <std::size_t M, std::size_t... Index, class Op>
    friend constexpr auto apply(std::index_sequence<Index...>, Op) noexcept;

    template <std::size_t M, std::size_t... Index, class Op>
    friend constexpr auto apply(const simd_mask<M>&, std::index_sequence<Index...>, Op) noexcept;

    template <std::size_t M, std::size_t... Index, class Op>
    friend constexpr auto apply(const simd_mask<M>&, const simd_mask<M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd_mask<M> compare(const simd<U, M>&, const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd<U, M> masked(const simd_mask<M>&, const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <class U, std::size_t M, std::size_t... Index, class Op>
    friend constexpr simd<U, M> masked(const simd_mask<M>&, const simd<U, M>&, const simd<U, M>&, std::index_sequence<Index...>, Op) noexcept;

    template <std::size_t M>
    friend constexpr bool any(const simd_mask<M>&) noexcept;

    template <std::size_t M>
    friend constexpr bool all(const simd_mask<M>&) noexcept;

    template <std::size_t... Index, class Op>
    constexpr explicit simd_mask(std::index_sequence<Index...>, Op op) noexcept
        : data{ op(Index)... }
    {
    }

    template <class... Blocks>
    constexpr explicit simd_mask(Blocks... blocks) noexcept
        : data{ blocks... }
    {
    }

    data_type data;
};

} // namespace arc
