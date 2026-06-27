#pragma once

#include <arc/simd/core/ops/detail.h>

namespace arc
{

template <class T, std::size_t N>
constexpr simd<T, N> select(const simd_mask<N>& mask, const simd<T, N>& a, const simd<T, N>& b) noexcept
{
    return [&]<std::size_t... Index>(std::index_sequence<Index...>)
    {
        return simd<T, N>{ ops_for<simd<T, N>>::blend(a.data[Index], b.data[Index], mask.data[Index])... };
    }(std::make_index_sequence<simd<T, N>::blocks()>{});
}

} // namespace arc
