#pragma once

#include <type_traits>

namespace arc
{

template <class T>
struct simd_traits
{
};

template <>
struct simd_traits<float>
{
    using scalar_type = float;

    // TODO: Add more later
};

}