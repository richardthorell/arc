#pragma once

#include <cassert>
#include <type_traits>
#include <utility>
#include <variant>

namespace arc::core
{

/// @brief Value-or-error return type for recoverable operations.
/// @tparam T Successful value type.
/// @tparam E Error value type.
template <class T, class E>
class [[nodiscard]] result
{
public:
    /// @brief Construct a successful result.
    /// @param value Value returned by the operation.
    [[nodiscard]] static result success(T value)
    {
        return result(std::in_place_index<0>, std::move(value));
    }

    /// @brief Construct a failed result.
    /// @param error Error returned by the operation.
    [[nodiscard]] static result failure(E error)
    {
        return result(std::in_place_index<1>, std::move(error));
    }

    /// @brief Return whether the operation produced a value.
    [[nodiscard]] bool has_value() const noexcept
    {
        return storage_.index() == 0;
    }

    /// @brief Return whether the operation produced a value.
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return has_value();
    }

    /// @brief Access the successful value.
    /// @pre `has_value()` is true.
    [[nodiscard]] T& value() & noexcept
    {
        assert(has_value());
        return *std::get_if<0>(&storage_);
    }

    /// @brief Access the successful value.
    /// @pre `has_value()` is true.
    [[nodiscard]] const T& value() const& noexcept
    {
        assert(has_value());
        return *std::get_if<0>(&storage_);
    }

    /// @brief Move the successful value out of this result.
    /// @pre `has_value()` is true.
    [[nodiscard]] T&& value() && noexcept
    {
        assert(has_value());
        return std::move(*std::get_if<0>(&storage_));
    }

    /// @brief Access the failure value.
    /// @pre `has_value()` is false.
    [[nodiscard]] E& error() & noexcept
    {
        assert(!has_value());
        return *std::get_if<1>(&storage_);
    }

    /// @brief Access the failure value.
    /// @pre `has_value()` is false.
    [[nodiscard]] const E& error() const& noexcept
    {
        assert(!has_value());
        return *std::get_if<1>(&storage_);
    }

    /// @brief Move the failure value out of this result.
    /// @pre `has_value()` is false.
    [[nodiscard]] E&& error() && noexcept
    {
        assert(!has_value());
        return std::move(*std::get_if<1>(&storage_));
    }

private:
    template <std::size_t Index, class Value>
    explicit result(std::in_place_index_t<Index> index, Value&& value)
        : storage_(index, std::forward<Value>(value))
    {
    }

    std::variant<T, E> storage_;
};

/// @brief Value-or-error return type for operations without a success payload.
/// @tparam E Error value type.
template <class E>
class [[nodiscard]] result<void, E>
{
public:
    /// @brief Construct a successful status.
    [[nodiscard]] static result success()
    {
        return result(std::in_place_index<0>);
    }

    /// @brief Construct a failed status.
    /// @param error Error returned by the operation.
    [[nodiscard]] static result failure(E error)
    {
        return result(std::in_place_index<1>, std::move(error));
    }

    /// @brief Return whether the operation succeeded.
    [[nodiscard]] bool has_value() const noexcept
    {
        return storage_.index() == 0;
    }

    /// @brief Return whether the operation succeeded.
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return has_value();
    }

    /// @brief Access the failure value.
    /// @pre `has_value()` is false.
    [[nodiscard]] E& error() & noexcept
    {
        assert(!has_value());
        return *std::get_if<1>(&storage_);
    }

    /// @brief Access the failure value.
    /// @pre `has_value()` is false.
    [[nodiscard]] const E& error() const& noexcept
    {
        assert(!has_value());
        return *std::get_if<1>(&storage_);
    }

    /// @brief Move the failure value out of this status.
    /// @pre `has_value()` is false.
    [[nodiscard]] E&& error() && noexcept
    {
        assert(!has_value());
        return std::move(*std::get_if<1>(&storage_));
    }

private:
    explicit result(std::in_place_index_t<0> index)
        : storage_(index)
    {
    }

    template <class Value>
    explicit result(std::in_place_index_t<1> index, Value&& value)
        : storage_(index, std::forward<Value>(value))
    {
    }

    std::variant<std::monostate, E> storage_;
};

/// @brief Recoverable operation status without a success payload.
/// @tparam E Error value type.
template <class E>
using status = result<void, E>;

} // namespace arc::core
