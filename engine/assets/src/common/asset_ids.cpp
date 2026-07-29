#include <arc/assets/assets.h>

#include <array>
#include <bit>
#include <charconv>
#include <fstream>
#include <random>

namespace arc::assets
{
namespace
{

constexpr char hex_digits[] = "0123456789abcdef";

constexpr int from_hex(char value) noexcept
{
    if (value >= '0' && value <= '9')
        return value - '0';
    if (value >= 'a' && value <= 'f')
        return value - 'a' + 10;
    if (value >= 'A' && value <= 'F')
        return value - 'A' + 10;
    return -1;
}

constexpr std::array<std::uint32_t, 64> sha256_constants{{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
}};

class sha256
{
public:
    void update(std::span<const std::byte> bytes) noexcept
    {
        for (std::byte value : bytes)
        {
            buffer_[buffer_size_++] = std::to_integer<std::uint8_t>(value);
            if (buffer_size_ == buffer_.size())
            {
                transform();
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    asset_hash finish() noexcept
    {
        bit_count_ += static_cast<std::uint64_t>(buffer_size_) * 8;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56)
        {
            while (buffer_size_ < 64)
                buffer_[buffer_size_++] = 0;
            transform();
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56)
            buffer_[buffer_size_++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8)
            buffer_[buffer_size_++] = static_cast<std::uint8_t>(bit_count_ >> shift);
        transform();

        asset_hash result;
        for (std::size_t word = 0; word < state_.size(); ++word)
            for (std::size_t byte = 0; byte < 4; ++byte)
                result.bytes[word * 4 + byte] =
                    static_cast<std::byte>(state_[word] >> ((3 - byte) * 8));
        return result;
    }

private:
    void transform() noexcept
    {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index)
        {
            words[index] =
                (static_cast<std::uint32_t>(buffer_[index * 4]) << 24) |
                (static_cast<std::uint32_t>(buffer_[index * 4 + 1]) << 16) |
                (static_cast<std::uint32_t>(buffer_[index * 4 + 2]) << 8) |
                static_cast<std::uint32_t>(buffer_[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index)
        {
            const auto s0 = std::rotr(words[index - 15], 7) ^
                std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const auto s1 = std::rotr(words[index - 2], 17) ^
                std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        auto a = state_[0];
        auto b = state_[1];
        auto c = state_[2];
        auto d = state_[3];
        auto e = state_[4];
        auto f = state_[5];
        auto g = state_[6];
        auto h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index)
        {
            const auto sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto temporary1 = h + sum1 + choice + sha256_constants[index] + words[index];
            const auto sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    }};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_{};
    std::uint64_t bit_count_{};
};

}

std::size_t asset_guid_hash::operator()(asset_guid value) const noexcept
{
    return core::uuid_hash<asset_guid_tag>{}(value);
}

std::size_t asset_type_id_hash::operator()(asset_type_id value) const noexcept
{
    return core::uuid_hash<asset_type_id_tag>{}(value);
}

std::size_t asset_importer_id_hash::operator()(asset_importer_id value) const noexcept
{
    return core::uuid_hash<asset_importer_id_tag>{}(value);
}

std::string to_string(asset_guid value)
{
    return core::to_string(value, core::uuid_text_format::hyphenated);
}

std::string to_string(asset_type_id value)
{
    return core::to_string(value, core::uuid_text_format::hyphenated);
}

std::string to_string(asset_importer_id value)
{
    return core::to_string(value, core::uuid_text_format::hyphenated);
}

std::optional<asset_guid> parse_asset_guid(std::string_view text) noexcept
{
    return core::parse_uuid<asset_guid_tag>(text);
}

std::optional<asset_type_id> parse_asset_type_id(std::string_view text) noexcept
{
    return core::parse_uuid<asset_type_id_tag>(text);
}

std::optional<asset_importer_id> parse_asset_importer_id(std::string_view text) noexcept
{
    return core::parse_uuid<asset_importer_id_tag>(text);
}

asset_guid generate_asset_guid() noexcept
{
    thread_local std::mt19937_64 generator([] {
        std::random_device random;
        std::seed_seq seed{ random(), random(), random(), random(),
            static_cast<unsigned>(reinterpret_cast<std::uintptr_t>(&random)) };
        return std::mt19937_64(seed);
    }());
    asset_guid result{ generator(), generator() };
    result.high = (result.high & 0xffffffffffff0fffull) | 0x0000000000004000ull;
    result.low = (result.low & 0x3fffffffffffffffull) | 0x8000000000000000ull;
    if (!result.valid())
        result.low = 1;
    return result;
}

std::string to_string(const asset_hash& value)
{
    std::string result;
    result.reserve(value.bytes.size() * 2);
    for (std::byte byte : value.bytes)
    {
        const unsigned number = std::to_integer<unsigned>(byte);
        result.push_back(hex_digits[number >> 4]);
        result.push_back(hex_digits[number & 0x0f]);
    }
    return result;
}

std::optional<asset_hash> parse_asset_hash(std::string_view text) noexcept
{
    if (text.size() != 64)
        return std::nullopt;
    asset_hash result;
    for (std::size_t index = 0; index < result.bytes.size(); ++index)
    {
        const int high = from_hex(text[index * 2]);
        const int low = from_hex(text[index * 2 + 1]);
        if (high < 0 || low < 0)
            return std::nullopt;
        result.bytes[index] = static_cast<std::byte>((high << 4) | low);
    }
    return result;
}

asset_hash hash_bytes(std::span<const std::byte> bytes) noexcept
{
    sha256 hash;
    hash.update(bytes);
    return hash.finish();
}

asset_hash_result hash_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return asset_hash_result::failure({
            .code = asset_error_code::io_failed,
            .path = path,
            .message = "Could not open source file for hashing"
        });
    sha256 hash;
    std::array<std::byte, 64u * 1024u> buffer{};
    while (input)
    {
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto count = input.gcount();
        if (count > 0)
            hash.update(std::span<const std::byte>(buffer.data(), static_cast<std::size_t>(count)));
    }
    if (!input.eof())
        return asset_hash_result::failure({
            .code = asset_error_code::io_failed,
            .path = path,
            .message = "Source file hashing failed"
        });
    return asset_hash_result::success(hash.finish());
}

asset_hash combine_hashes(std::span<const asset_hash> hashes) noexcept
{
    sha256 hash;
    for (const asset_hash& value : hashes)
        hash.update(value.bytes);
    return hash.finish();
}

} // namespace arc::assets
