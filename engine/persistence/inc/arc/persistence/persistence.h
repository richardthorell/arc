#pragma once

/** @namespace arc::persistence
 * @brief Reflected authoring archives, runtime archives, migrations, and recovery.
 */

#include <arc/assets/assets.h>
#include <arc/core/core.h>
#include <arc/ecs/reflection.h>
#include <arc/ecs/identity.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::persistence
{

enum class persistence_error_code : std::uint8_t
{
    invalid_argument,
    codec_missing,
    encode_failed,
    decode_failed,
    migration_invalid,
    migration_missing,
    integrity_failed,
    malformed_archive
};

struct persistence_error
{
    persistence_error_code code{persistence_error_code::invalid_argument};
    ecs::component_type_id component{};
    ecs::component_field_id field{};
    std::string message;
};

using persistence_status = core::status<persistence_error>;

enum class document_kind : std::uint8_t
{
    scene,
    prefab
};
enum class archive_value_kind : std::uint8_t
{
    null,
    boolean,
    signed_integer,
    unsigned_integer,
    floating_point,
    string,
    bytes,
    array,
    object
};

struct archive_object_member;

struct archive_value
{
    archive_value_kind kind{archive_value_kind::null};
    bool boolean{};
    std::int64_t signed_integer{};
    std::uint64_t unsigned_integer{};
    double floating_point{};
    std::string string;
    std::vector<std::byte> bytes;
    std::vector<archive_value> array;
    std::vector<archive_object_member> object;

    bool operator==(const archive_value&) const;
};

struct archive_object_member
{
    std::string name;
    archive_value value;

    archive_object_member() = default;
    archive_object_member(std::string member_name, archive_value member_value)
        : name(std::move(member_name)), value(std::move(member_value))
    {
    }

    friend bool operator==(const archive_object_member&, const archive_object_member&) = default;
};

inline bool archive_value::operator==(const archive_value&) const = default;

struct archive_field_record
{
    ecs::component_field_id id{};
    std::string name;
    archive_value value;
    bool known{true};

    friend bool operator==(const archive_field_record&, const archive_field_record&) = default;
};

struct archive_component_record
{
    ecs::component_type_id type{};
    std::string name;
    std::uint32_t schema_version{1};
    std::vector<archive_field_record> fields;
    bool known{true};

    friend bool operator==(const archive_component_record&, const archive_component_record&) = default;
};

struct archive_entity_record
{
    ecs::entity_guid id{};
    ecs::entity_guid parent{};
    std::uint32_t sibling_order{};
    ecs::entity_guid region{};
    std::vector<archive_component_record> components;
    archive_value extensions;

    friend bool operator==(const archive_entity_record&, const archive_entity_record&) = default;
};

struct dependency_manifest_entry
{
    assets::asset_reference reference;
    ecs::entity_guid owner_entity{};
    ecs::component_type_id owner_component{};
    ecs::component_field_id owner_field{};
    bool required{true};

    friend bool operator==(const dependency_manifest_entry&, const dependency_manifest_entry&) = default;
};

struct archive_document
{
    static constexpr std::uint32_t current_scene_version = 3;
    static constexpr std::uint32_t current_prefab_version = 2;

    document_kind kind{document_kind::scene};
    std::uint32_t format_version{current_scene_version};
    ecs::entity_guid id{};
    ecs::entity_guid root{};
    std::string name;
    std::vector<archive_entity_record> entities;
    std::vector<dependency_manifest_entry> dependencies;
    archive_value extensions;

    friend bool operator==(const archive_document&, const archive_document&) = default;
};

struct component_persistence_descriptor
{
    const ecs::component_descriptor* component{};
    std::vector<std::string> legacy_names;
    std::function<persistence_status(const void* component, archive_component_record& output)> encode;
    std::function<persistence_status(void* component, const archive_component_record& input)> decode;
};

class component_persistence_registry
{
public:
    bool register_component(component_persistence_descriptor descriptor);
    bool freeze();
    bool frozen() const noexcept;
    const component_persistence_descriptor* find(ecs::component_type_id type) const noexcept;
    const component_persistence_descriptor* find(std::string_view name) const noexcept;
    [[nodiscard]] persistence_status encode(ecs::component_type_id type, const void* component,
                                            archive_component_record& output) const;
    [[nodiscard]] persistence_status decode(ecs::component_type_id type, void* component,
                                            const archive_component_record& input) const;

private:
    std::vector<component_persistence_descriptor> descriptors_;
    bool frozen_{};
};

using component_migration = std::function<persistence_status(archive_component_record&)>;
using document_migration = std::function<persistence_status(archive_document&)>;

class schema_migration_registry
{
public:
    bool register_component(ecs::component_type_id type, std::uint32_t from_version, std::uint32_t to_version,
                            component_migration migration);
    bool register_document(document_kind kind, std::uint32_t from_version, std::uint32_t to_version,
                           document_migration migration);
    [[nodiscard]] persistence_status freeze();
    [[nodiscard]] persistence_status migrate(archive_component_record& component, std::uint32_t target_version) const;
    [[nodiscard]] persistence_status migrate(archive_document& document, std::uint32_t target_version) const;

private:
    struct component_edge
    {
        ecs::component_type_id type{};
        std::uint32_t from{};
        std::uint32_t to{};
        component_migration function;
    };
    struct document_edge
    {
        document_kind kind{};
        std::uint32_t from{};
        std::uint32_t to{};
        document_migration function;
    };
    std::vector<component_edge> component_edges_;
    std::vector<document_edge> document_edges_;
    bool frozen_{};
};

struct archive_limits
{
    std::size_t maximum_document_bytes{256u * 1024u * 1024u};
    std::size_t maximum_entities{1'000'000};
    std::size_t maximum_components_per_entity{1024};
    std::size_t maximum_fields_per_component{4096};
    std::size_t maximum_nesting{64};
};

struct archive_diagnostic
{
    std::string category;
    std::string message;
};

struct [[nodiscard]] archive_result
{
    archive_document document;
    std::string target_identity;
    std::vector<archive_diagnostic> diagnostics;
    std::string error;
    bool integrity_verified{};
    bool migrated{};
    bool succeeded() const noexcept
    {
        return error.empty();
    }
};

using json_archive_result = core::result<std::string, persistence_error>;
using binary_archive_result = core::result<std::vector<std::byte>, persistence_error>;

[[nodiscard]] json_archive_result write_reflected_json(const archive_document& document, bool pretty);
archive_result read_reflected_json(std::string_view text, const component_persistence_registry& components,
                                   const schema_migration_registry* migrations = nullptr, archive_limits limits = {});

[[nodiscard]] binary_archive_result write_tagged_binary(const archive_document& document,
                                                        std::string_view target_identity);
archive_result read_tagged_binary(std::span<const std::byte> bytes, const component_persistence_registry& components,
                                  const schema_migration_registry* migrations = nullptr, archive_limits limits = {});

struct [[nodiscard]] json_seal_result
{
    std::string text;
    assets::asset_hash payload_hash{};
    std::string error;
    bool succeeded() const noexcept
    {
        return error.empty();
    }
};

json_seal_result seal_json_document(std::string_view unsealed_text, bool pretty = true);
[[nodiscard]] core::result<assets::asset_hash, persistence_error> verify_json_document(std::string_view text,
                                                                                       bool require_integrity = true);

struct [[nodiscard]] document_save_result
{
    bool succeeded{};
    assets::asset_hash payload_hash{};
    std::string error;
};

struct [[nodiscard]] document_load_result
{
    bool succeeded{};
    bool recovered{};
    bool integrity_verified{};
    std::filesystem::path source_path;
    std::string text;
    std::vector<archive_diagnostic> diagnostics;
    std::string error;
};

class document_store
{
public:
    explicit document_store(std::size_t backup_generations = 3);
    document_save_result save_json(const std::filesystem::path& path, std::string_view unsealed_text,
                                   bool pretty = true) const;
    document_load_result load_json(const std::filesystem::path& path) const;

private:
    std::size_t backup_generations_{3};
};

std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept;

} // namespace arc::persistence
