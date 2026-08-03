#pragma once

/**
 * @file reflection.h
 * @brief Source annotations consumed by the ARC project reflection generator.
 *
 * The annotations intentionally expand to nothing for the C++ compiler. The
 * generator reads them before compilation and requires explicit stable IDs.
 */

/**
 * @brief Annotate the following struct as a reflected ARC component.
 * @param stable_id 32 hexadecimal digits retained across type renames.
 * @param schema_version Positive schema version.
 * @param display_name Inspector-facing name.
 * @param category Add Component menu category.
 * @param tooltip User-facing component description.
 */
#define ARC_COMPONENT(stable_id, schema_version, display_name, category, tooltip)

/**
 * @brief Annotate the following data member as a reflected property.
 * @param stable_id 16 hexadecimal digits retained across field renames.
 * @param display_name Inspector-facing name.
 * @param category Inspector grouping category.
 * @param tooltip User-facing field description.
 * @param kind One of bool, int, uint, float, string, enum, vector2, vector3, vector4, quaternion, entity, asset,
 * structure, or sequence.
 * @param default_json Canonical JSON representation of the default value.
 * @param minimum Minimum numeric value or an empty string.
 * @param maximum Maximum numeric value or an empty string.
 * @param flags Pipe-separated editable, readonly, transient, save_game, prefab, replicated, and serialized flags.
 * @param asset_type Stable asset-type restriction or an empty string.
 * @param entity_component Stable required component ID for entity references or an empty string.
 */
#define ARC_PROPERTY(stable_id, display_name, category, tooltip, kind, default_json, minimum, maximum, flags,          \
                     asset_type, entity_component)
