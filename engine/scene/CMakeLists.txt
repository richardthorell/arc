find_package(Python3 COMPONENTS Interpreter REQUIRED)
# This module may be built by the standalone Android NDK CMake host, whose
# CMAKE_SOURCE_DIR is its app/src/main/cpp folder, not the ARC repository.
get_filename_component(ARC_SCENE_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

set(ARC_SCENE_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/schema/components.arccomponents.json")
set(ARC_ECS_CODEGEN "${ARC_SCENE_REPO_ROOT}/tools/ecs_codegen.py")
set(ARC_SCENE_GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ARC_SCENE_GENERATED_METADATA
    "${ARC_SCENE_GENERATED_ROOT}/arc/scene/generated_component_metadata.h")
set(ARC_SCENE_GENERATED_TYPESCRIPT
    "${ARC_SCENE_REPO_ROOT}/editor/src/renderer/src/inspector/generatedEcsMetadata.ts")

add_custom_command(
    OUTPUT "${ARC_SCENE_GENERATED_METADATA}"
    COMMAND "${Python3_EXECUTABLE}" "${ARC_ECS_CODEGEN}"
        --schema "${ARC_SCENE_SCHEMA}"
        --cpp-output "${ARC_SCENE_GENERATED_METADATA}"
    DEPENDS "${ARC_ECS_CODEGEN}" "${ARC_SCENE_SCHEMA}"
    COMMENT "Generating ARC scene component metadata"
    VERBATIM
)

arc_add_module(
    NAME arc-scene
    TYPE STATIC
    DEPS arc-assets arc-ecs arc-render arc-water arc-math arc-framework arc-diagnostics arc-memory arc-persistence
    NO_TESTS
)

target_link_libraries(arc-scene PRIVATE "$<BUILD_INTERFACE:arc-third-party-json>")
target_sources(arc-scene PRIVATE "${ARC_SCENE_GENERATED_METADATA}")
target_include_directories(arc-scene PUBLIC
    "$<BUILD_INTERFACE:${ARC_SCENE_GENERATED_ROOT}>"
    "$<INSTALL_INTERFACE:include>")
install(FILES "${ARC_SCENE_GENERATED_METADATA}"
    DESTINATION include/arc/scene
    COMPONENT sdk-headers)

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
    add_test(
        NAME arc-scene-codegen-check
        COMMAND "${Python3_EXECUTABLE}" "${ARC_ECS_CODEGEN}"
            --schema "${ARC_SCENE_SCHEMA}"
            --cpp-output "${ARC_SCENE_GENERATED_METADATA}"
            --ts-output "${ARC_SCENE_GENERATED_TYPESCRIPT}"
            --check
    )
endif()
