find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(ARC_SCENE_SCHEMA "${CMAKE_CURRENT_SOURCE_DIR}/schema/components.arccomponents.json")
set(ARC_ECS_CODEGEN "${CMAKE_SOURCE_DIR}/tools/ecs_codegen.py")
set(ARC_SCENE_GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ARC_SCENE_GENERATED_METADATA
    "${ARC_SCENE_GENERATED_ROOT}/arc/scene/generated_component_metadata.h")
set(ARC_SCENE_GENERATED_TYPESCRIPT
    "${CMAKE_SOURCE_DIR}/editor/src/renderer/src/inspector/generatedEcsMetadata.ts")

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
    DEPS arc-ecs arc-render arc-math arc-framework arc-diagnostics arc-memory
    NO_TESTS
)

target_sources(arc-scene PRIVATE "${ARC_SCENE_GENERATED_METADATA}")
target_include_directories(arc-scene PUBLIC "${ARC_SCENE_GENERATED_ROOT}")

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
