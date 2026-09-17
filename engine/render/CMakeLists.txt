arc_add_module(
    NAME arc-render
    TYPE STATIC
    DEPS arc-framework arc-math arc-water arc-jobs arc-memory arc-io arc-diagnostics arc-third-party-meshoptimizer
    NO_TESTS
)

if(NOT TARGET arc-third-party-tinyexr)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tinyexr"
        "${CMAKE_BINARY_DIR}/third_party/tinyexr"
    )
endif()

if(NOT TARGET arc-third-party-tiff)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/tiff"
        "${CMAKE_BINARY_DIR}/third_party/tiff"
    )
endif()

target_sources(arc-render PRIVATE src/common/lighting_scene.cpp)

if(TARGET arc-third-party-ufbx)
    target_link_libraries(arc-render PRIVATE arc-third-party-ufbx)
    if(ARC_THIRD_PARTY_UFBX_AVAILABLE)
        target_compile_definitions(arc-render PRIVATE ARC_RENDER_HAS_UFBX=1)
    endif()
endif()

if(TARGET arc-third-party-stb)
    target_link_libraries(arc-render PRIVATE "$<BUILD_INTERFACE:arc-third-party-stb>")
endif()

if(TARGET arc-third-party-tinyexr)
    target_link_libraries(arc-render PRIVATE "$<BUILD_INTERFACE:arc-third-party-tinyexr>")
endif()

if(TARGET arc-third-party-tiff)
    target_link_libraries(arc-render PRIVATE "$<BUILD_INTERFACE:arc-third-party-tiff>")
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
