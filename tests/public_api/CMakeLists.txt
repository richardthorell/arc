find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(GLOB_RECURSE ARC_PUBLIC_HEADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/engine/*/inc/arc/*.h"
    "${CMAKE_SOURCE_DIR}/engine/*/inc/arc/**/*.h")

set(ARC_PUBLIC_HEADER_SOURCES)
foreach(header IN LISTS ARC_PUBLIC_HEADER_FILES)
    if(header MATCHES "/simd/arch/" OR
       header MATCHES "/detail/" OR
       header MATCHES "/simd/core/ops/detail\\.h$")
        continue()
    endif()
    if(header MATCHES "/render-vulkan/" AND NOT TARGET arc-render-vulkan)
        continue()
    endif()
    string(REGEX REPLACE "^.*/inc/" "" include_path "${header}")
    string(MAKE_C_IDENTIFIER "${include_path}" source_name)
    set(source_path "${CMAKE_CURRENT_BINARY_DIR}/${source_name}.cpp")
    file(WRITE "${source_path}" "#include <${include_path}>\n")
    list(APPEND ARC_PUBLIC_HEADER_SOURCES "${source_path}")
endforeach()

set(ARC_PUBLIC_UMBRELLA_HEADERS
    arc/scene/scene.h
    arc/render/render.h
    arc/project/project.h
    arc/persistence/persistence.h
    arc/ecs/ecs.h
    arc/assets/assets.h
    arc/framework/framework.h
    arc/io/io.h
    arc/input/input.h
    arc/memory/memory.h
    arc/jobs/jobs.h
    arc/diagnostics/diagnostics.h
    arc/geometric/geometric.h
    arc/math/math.h
    arc/simd/simd.h
    arc/core/core.h)
if(TARGET arc-render-tools)
    list(PREPEND ARC_PUBLIC_UMBRELLA_HEADERS arc/render_tools/render_tools.h)
endif()
if(TARGET arc-render-vulkan)
    list(PREPEND ARC_PUBLIC_UMBRELLA_HEADERS arc/render/vulkan/vulkan_backend.h)
endif()

set(umbrella_forward_path "${CMAKE_CURRENT_BINARY_DIR}/arc_public_umbrellas_forward.cpp")
set(umbrella_reverse_path "${CMAKE_CURRENT_BINARY_DIR}/arc_public_umbrellas_reverse.cpp")
file(WRITE "${umbrella_forward_path}" "")
file(WRITE "${umbrella_reverse_path}" "")
foreach(header IN LISTS ARC_PUBLIC_UMBRELLA_HEADERS)
    file(APPEND "${umbrella_forward_path}" "#include <${header}>\n")
endforeach()
list(REVERSE ARC_PUBLIC_UMBRELLA_HEADERS)
foreach(header IN LISTS ARC_PUBLIC_UMBRELLA_HEADERS)
    file(APPEND "${umbrella_reverse_path}" "#include <${header}>\n")
endforeach()
list(APPEND ARC_PUBLIC_HEADER_SOURCES
    "${umbrella_forward_path}"
    "${umbrella_reverse_path}")

add_library(arc-public-header-tests OBJECT ${ARC_PUBLIC_HEADER_SOURCES})
arc_configure_first_party_target(arc-public-header-tests)
target_compile_features(arc-public-header-tests PRIVATE cxx_std_20)
foreach(target
    arc-core arc-simd arc-math arc-geometric arc-diagnostics arc-jobs arc-memory
    arc-io arc-framework arc-assets arc-input arc-ecs arc-persistence arc-project arc-render
    arc-render-tools arc-render-vulkan arc-scene)
    if(TARGET ${target})
        target_link_libraries(arc-public-header-tests PRIVATE ${target})
    endif()
endforeach()
if(MSVC)
    target_compile_options(arc-public-header-tests PRIVATE /W4)
else()
    target_compile_options(arc-public-header-tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(
    NAME arc-public-api-policy
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/check_public_api.py")
