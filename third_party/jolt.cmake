# Jolt Physics is a private implementation detail of arc-physics. Keep this
# target out of ARC public interfaces so Jolt types cannot leak into gameplay-
# facing APIs.
include(FetchContent)

set(ARC_PINNED_JOLT_VERSION "v5.6.0" CACHE STRING
    "Exact Jolt Physics release used by ARC")

if(NOT TARGET arc-third-party-jolt)
    add_library(arc-third-party-jolt INTERFACE)
    add_library(arc::third-party-jolt ALIAS arc-third-party-jolt)
endif()

if(ARC_FETCH_THIRD_PARTY)
    set(JPH_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(JPH_USE_DX12 OFF CACHE BOOL "" FORCE)
    set(JPH_USE_VK OFF CACHE BOOL "" FORCE)
    set(JPH_USE_MTL OFF CACHE BOOL "" FORCE)
    set(JPH_USE_CPU_COMPUTE OFF CACHE BOOL "" FORCE)
    set(ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        jolt
        GIT_REPOSITORY https://github.com/jrouwe/JoltPhysics.git
        GIT_TAG ${ARC_PINNED_JOLT_VERSION}
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR Build
    )
    FetchContent_MakeAvailable(jolt)

    if(NOT TARGET Jolt)
        message(FATAL_ERROR "Pinned Jolt ${ARC_PINNED_JOLT_VERSION} did not provide the expected Jolt target")
    endif()

    if(MSVC)
        target_compile_options(Jolt PRIVATE /W0)
    else()
        target_compile_options(Jolt PRIVATE -w)
    endif()

    # Keep the fetched Jolt target build-tree-only. The installed ARC SDK
    # exports the private wrapper so arc-physics' static-library dependency graph
    # remains valid without exposing Jolt as an SDK dependency before the backend
    # itself is shipped.
    target_link_libraries(arc-third-party-jolt INTERFACE "$<BUILD_INTERFACE:Jolt>")
else()
    message(FATAL_ERROR "arc-physics requires Jolt and ARC_FETCH_THIRD_PARTY is OFF")
endif()

set_target_properties(arc-third-party-jolt PROPERTIES EXPORT_NAME _Jolt)
install(TARGETS arc-third-party-jolt
    EXPORT ARCTargets
    COMPONENT sdk-private)
