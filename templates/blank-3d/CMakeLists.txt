cmake_minimum_required(VERSION 3.20)
project({{PROJECT_TOKEN}} LANGUAGES CXX)

# Android builds an independently packaged runtime module against the ARM64
# host shipped in the SDK. Do not import desktop ARC libraries into NDK.
if(ANDROID AND ARC_ANDROID_PROJECT_BUILD)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/AndroidRuntime.cmake")
  return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(arc_components Runtime Project Scene EditorModuleSDK)
if(WIN32)
  list(APPEND arc_components PlatformWindows Vulkan)
endif()
find_package(ARC {{ENGINE_VERSION}} EXACT CONFIG REQUIRED COMPONENTS ${arc_components})
unset(arc_components)

add_library({{PROJECT_TOKEN}}Runtime STATIC Source/{{PROJECT_TOKEN}}Runtime/Runtime.cpp)
target_compile_features({{PROJECT_TOKEN}}Runtime PUBLIC cxx_std_20)
target_link_libraries({{PROJECT_TOKEN}}Runtime PUBLIC ARC::Runtime ARC::Scene)
arc_generate_reflection(
  TARGET {{PROJECT_TOKEN}}Runtime
  NAMESPACE {{PROJECT_TOKEN}}::generated
  COMPONENT_NAMESPACE {{PROJECT_TOKEN}}
  HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/Source/{{PROJECT_TOKEN}}Runtime/Components.h")

add_library({{PROJECT_TOKEN}}Editor MODULE Source/{{PROJECT_TOKEN}}Editor/Editor.cpp)
target_compile_features({{PROJECT_TOKEN}}Editor PRIVATE cxx_std_20)
target_link_libraries({{PROJECT_TOKEN}}Editor PRIVATE {{PROJECT_TOKEN}}Runtime ARC::EditorModuleSDK)

if(WIN32)
  # Explicitly opt in only for an editor-launched Development game. Normal
  # packaged/Shipping builds keep ARC::PlatformWindowsPackagedEntry unchanged.
  option(ARC_ENABLE_LIVE_PLAY_DEV "Attach game runtime to ARC Editor in Development" OFF)
  set(arc_windows_entry ARC::PlatformWindowsPackagedEntry)
  if(ARC_ENABLE_LIVE_PLAY_DEV)
    # Fail closed: an accidentally enabled Development bridge must never
    # be linked into a Shipping executable, even on multi-config generators.
    if(CMAKE_BUILD_TYPE STREQUAL "Shipping" OR "Shipping" IN_LIST CMAKE_CONFIGURATION_TYPES)
      message(FATAL_ERROR "ARC_ENABLE_LIVE_PLAY_DEV cannot be used with Shipping configurations")
    endif()
    if(NOT TARGET ARC::PlatformWindowsLivePlayEntry OR NOT TARGET ARC::LivePlayDev)
      message(FATAL_ERROR "ARC SDK has no Development Live Play entrypoint. Rebuild SDK with ARC_BUILD_LIVE_PLAY_DEV=ON")
    endif()
    set(arc_windows_entry ARC::PlatformWindowsLivePlayEntry)
  endif()
  add_executable({{PROJECT_TOKEN}}Game WIN32 Source/{{PROJECT_TOKEN}}Runtime/Game.cpp)
  target_compile_definitions({{PROJECT_TOKEN}}Game PRIVATE
    ARC_GAME_BUILD_CONFIGURATION="$<CONFIG>")
  target_link_libraries({{PROJECT_TOKEN}}Game PRIVATE
    ${arc_windows_entry} {{PROJECT_TOKEN}}Runtime ARC::Project ARC::RenderVulkan)
  if(ARC_WORKFLOW_PROFILE)
    arc_register_runtime_output(TARGET {{PROJECT_TOKEN}}Game
      PROJECT_GUID "{{PROJECT_GUID}}" ENGINE_VERSION "{{ENGINE_VERSION}}"
      PROFILE "${ARC_WORKFLOW_PROFILE}" RENDERER vulkan ROLE client
      LIBRARIES ARC::RenderVulkan)
  endif()
  add_custom_command(TARGET {{PROJECT_TOKEN}}Game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:ARC::RenderVulkan>"
      "$<TARGET_FILE_DIR:{{PROJECT_TOKEN}}Game>"
    VERBATIM)
endif()

file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/arc-modules-$<CONFIG>.json" CONTENT
  "{\n  \"format\": \"arc-project-modules\",\n  \"formatVersion\": 1,\n  \"projectGuid\": \"{{PROJECT_GUID}}\",\n  \"engineVersion\": \"{{ENGINE_VERSION}}\",\n  \"configuration\": \"$<CONFIG>\",\n  \"modules\": [{\"id\":\"{{PROJECT_TOKEN}}.editor\",\"path\":\"$<TARGET_FILE:{{PROJECT_TOKEN}}Editor>\"}]\n}\n")
