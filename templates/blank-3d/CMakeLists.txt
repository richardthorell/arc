cmake_minimum_required(VERSION 3.20)
project({{PROJECT_TOKEN}} LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(ARC {{ENGINE_VERSION}} EXACT CONFIG REQUIRED COMPONENTS Runtime EditorModuleSDK Vulkan)

add_library({{PROJECT_TOKEN}}Runtime STATIC Source/{{PROJECT_TOKEN}}Runtime/Runtime.cpp)
target_compile_features({{PROJECT_TOKEN}}Runtime PUBLIC cxx_std_20)
target_link_libraries({{PROJECT_TOKEN}}Runtime PUBLIC ARC::Runtime)
arc_generate_reflection(
  TARGET {{PROJECT_TOKEN}}Runtime
  NAMESPACE {{PROJECT_TOKEN}}::generated
  COMPONENT_NAMESPACE {{PROJECT_TOKEN}}
  HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/Source/{{PROJECT_TOKEN}}Runtime/Components.h")

add_library({{PROJECT_TOKEN}}Editor MODULE Source/{{PROJECT_TOKEN}}Editor/Editor.cpp)
target_compile_features({{PROJECT_TOKEN}}Editor PRIVATE cxx_std_20)
target_link_libraries({{PROJECT_TOKEN}}Editor PRIVATE {{PROJECT_TOKEN}}Runtime ARC::EditorModuleSDK)

if(WIN32)
  add_executable({{PROJECT_TOKEN}}Game WIN32 Source/{{PROJECT_TOKEN}}Runtime/Game.cpp)
  target_link_libraries({{PROJECT_TOKEN}}Game PRIVATE {{PROJECT_TOKEN}}Runtime ARC::PlatformWindows ARC::RenderVulkan)
  add_custom_command(TARGET {{PROJECT_TOKEN}}Game POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:ARC::RenderVulkan>"
      "$<TARGET_FILE_DIR:{{PROJECT_TOKEN}}Game>"
    VERBATIM)
endif()

file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/arc-modules-$<CONFIG>.json" CONTENT
  "{\n  \"format\": \"arc-project-modules\",\n  \"formatVersion\": 1,\n  \"projectGuid\": \"{{PROJECT_GUID}}\",\n  \"engineVersion\": \"{{ENGINE_VERSION}}\",\n  \"configuration\": \"$<CONFIG>\",\n  \"modules\": [{\"id\":\"{{PROJECT_TOKEN}}.editor\",\"path\":\"$<TARGET_FILE:{{PROJECT_TOKEN}}Editor>\"}]\n}\n")
