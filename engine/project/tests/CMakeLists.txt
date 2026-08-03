include(FetchContent)

if(NOT TARGET Catch2::Catch2WithMain)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.7.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

FetchContent_GetProperties(Catch2 SOURCE_DIR catch2_SOURCE_DIR)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")

add_executable(arc_project_tests project_tests.cpp)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(ARC_PROJECT_TEST_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ARC_PROJECT_TEST_REFLECTION_HEADER "${ARC_PROJECT_TEST_GENERATED_DIR}/arc_project_test.reflection.h")
set(ARC_PROJECT_TEST_REFLECTION_SCHEMA "${ARC_PROJECT_TEST_GENERATED_DIR}/arc_project_test.schema.json")
add_custom_command(
    OUTPUT "${ARC_PROJECT_TEST_REFLECTION_HEADER}" "${ARC_PROJECT_TEST_REFLECTION_SCHEMA}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/project_reflection.py"
        --header "${CMAKE_CURRENT_SOURCE_DIR}/reflection_fixture.h"
        --cpp-output "${ARC_PROJECT_TEST_REFLECTION_HEADER}"
        --json-output "${ARC_PROJECT_TEST_REFLECTION_SCHEMA}"
        --namespace arc::project::tests::generated
        --component-namespace arc::project::tests
    DEPENDS reflection_fixture.h "${CMAKE_SOURCE_DIR}/tools/project_reflection.py"
    VERBATIM)
target_sources(arc_project_tests PRIVATE "${ARC_PROJECT_TEST_REFLECTION_HEADER}" "${ARC_PROJECT_TEST_REFLECTION_SCHEMA}")
target_include_directories(arc_project_tests PRIVATE "${ARC_PROJECT_TEST_GENERATED_DIR}")
arc_configure_first_party_target(arc_project_tests)
target_link_libraries(arc_project_tests PRIVATE arc-project Catch2::Catch2WithMain)
target_compile_definitions(arc_project_tests PRIVATE ARC_TEST_TEMPLATE_ROOT="${CMAKE_SOURCE_DIR}/templates")

include(Catch)
catch_discover_tests(arc_project_tests)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/reflection-generator-tests")
add_test(NAME arc-project-reflection-generator-tests
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/reflection_generator_tests.py"
            "${CMAKE_SOURCE_DIR}/tools/project_reflection.py"
            "${CMAKE_CURRENT_BINARY_DIR}/reflection-generator-tests")
