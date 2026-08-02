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
arc_configure_first_party_target(arc_project_tests)
target_link_libraries(arc_project_tests PRIVATE arc-project Catch2::Catch2WithMain)
target_compile_definitions(arc_project_tests PRIVATE ARC_TEST_TEMPLATE_ROOT="${CMAKE_SOURCE_DIR}/templates")

include(Catch)
catch_discover_tests(arc_project_tests)
