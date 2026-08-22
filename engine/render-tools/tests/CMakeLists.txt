file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(arc_render_tools_tests ${TEST_SOURCES})
arc_configure_first_party_target(arc_render_tools_tests)
target_link_libraries(arc_render_tools_tests PRIVATE arc-render-tools Catch2::Catch2WithMain)

add_test(NAME arc_render_tools_tests COMMAND arc_render_tools_tests)
