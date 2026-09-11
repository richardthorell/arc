arc_add_module(
    NAME arc-flow
    TYPE STATIC
    TEST_DEPS arc-project
)

target_link_libraries(arc-flow
    PRIVATE
        "$<BUILD_INTERFACE:arc-project>"
        "$<BUILD_INTERFACE:arc-third-party-json>"
)
