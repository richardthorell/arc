arc_add_module(
    NAME arc-flow
    TYPE STATIC
)

target_link_libraries(arc-flow PRIVATE "$<BUILD_INTERFACE:arc-third-party-json>")
