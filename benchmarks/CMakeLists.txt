add_executable(arc-benchmarks
    main.cpp
)
arc_configure_first_party_target(arc-benchmarks)
target_compile_features(arc-benchmarks PRIVATE cxx_std_20)
target_link_libraries(arc-benchmarks PRIVATE
    arc-assets
    arc-ecs
    arc-framework
    arc-io
    arc-jobs
    arc-memory
    arc-persistence
    arc-render
    arc-scene
)

add_custom_target(arc-benchmarks-check
    COMMAND arc-benchmarks
        --baseline "${CMAKE_CURRENT_SOURCE_DIR}/baselines.json"
        --output "${CMAKE_CURRENT_BINARY_DIR}/benchmark-results.json"
        --minimum-sample-ms 250
        --warmups 2
        --samples 7
        --threshold 0.20
    DEPENDS arc-benchmarks
    VERBATIM)
