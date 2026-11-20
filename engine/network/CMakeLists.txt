arc_add_module(
    NAME arc-network
    TYPE STATIC
)

if(BUILD_TESTING)
    add_executable(arc-network-foundation-tests
        test_support/foundation_tests.cpp
    )
    arc_configure_first_party_target(arc-network-foundation-tests)
    target_compile_features(arc-network-foundation-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-network-foundation-tests PRIVATE arc-network)
    add_test(NAME arc-network-foundation-tests COMMAND arc-network-foundation-tests)
endif()

# Live Play is explicitly opt-in Development tooling, never a dependency of
# arc-network / arc-runtime / packaged Shipping games.
option(ARC_BUILD_LIVE_PLAY_DEV "Build the optional native Live Play Development bridge" OFF)
if(ARC_BUILD_LIVE_PLAY_DEV OR (BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS))
    add_library(arc-live-play-dev STATIC src/dev/live_play_dev_client.cpp)
    arc_configure_first_party_target(arc-live-play-dev)
    target_compile_features(arc-live-play-dev PUBLIC cxx_std_20)
    target_include_directories(arc-live-play-dev PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/dev/include>")
    target_link_libraries(arc-live-play-dev PRIVATE arc-third-party-json)
    add_library(arc::live-play-dev ALIAS arc-live-play-dev)
    if(WIN32)
        target_link_libraries(arc-live-play-dev PUBLIC ws2_32)
    endif()
    if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
        find_package(Threads REQUIRED)
        add_executable(arc-live-play-dev-tests test_support/live_play_dev_tests.cpp)
        arc_configure_first_party_target(arc-live-play-dev-tests)
        target_compile_features(arc-live-play-dev-tests PRIVATE cxx_std_20)
        target_link_libraries(arc-live-play-dev-tests PRIVATE
            arc-live-play-dev arc-third-party-json Threads::Threads)
        target_compile_definitions(arc-live-play-dev-tests PRIVATE
            ARC_LIVE_PLAY_WIRE_VECTOR_PATH="${CMAKE_CURRENT_SOURCE_DIR}/test_support/live_play_wire_v1.json")
        add_test(NAME arc-live-play-dev-tests COMMAND arc-live-play-dev-tests)
    endif()
endif()
