include("${CMAKE_CURRENT_LIST_DIR}/../../third_party/miniaudio.cmake")

arc_add_module(
    NAME arc-audio
    TYPE STATIC
    DEPS arc-math
)

target_link_libraries(arc-audio PRIVATE arc-third-party-miniaudio)

if(BUILD_TESTING)
    add_executable(arc-audio-foundation-tests
        test_support/foundation_tests.cpp
    )
    arc_configure_first_party_target(arc-audio-foundation-tests)
    target_compile_features(arc-audio-foundation-tests PRIVATE cxx_std_20)
    target_link_libraries(arc-audio-foundation-tests PRIVATE arc-audio)
    add_test(NAME arc-audio-foundation-tests COMMAND arc-audio-foundation-tests)
endif()
