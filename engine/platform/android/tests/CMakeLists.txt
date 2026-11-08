cmake_minimum_required(VERSION 3.20)
project(arc_android_host_lifecycle_tests LANGUAGES CXX)

enable_testing()
add_executable(arc_android_lifecycle_tests
    lifecycle_tests.cpp
    ../app/src/main/cpp/android_lifecycle.cpp)
target_compile_features(arc_android_lifecycle_tests PRIVATE cxx_std_20)
add_test(NAME arc_android_lifecycle_tests COMMAND arc_android_lifecycle_tests)
