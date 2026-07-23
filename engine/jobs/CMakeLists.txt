arc_add_module(
    NAME arc-jobs
    TYPE STATIC
    DEPS arc-diagnostics arc-memory
    NO_TESTS
)

option(ARC_ENABLE_JOB_COROUTINES "Enable C++20 coroutine integration in arc-jobs" ON)
if(ARC_ENABLE_JOB_COROUTINES)
    target_compile_definitions(arc-jobs PUBLIC ARC_ENABLE_JOB_COROUTINES=1)
endif()

if(BUILD_TESTING AND ARC_BUILD_ENGINE_TESTS)
    add_subdirectory(tests)
endif()
