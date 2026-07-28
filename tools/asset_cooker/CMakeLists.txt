add_executable(arc-cook
    main.cpp
)

target_compile_features(arc-cook PRIVATE cxx_std_20)
target_link_libraries(arc-cook
    PRIVATE
        arc-assets
        arc-render
        arc-third-party-json
)

set_target_properties(arc-cook PROPERTIES FOLDER "Tools")
