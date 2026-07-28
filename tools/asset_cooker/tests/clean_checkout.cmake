if(NOT DEFINED ARC_COOK OR NOT DEFINED ARC_SOURCE_ROOT OR NOT DEFINED ARC_TEST_ROOT)
    message(FATAL_ERROR "clean checkout cook test is missing required paths")
endif()

file(REMOVE_RECURSE "${ARC_TEST_ROOT}")
file(MAKE_DIRECTORY
    "${ARC_TEST_ROOT}/assets/fixtures"
    "${ARC_TEST_ROOT}/assets/models"
    "${ARC_TEST_ROOT}/assets/materials"
    "${ARC_TEST_ROOT}/assets/textures/terrain/aerial_grass_rock"
    "${ARC_TEST_ROOT}/assets/shaders/include"
    "${ARC_TEST_ROOT}/assets/environments"
)

file(COPY
    "${ARC_SOURCE_ROOT}/assets/fixtures/persistence_fixture.arcscene"
    "${ARC_SOURCE_ROOT}/assets/fixtures/persistence_fixture.arcscene.arcmeta"
    "${ARC_SOURCE_ROOT}/assets/fixtures/persistence_fixture.arcprefab"
    "${ARC_SOURCE_ROOT}/assets/fixtures/persistence_fixture.arcprefab.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/fixtures"
)
file(COPY
    "${ARC_SOURCE_ROOT}/assets/models/UAL2_Standard.glb"
    "${ARC_SOURCE_ROOT}/assets/models/UAL2_Standard.glb.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/models"
)
file(COPY
    "${ARC_SOURCE_ROOT}/assets/materials/default_phong.arcmat"
    "${ARC_SOURCE_ROOT}/assets/materials/default_phong.arcmat.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/materials"
)
file(COPY
    "${ARC_SOURCE_ROOT}/assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg"
    "${ARC_SOURCE_ROOT}/assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/textures/terrain/aerial_grass_rock"
)
file(COPY
    "${ARC_SOURCE_ROOT}/assets/shaders/default_phong.frag"
    "${ARC_SOURCE_ROOT}/assets/shaders/default_phong.frag.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/shaders"
)
file(COPY
    "${ARC_SOURCE_ROOT}/assets/shaders/include/"
    DESTINATION "${ARC_TEST_ROOT}/assets/shaders/include"
    FILES_MATCHING PATTERN "*.glsl" PATTERN "*.arcmeta"
)
file(COPY
    "${ARC_SOURCE_ROOT}/assets/environments/autumn_field_puresky_1k.hdr"
    "${ARC_SOURCE_ROOT}/assets/environments/autumn_field_puresky_1k.hdr.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/environments"
)
file(COPY "${ARC_SOURCE_ROOT}/arc.cook.json" DESTINATION "${ARC_TEST_ROOT}")

set(output "${ARC_TEST_ROOT}/out")
set(manifest "${output}/windows-x64-vulkan.arccookmanifest")

execute_process(
    COMMAND "${ARC_COOK}" cook --project "${ARC_TEST_ROOT}"
        --root assets/fixtures/persistence_fixture.arcscene
        --output "${output}" --json
    RESULT_VARIABLE first_result
    OUTPUT_VARIABLE first_output
    ERROR_VARIABLE first_error
)
if(NOT first_result EQUAL 0)
    message(FATAL_ERROR "clean checkout cook failed:\n${first_output}\n${first_error}")
endif()

execute_process(
    COMMAND "${ARC_COOK}" cook --project "${ARC_TEST_ROOT}"
        --root assets/fixtures/persistence_fixture.arcscene
        --output "${output}" --json
    RESULT_VARIABLE second_result
    OUTPUT_VARIABLE second_output
    ERROR_VARIABLE second_error
)
if(NOT second_result EQUAL 0 OR NOT second_output MATCHES "\"cooked\":0")
    message(FATAL_ERROR "incremental cook was not a complete cache hit:\n${second_output}\n${second_error}")
endif()

execute_process(
    COMMAND "${ARC_COOK}" package --project "${ARC_TEST_ROOT}"
        --manifest "${manifest}" --output "${output}" --json
    RESULT_VARIABLE package_result
    OUTPUT_VARIABLE package_output
    ERROR_VARIABLE package_error
)
if(NOT package_result EQUAL 0)
    message(FATAL_ERROR "clean checkout package failed:\n${package_output}\n${package_error}")
endif()

execute_process(
    COMMAND "${ARC_COOK}" verify --project "${ARC_TEST_ROOT}"
        --manifest "${manifest}" --output "${output}" --json
    RESULT_VARIABLE verify_result
    OUTPUT_VARIABLE verify_output
    ERROR_VARIABLE verify_error
)
if(NOT verify_result EQUAL 0)
    message(FATAL_ERROR "clean checkout package verification failed:\n${verify_output}\n${verify_error}")
endif()
