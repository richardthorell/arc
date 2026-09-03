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
file(WRITE "${ARC_TEST_ROOT}/CookFixture.arcproject" [=[
{
  "format":"arc-project","formatVersion":2,
  "guid":"00000000-0000-4000-8000-00000000c001","name":"Cook Fixture","engineVersion":"0.1.0",
  "paths":{"source":"Source","content":"assets","config":"Config","plugins":"Plugins","saved":"Saved","intermediate":"Intermediate","build":"Build"},
  "assetRoots":["assets"],"modules":[],"plugins":[],"startupScenes":[],
  "targetPlatforms":[{"id":"windows-x64-vulkan","enabled":true}],
  "toolchain":{"compiler":"auto","minimumVersion":"","generator":"auto","architecture":"x86_64","cppStandard":20},
  "buildConfigurations":["Debug","RelWithDebInfo","Shipping"],
  "renderer":{"backend":"vulkan","api":"1.2","quality":"standard"},
  "cookProfiles":[{"id":"windows-x64-vulkan","platform":"windows","architecture":"x86_64","renderer":"vulkan","api":"1.2","textureFamily":"bc","configuration":"Shipping"}],
  "package":{"applicationName":"Cook Fixture","companyName":"","output":"Build/Packages","regionChunks":true},
  "settings":{"editor":"Config/Editor.json","renderer":"Config/Renderer.json","input":"Config/Input.json"}
}
]=])

set(output "${ARC_TEST_ROOT}/out")
set(manifest "${output}/windows-x64-vulkan.arccookmanifest")

message(STATUS "arc-cook-clean-checkout: starting cold cook (timeout: 180s)")
execute_process(
    COMMAND "${ARC_COOK}" cook --project "${ARC_TEST_ROOT}"
        --root assets/fixtures/persistence_fixture.arcscene
        --output "${output}" --json
    RESULT_VARIABLE first_result
    OUTPUT_VARIABLE first_output
    ERROR_VARIABLE first_error
    TIMEOUT 180
)
if(NOT first_result EQUAL 0)
    message(FATAL_ERROR "clean checkout cook failed (${first_result}):\n${first_output}\n${first_error}")
endif()
message(STATUS "arc-cook-clean-checkout: cold cook completed")

message(STATUS "arc-cook-clean-checkout: starting warm cache cook (timeout: 90s)")
execute_process(
    COMMAND "${ARC_COOK}" cook --project "${ARC_TEST_ROOT}"
        --root assets/fixtures/persistence_fixture.arcscene
        --output "${output}" --json
    RESULT_VARIABLE second_result
    OUTPUT_VARIABLE second_output
    ERROR_VARIABLE second_error
    TIMEOUT 90
)
if(NOT second_result EQUAL 0 OR NOT second_output MATCHES "\"cooked\":0")
    message(FATAL_ERROR "incremental cook was not a complete cache hit (${second_result}):\n${second_output}\n${second_error}")
endif()
message(STATUS "arc-cook-clean-checkout: warm cache cook completed")

message(STATUS "arc-cook-clean-checkout: starting package (timeout: 120s)")
execute_process(
    COMMAND "${ARC_COOK}" package --project "${ARC_TEST_ROOT}"
        --manifest "${manifest}" --output "${output}" --json
    RESULT_VARIABLE package_result
    OUTPUT_VARIABLE package_output
    ERROR_VARIABLE package_error
    TIMEOUT 120
)
if(NOT package_result EQUAL 0)
    message(FATAL_ERROR "clean checkout package failed (${package_result}):\n${package_output}\n${package_error}")
endif()
message(STATUS "arc-cook-clean-checkout: package completed")

message(STATUS "arc-cook-clean-checkout: starting verification (timeout: 60s)")
execute_process(
    COMMAND "${ARC_COOK}" verify --project "${ARC_TEST_ROOT}"
        --manifest "${manifest}" --output "${output}" --json
    RESULT_VARIABLE verify_result
    OUTPUT_VARIABLE verify_output
    ERROR_VARIABLE verify_error
    TIMEOUT 60
)
if(NOT verify_result EQUAL 0)
    message(FATAL_ERROR "clean checkout package verification failed (${verify_result}):\n${verify_output}\n${verify_error}")
endif()
message(STATUS "arc-cook-clean-checkout: verification completed")
