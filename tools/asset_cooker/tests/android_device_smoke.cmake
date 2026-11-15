# Android V3.4f.1: isolated target-cooked sample project.
# Copy bytes rather than symlink: the cook must not depend on a source checkout
# once the assembled project has been relocated.
foreach(required ARC_COOK ARC_DEVICE_SMOKE_VERIFY ARC_SOURCE_ROOT ARC_TEST_ROOT)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Android device smoke: missing ${required}")
    endif()
endforeach()
if(NOT DEFINED ARC_DEVICE_SMOKE_TIMEOUT)
    set(ARC_DEVICE_SMOKE_TIMEOUT 360)
endif()

set(fixture "${ARC_SOURCE_ROOT}/tests/fixtures/android-device-smoke")
file(REMOVE_RECURSE "${ARC_TEST_ROOT}")
file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/Content/Scenes")
file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/Content/models")
file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/Content/materials")
file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/Content/textures/terrain/aerial_grass_rock")
file(COPY "${fixture}/AndroidDeviceSmoke.arcproject" DESTINATION "${ARC_TEST_ROOT}")
file(COPY "${fixture}/Content/Scenes/Startup.arcscene"
          "${fixture}/Content/Scenes/Startup.arcscene.arcmeta"
     DESTINATION "${ARC_TEST_ROOT}/Content/Scenes")
file(COPY "${ARC_SOURCE_ROOT}/templates/blank-3d/Content/models/UAL2_Standard.glb"
          "${ARC_SOURCE_ROOT}/templates/blank-3d/Content/models/UAL2_Standard.glb.arcmeta"
     DESTINATION "${ARC_TEST_ROOT}/Content/models")
file(COPY "${ARC_SOURCE_ROOT}/assets/materials/baseline/unlit_texture.arcmat"
          "${ARC_SOURCE_ROOT}/assets/materials/baseline/unlit_texture.arcmat.arcmeta"
     DESTINATION "${ARC_TEST_ROOT}/Content/materials")
file(COPY
    "${ARC_SOURCE_ROOT}/assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg"
    "${ARC_SOURCE_ROOT}/assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/Content/textures/terrain/aerial_grass_rock")

set(output "${ARC_TEST_ROOT}/Cooked")
set(manifest "${output}/android-arm64-vulkan.arccookmanifest")
set(scene "Content/Scenes/Startup.arcscene")

message(STATUS "Android V3.4f.1: cooking isolated Android arm64 Vulkan scene")
execute_process(
    COMMAND "${ARC_COOK}" cook --profile android-arm64-vulkan --project "${ARC_TEST_ROOT}" --root "${scene}"
            --output "${output}" --json
    RESULT_VARIABLE cook_status OUTPUT_VARIABLE cook_output ERROR_VARIABLE cook_error
    TIMEOUT ${ARC_DEVICE_SMOKE_TIMEOUT})
if(NOT cook_status EQUAL 0 OR NOT EXISTS "${manifest}")
    message(FATAL_ERROR "Android smoke cook failed (${cook_status}):\n${cook_output}\n${cook_error}")
endif()

message(STATUS "Android V3.4f.1: packaging source-free asset manifest")
execute_process(
    COMMAND "${ARC_COOK}" package --profile android-arm64-vulkan --project "${ARC_TEST_ROOT}" --root "${scene}"
            --output "${output}" --json
    RESULT_VARIABLE package_status OUTPUT_VARIABLE package_output ERROR_VARIABLE package_error
    TIMEOUT ${ARC_DEVICE_SMOKE_TIMEOUT})
if(NOT package_status EQUAL 0)
    message(FATAL_ERROR "Android smoke package failed (${package_status}):\n${package_output}\n${package_error}")
endif()

execute_process(
    COMMAND "${ARC_COOK}" verify --profile android-arm64-vulkan --project "${ARC_TEST_ROOT}" --output "${output}" --json
    RESULT_VARIABLE verify_status OUTPUT_VARIABLE verify_output ERROR_VARIABLE verify_error
    TIMEOUT 120)
if(NOT verify_status EQUAL 0)
    message(FATAL_ERROR "Android smoke package verification failed (${verify_status}):\n${verify_output}\n${verify_error}")
endif()

message(STATUS "Android V3.4f.1: runtime contract verification")
execute_process(
    COMMAND "${ARC_DEVICE_SMOKE_VERIFY}" "${manifest}" "${ARC_TEST_ROOT}"
    RESULT_VARIABLE inspect_status OUTPUT_VARIABLE inspect_output ERROR_VARIABLE inspect_error
    TIMEOUT 120)
if(NOT inspect_status EQUAL 0)
    message(FATAL_ERROR "Android smoke runtime artifact checks failed (${inspect_status}):\n${inspect_output}\n${inspect_error}")
endif()
message(STATUS "${inspect_output}")

# Distribution A3: pass the real Android cook/package result through the
# shared ARC D0 assembler and inspect the staged APK asset payload.
if(DEFINED ARC_PROJECT_TOOL)
    file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/Config")
    set(config "${ARC_TEST_ROOT}/Config/runtime.json")
    file(WRITE "${config}" "{\n  \"tickRate\": 60\n}\n")
    set(stage "${ARC_TEST_ROOT}/StagedAndroidAssets")
    set(stage_command
        "${ARC_PROJECT_TOOL}" package android-content
        --project "${ARC_TEST_ROOT}/AndroidDeviceSmoke.arcproject"
        --profile android-arm64-vulkan --target android-arm64-vulkan
        --cook-manifest "${manifest}" --runtime-config "${config}"
        --output "${stage}" --json)
    execute_process(COMMAND ${stage_command}
        RESULT_VARIABLE stage_status OUTPUT_VARIABLE stage_output ERROR_VARIABLE stage_error
        TIMEOUT 120)
    if(NOT stage_status EQUAL 0)
        message(FATAL_ERROR "Android A3 distribution stage failed (${stage_status}):\n${stage_output}\n${stage_error}")
    endif()
    if(NOT EXISTS "${stage}/arc-distribution.json" OR
       NOT EXISTS "${stage}/Config/runtime.json" OR
       NOT EXISTS "${stage}/Content/android-arm64-vulkan.arccookmanifest")
        message(FATAL_ERROR "Android A3 stage did not publish complete D0 cooked assets")
    endif()
    file(READ "${stage}/arc-distribution.json" d0_json)
    string(JSON d0_platform GET "${d0_json}" target platform)
    string(JSON d0_entrypoint GET "${d0_json}" entrypoint kind)
    string(JSON d0_scene GET "${d0_json}" startupScenes 0)
    if(NOT d0_platform STREQUAL "android" OR NOT d0_entrypoint STREQUAL "android-activity" OR
       NOT d0_scene STREQUAL "b80a0000-0000-4000-8000-000000000005")
        message(FATAL_ERROR "Android A3 D0 identity, activity or startup scene is incorrect")
    endif()
    file(GLOB staged_packages "${stage}/Content/*.arcpak")
    if(NOT staged_packages)
        message(FATAL_ERROR "Android A3 stage did not publish cooked package chunks")
    endif()
    execute_process(COMMAND ${stage_command} RESULT_VARIABLE repeated_status
        OUTPUT_QUIET ERROR_QUIET TIMEOUT 120)
    if(repeated_status EQUAL 0)
        message(FATAL_ERROR "Android A3 content stage unexpectedly overwrote an existing distribution")
    endif()
    # A missing chunk must fail without publishing an incomplete output.
    list(GET staged_packages 0 first_package)
    get_filename_component(first_name "${first_package}" NAME)
    set(source_package "${output}/${first_name}")
    file(RENAME "${source_package}" "${source_package}.hidden")
    execute_process(
        COMMAND "${ARC_PROJECT_TOOL}" package android-content
            --project "${ARC_TEST_ROOT}/AndroidDeviceSmoke.arcproject"
            --profile android-arm64-vulkan --target android-arm64-vulkan
            --cook-manifest "${manifest}" --runtime-config "${config}"
            --output "${ARC_TEST_ROOT}/MissingChunkStage" --json
        RESULT_VARIABLE missing_status OUTPUT_QUIET ERROR_QUIET TIMEOUT 120)
    file(RENAME "${source_package}.hidden" "${source_package}")
    if(missing_status EQUAL 0 OR EXISTS "${ARC_TEST_ROOT}/MissingChunkStage" OR
       EXISTS "${ARC_TEST_ROOT}/MissingChunkStage.assemble" OR
       EXISTS "${ARC_TEST_ROOT}/MissingChunkStage.pending")
        message(FATAL_ERROR "Android A3 did not roll back a missing cooked chunk")
    endif()
    message(STATUS "Android A3: verified D0 content stage, activity, scene, chunk closure and rollback")
endif()

