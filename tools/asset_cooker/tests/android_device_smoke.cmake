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
