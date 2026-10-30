# Native, clean-checkout Materials Baseline cook/package smoke for #1086.
# No editor process, GPU, or mutable source-tree cook cache is required.
foreach(required ARC_COOK ARC_MATERIALS_VERIFY ARC_SOURCE_ROOT ARC_TEST_ROOT)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Materials Baseline smoke: missing ${required}")
    endif()
endforeach()

if(NOT DEFINED ARC_BASELINE_TIMEOUT)
    set(ARC_BASELINE_TIMEOUT 360)
endif()

file(REMOVE_RECURSE "${ARC_TEST_ROOT}")
file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/assets/fixtures")
file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/assets/textures/terrain/aerial_grass_rock")
file(COPY "${ARC_SOURCE_ROOT}/MaterialsBaseline.arcproject" DESTINATION "${ARC_TEST_ROOT}")
file(COPY
    "${ARC_SOURCE_ROOT}/assets/fixtures/materials_baseline.arcscene"
    "${ARC_SOURCE_ROOT}/assets/fixtures/materials_baseline.arcscene.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/fixtures")
file(COPY "${ARC_SOURCE_ROOT}/assets/materials"
          "${ARC_SOURCE_ROOT}/assets/material_functions"
          "${ARC_SOURCE_ROOT}/assets/shaders"
     DESTINATION "${ARC_TEST_ROOT}/assets")
file(COPY
    "${ARC_SOURCE_ROOT}/assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg"
    "${ARC_SOURCE_ROOT}/assets/textures/terrain/aerial_grass_rock/aerial_grass_rock_diff_1k.jpg.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/textures/terrain/aerial_grass_rock")

set(output "${ARC_TEST_ROOT}/Cooked")
set(manifest "${output}/windows-x64-vulkan.arccookmanifest")
set(root "assets/fixtures/materials_baseline.arcscene")

message(STATUS "Materials Baseline: cold native cook")
execute_process(
    COMMAND "${ARC_COOK}" cook --project "${ARC_TEST_ROOT}" --root "${root}" --output "${output}" --json
    RESULT_VARIABLE cold_status OUTPUT_VARIABLE cold_output ERROR_VARIABLE cold_error
    TIMEOUT ${ARC_BASELINE_TIMEOUT})
if(NOT cold_status EQUAL 0 OR NOT EXISTS "${manifest}")
    message(FATAL_ERROR "Materials Baseline cold cook failed (${cold_status}):\n${cold_output}\n${cold_error}")
endif()

message(STATUS "Materials Baseline: warm native cook must be a cache hit")
execute_process(
    COMMAND "${ARC_COOK}" cook --project "${ARC_TEST_ROOT}" --root "${root}" --output "${output}" --json
    RESULT_VARIABLE warm_status OUTPUT_VARIABLE warm_output ERROR_VARIABLE warm_error
    TIMEOUT ${ARC_BASELINE_TIMEOUT})
if(NOT warm_status EQUAL 0 OR NOT warm_output MATCHES "\"cooked\":0")
    message(FATAL_ERROR "Materials Baseline cache hit failed (${warm_status}):\n${warm_output}\n${warm_error}")
endif()

message(STATUS "Materials Baseline: package and verify real runtime artifacts")
execute_process(
    COMMAND "${ARC_COOK}" package --project "${ARC_TEST_ROOT}" --root "${root}" --output "${output}" --json
    RESULT_VARIABLE package_status OUTPUT_VARIABLE package_output ERROR_VARIABLE package_error
    TIMEOUT ${ARC_BASELINE_TIMEOUT})
if(NOT package_status EQUAL 0 OR NOT EXISTS "${manifest}")
    message(FATAL_ERROR "Materials Baseline package failed (${package_status}):\n${package_output}\n${package_error}")
endif()

execute_process(
    COMMAND "${ARC_COOK}" verify --project "${ARC_TEST_ROOT}" --output "${output}" --json
    RESULT_VARIABLE verify_status OUTPUT_VARIABLE verify_output ERROR_VARIABLE verify_error
    TIMEOUT 90)
if(NOT verify_status EQUAL 0)
    message(FATAL_ERROR "Materials Baseline package verify failed (${verify_status}):\n${verify_output}\n${verify_error}")
endif()

execute_process(
    COMMAND "${ARC_MATERIALS_VERIFY}" "${manifest}" "${ARC_TEST_ROOT}"
    RESULT_VARIABLE inspect_status OUTPUT_VARIABLE inspect_output ERROR_VARIABLE inspect_error
    TIMEOUT 90)
if(NOT inspect_status EQUAL 0)
    message(FATAL_ERROR "Materials Baseline native package/load checks failed (${inspect_status}):\n${inspect_output}\n${inspect_error}")
endif()
message(STATUS "${inspect_output}")

# A targeted authored change must invalidate the warm cache rather than
# reusing the old cooked Material Instance package.
set(instance "${ARC_TEST_ROOT}/assets/materials/baseline/dielectric_satin.arcmatinst")
file(READ "${instance}" original_instance)
string(REPLACE "0.56" "0.57" changed_instance "${original_instance}")
if(changed_instance STREQUAL original_instance)
    message(FATAL_ERROR "Materials Baseline cache-invalidation fixture did not match expected override")
endif()
file(WRITE "${instance}" "${changed_instance}")
execute_process(
    COMMAND "${ARC_COOK}" cook --project "${ARC_TEST_ROOT}" --root "${root}" --output "${output}" --json
    RESULT_VARIABLE changed_status OUTPUT_VARIABLE changed_output ERROR_VARIABLE changed_error
    TIMEOUT ${ARC_BASELINE_TIMEOUT})
if(NOT changed_status EQUAL 0 OR changed_output MATCHES "\"cooked\":0")
    message(FATAL_ERROR "Materials Baseline override change did not invalidate cook (${changed_status}):\n${changed_output}\n${changed_error}")
endif()
file(WRITE "${instance}" "${original_instance}")

message(STATUS "Materials Baseline: native cook/package/deserialize/cache smoke passed")
