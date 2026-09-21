if(NOT DEFINED ARC_COOK OR NOT DEFINED ARC_SOURCE_ROOT OR NOT DEFINED ARC_TEST_ROOT)
    message(FATAL_ERROR "clean checkout cook test is missing required paths")
endif()

file(REMOVE_RECURSE "${ARC_TEST_ROOT}")
file(MAKE_DIRECTORY "${ARC_TEST_ROOT}/assets/fixtures")

file(COPY
    "${ARC_SOURCE_ROOT}/assets/fixtures/persistence_fixture.arcscene"
    "${ARC_SOURCE_ROOT}/assets/fixtures/persistence_fixture.arcscene.arcmeta"
    "${ARC_SOURCE_ROOT}/assets/fixtures/persistence_fixture.arcprefab.arcmeta"
    DESTINATION "${ARC_TEST_ROOT}/assets/fixtures"
)

# Keep this smoke focused on clean-checkout dependency discovery, incremental
# cooking, packaging, and verification. The production persistence prefab pulls
# in the editor startup mesh plus render assets, which makes the child arc-cook
# process unnecessarily expensive under LLVM coverage instrumentation.
file(WRITE "${ARC_TEST_ROOT}/assets/fixtures/persistence_fixture.arcprefab" [=[
{
  "format": "arc.prefab",
  "formatVersion": 2,
  "prefab": {
    "id": "d499b8de-bb46-4cb7-b58f-0dc6f53e0102",
    "name": "Persistence Fixture Prop",
    "root": "b60e59c7-2781-4cf8-99a9-ff70193c1001"
  },
  "entities": [
    {
      "id": "b60e59c7-2781-4cf8-99a9-ff70193c1001",
      "parent": null,
      "order": 0,
      "components": {
        "Name": {
          "typeId": "a7c00000000000010000000000000001",
          "version": 1,
          "value": "Cooked Persistence Prop"
        },
        "Transform": {
          "typeId": "a7c00000000000010000000000000002",
          "version": 1,
          "position": [0.0, 0.0, 0.0],
          "rotation": [0.0, 0.0, 0.0, 1.0],
          "scale": [1.0, 1.0, 1.0],
          "futureEditorNote": "This unknown field must survive a round trip."
        },
        "FutureGameplay": {
          "typeId": "f0010000000000010000000000000001",
          "version": 7,
          "opaqueState": {
            "enabled": true,
            "values": [1, 2, 3]
          }
        }
      }
    }
  ],
  "dependencies": []
}
]=])

file(WRITE "${ARC_TEST_ROOT}/CookFixture.arcproject" [=[
{
  "format":"arc-project","formatVersion":3,
  "guid":"00000000-0000-4000-8000-00000000c001","name":"Cook Fixture","engineVersion":"0.1.0",
  "paths":{"source":"Source","content":"assets","config":"Config","plugins":"Plugins","saved":"Saved","intermediate":"Intermediate","build":"Build"},
  "assetRoots":["assets"],"modules":[],"plugins":[],"startupScenes":[],
  "targetPlatforms":[{"id":"windows-x64-vulkan","enabled":true}],
  "toolchain":{"compiler":"auto","minimumVersion":"","generator":"auto","architecture":"x86_64","cppStandard":20},
  "buildConfigurations":["Debug","RelWithDebInfo","Shipping"],
  "renderer":{"backend":"vulkan","api":"1.2","quality":"standard"},
  "cookProfiles":[{"id":"windows-x64-vulkan","platform":"windows","architecture":"x86_64","renderer":"vulkan","api":"1.2","textures":{"outputs":["bc","astc"],"quality":"balanced"},"configuration":"Shipping"}],
  "package":{"applicationName":"Cook Fixture","companyName":"","output":"Build/Packages","regionChunks":true},
  "settings":{"editor":"Config/Editor.json","renderer":"Config/Renderer.json","input":"Config/Input.json"}
}
]=])

set(output "${ARC_TEST_ROOT}/out")
set(bc_output "${output}/bc")
set(astc_output "${output}/astc")
set(bc_manifest "${bc_output}/windows-x64-vulkan.arccookmanifest")
set(astc_manifest "${astc_output}/windows-x64-vulkan.arccookmanifest")

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
if(NOT EXISTS "${bc_manifest}" OR NOT EXISTS "${astc_manifest}")
    message(FATAL_ERROR "multi-output cook did not publish both texture-family manifests")
endif()
file(READ "${bc_manifest}" bc_manifest_json)
file(READ "${astc_manifest}" astc_manifest_json)
string(FIND "${bc_manifest_json}" "\"textures\": \"bc\"" bc_family_index)
string(FIND "${astc_manifest_json}" "\"textures\": \"astc\"" astc_family_index)
if(bc_family_index EQUAL -1)
    message(FATAL_ERROR "BC cook manifest does not declare the BC texture family")
endif()
if(astc_family_index EQUAL -1)
    message(FATAL_ERROR "ASTC cook manifest does not declare the ASTC texture family")
endif()

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
        --output "${output}" --json
    RESULT_VARIABLE package_result
    OUTPUT_VARIABLE package_output
    ERROR_VARIABLE package_error
    TIMEOUT 120
)
if(NOT package_result EQUAL 0)
    message(FATAL_ERROR "clean checkout package failed (${package_result}):\n${package_output}\n${package_error}")
endif()
message(STATUS "arc-cook-clean-checkout: package completed")
file(GLOB bc_packages "${bc_output}/*.arcpak")
file(GLOB astc_packages "${astc_output}/*.arcpak")
if(NOT bc_packages OR NOT astc_packages)
    message(FATAL_ERROR "multi-output package did not publish package chunks for both texture families")
endif()

message(STATUS "arc-cook-clean-checkout: starting verification (timeout: 60s)")
execute_process(
    COMMAND "${ARC_COOK}" verify --project "${ARC_TEST_ROOT}"
        --output "${output}" --json
    RESULT_VARIABLE verify_result
    OUTPUT_VARIABLE verify_output
    ERROR_VARIABLE verify_error
    TIMEOUT 60
)
if(NOT verify_result EQUAL 0)
    message(FATAL_ERROR "clean checkout package verification failed (${verify_result}):\n${verify_output}\n${verify_error}")
endif()
message(STATUS "arc-cook-clean-checkout: verification completed")
