import groovy.json.JsonSlurper
import java.io.File
import java.security.MessageDigest

plugins {
    id("com.android.application")
}

// The checkout fixture compiles the ARC host from source. A distributed Android
// launcher consumes prebuilt native host libraries exported into an installed
// ARC SDK, without reading the engine source tree or invoking CMake.
val sdkRootArgument = providers.gradleProperty("arcSdkRoot").orNull
val engineRootArgument = providers.gradleProperty("arcEngineRoot").orNull
val projectRootArgument = providers.gradleProperty("arcProjectRoot").orNull
require(projectRootArgument == null || projectRootArgument.isNotBlank()) {
    "arcProjectRoot must not be empty"
}
require(projectRootArgument == null || sdkRootArgument != null) {
    "arcProjectRoot requires a selected installed ARC SDK via -ParcSdkRoot"
}
require(sdkRootArgument == null || sdkRootArgument.isNotBlank()) { "arcSdkRoot must not be empty" }
require(engineRootArgument == null || engineRootArgument.isNotBlank()) { "arcEngineRoot must not be empty" }
require(sdkRootArgument == null || engineRootArgument == null) {
    "arcSdkRoot and arcEngineRoot are mutually exclusive"
}

fun absoluteDirectory(argument: String, name: String): File {
    val directory = File(argument)
    require(directory.isAbsolute && directory.isDirectory) {
        name + " must be an existing absolute directory: " + argument
    }
    return directory.canonicalFile
}

val sdkRoot = sdkRootArgument?.let { absoluteDirectory(it, "arcSdkRoot") }
val projectRoot = projectRootArgument?.let { absoluteDirectory(it, "arcProjectRoot") }
val engineRoot = if (sdkRoot == null) {
    val root = engineRootArgument?.let { absoluteDirectory(it, "arcEngineRoot") }
        ?: File(rootProject.projectDir, "../../..").canonicalFile
    require(File(root, "engine/project/src/common/runtime_boot.cpp").isFile &&
            File(root, "cmake/ArcBuildPolicy.cmake").isFile) {
        "ARC Android source build requires the full engine tree; pass -ParcEngineRoot=/absolute/path"
    }
    root
} else null

val hostRoot = sdkRoot?.resolve("share/arc/android/host")
if (sdkRoot != null) {
    require(File(sdkRoot, "lib/cmake/ARC/ARCConfig.cmake").isFile) {
        "Selected ARC SDK does not contain lib/cmake/ARC/ARCConfig.cmake"
    }
    val installation = File(sdkRoot, "arc-installation.json")
    require(installation.isFile) { "Selected ARC SDK has no arc-installation.json" }
    val installed = JsonSlurper().parse(installation) as? Map<*, *>
        ?: error("Invalid ARC SDK installation manifest")
    val sdkVersion = installed["engineVersion"] as? String
    require(installed["format"] == "arc-installation" && !sdkVersion.isNullOrBlank()) {
        "Invalid ARC SDK version or installation format"
    }
    val descriptor = File(hostRoot!!, "host.json")
    require(descriptor.isFile) { "Selected ARC SDK has no Android host; export the Android Gradle host first" }
    val host = JsonSlurper().parse(descriptor) as? Map<*, *>
        ?: error("Invalid Android host SDK descriptor")
    require(host["format"] == "arc.android-host-sdk" && host["version"] == 1 &&
            host["engineVersion"] == sdkVersion && host["abi"] == "arm64-v8a") {
        "Android host SDK format, architecture or engine version does not match the installed ARC SDK"
    }
    val configurations = host["configurations"] as? Map<*, *>
        ?: error("Android host SDK has no configurations")
    for (variant in listOf("debug", "release")) {
        val libraries = configurations[variant] as? List<*>
            ?: error("Android host SDK is missing " + variant + " native libraries")
        val seen = mutableSetOf<String>()
        for (item in libraries) {
            val entry = item as? Map<*, *> ?: error("Invalid Android host library record")
            val filename = entry["name"] as? String ?: error("Missing Android host library filename")
            val expectedSha = entry["sha256"] as? String ?: error("Missing Android host library digest")
            require(filename.matches(Regex("lib[A-Za-z0-9_.+-]+\\.so")) && seen.add(filename) &&
                    expectedSha.matches(Regex("[0-9a-f]{64}"))) {
                "Invalid or duplicate Android " + variant + " native library entry: " + filename
            }
            val library = File(hostRoot, variant + "/jniLibs/arm64-v8a/" + filename)
            require(library.isFile) { "Missing Android " + variant + " native library: " + filename }
            val actualSha = library.inputStream().use { stream ->
                val digest = MessageDigest.getInstance("SHA-256")
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
                digest.digest().joinToString("") { "%02x".format(it) }
            }
            require(actualSha == expectedSha) {
                "Android " + variant + " native library digest mismatch: " + filename
            }
        }
        require("libarc_android_host.so" in seen) {
            "Android " + variant + " host library is missing from the installed ARC SDK"
        }
    }
}


// Produced by `arc-project settings export-android --project ... --json`.
// A supplied project export is mandatory and validated; never fall back to the
// engine-host identity when a project-bound invocation is misconfigured.
val projectSettingsPath = providers.gradleProperty("arcProjectSettings").orNull
val projectSettings: Map<*, *>? = projectSettingsPath?.let {
    val source = file(it)
    require(source.isFile && source.length() <= 65536) { "Missing or oversized ARC project settings export" }
    val value = groovy.json.JsonSlurper().parse(source) as? Map<*, *>
        ?: error("Invalid ARC project settings export")
    require(value["format"] == "arc.android-project-settings" && value["version"] == 1) { "Unsupported ARC project settings export" }
    require((value["projectGuid"] as? String)?.isNotBlank() == true) { "Missing ARC project identity" }
    require((value["applicationId"] as? String)?.matches(Regex("[A-Za-z][A-Za-z0-9_]*(\\.[A-Za-z][A-Za-z0-9_]*)+")) == true) { "Invalid Android application ID" }
    require((value["versionName"] as? String)?.isNotBlank() == true) { "Missing Android version name" }
    require((value["displayName"] as? String)?.isNotBlank() == true) { "Missing Android display name" }
    fun integer(key: String): Int {
        val number = value[key] as? Number ?: error("Missing integer $key")
        val result = number.toLong()
        require(result in 1..2100000000 && result.toDouble() == number.toDouble()) { "Invalid integer $key" }
        return result.toInt()
    }
    integer("versionCode")
    require(integer("minimumSdk") >= 26 && integer("targetSdk") in integer("minimumSdk")..35) { "Unsupported Android SDK settings" }
    value
}

android {
    namespace = "org.arcengine.host"
    compileSdk = 35
    ndkVersion = "30.0.16248370"

    defaultConfig {
        applicationId = projectSettings?.get("applicationId") as? String ?: "org.arcengine.host"
        minSdk = (projectSettings?.get("minimumSdk") as? Number)?.toInt() ?: 26
        targetSdk = (projectSettings?.get("targetSdk") as? Number)?.toInt() ?: 28
        projectSettings?.let {
            versionName = it["versionName"] as String
            versionCode = (it["versionCode"] as Number).toInt()
        }
        manifestPlaceholders["arcApplicationLabel"] = projectSettings?.get("displayName") ?: "ARC Native Host"

        if (sdkRoot == null || projectRoot != null) {
            externalNativeBuild {
                cmake {
                    cppFlags += "-std=c++20"
                    if (sdkRoot == null) {
                        arguments += "-DARC_ANDROID_ENGINE_ROOT=" + engineRoot!!.absolutePath
                    } else {
                        arguments += "-DARC_ANDROID_SDK_ROOT=" + sdkRoot.absolutePath
                        arguments += "-DARC_ANDROID_PROJECT_ROOT=" + projectRoot!!.absolutePath
                    }
                }
            }
        }

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        getByName("debug") {
            isDebuggable = true
        }
        getByName("release") {
            isMinifyEnabled = false
        }
    }

    androidResources {
        // Keep package chunks seekable through AAssetManager's ranged API.
        noCompress += listOf("arcpak", "arccookmanifest")
    }

    if (sdkRoot == null || projectRoot != null) {
        externalNativeBuild {
            cmake {
                path = file(
                    if (projectRoot != null) "src/main/cpp/project_module/CMakeLists.txt"
                    else "src/main/cpp/CMakeLists.txt"
                )
                version = "3.22.1"
            }
        }
    } else {
        // A1 host-only mode: no project native code or CMake invocation.
        sourceSets {
            getByName("debug") {
                jniLibs.srcDir(File(hostRoot, "debug/jniLibs"))
            }
            getByName("release") {
                jniLibs.srcDir(File(hostRoot, "release/jniLibs"))
            }
        }
    }
}

// Configuration-only regression probe: no APK build or device is required.
tasks.register("verifyArcProjectSettings") {
    doLast {
        val settings = projectSettings ?: error("Supply -ParcProjectSettings for this verification")
        check(android.namespace == "org.arcengine.host")
        check(android.defaultConfig.applicationId == settings["applicationId"])
        check(android.defaultConfig.versionName == settings["versionName"])
        check(android.defaultConfig.versionCode == (settings["versionCode"] as Number).toInt())
        check(android.defaultConfig.minSdk == (settings["minimumSdk"] as Number).toInt())
        check(android.defaultConfig.targetSdk == (settings["targetSdk"] as Number).toInt())
        check(android.defaultConfig.manifestPlaceholders["arcApplicationLabel"] == settings["displayName"])
    }
}

// Configuration-only smoke check for the selected host mode.
tasks.register("verifyArcAndroidHostMode") {
    doLast {
        check(if (sdkRoot != null) hostRoot!!.resolve("host.json").isFile else engineRoot!!.isDirectory)
    }
}
