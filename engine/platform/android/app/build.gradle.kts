import groovy.json.JsonSlurper
import java.io.File
import java.security.MessageDigest
import java.nio.file.Files

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


// A3: consume only a fully assembled and hashed D0 content stage. There is no
// implicit working-tree assets fallback for project-bound APKs.
val distributionAssetsArgument = providers.gradleProperty("arcDistributionAssets").orNull
require(distributionAssetsArgument == null || distributionAssetsArgument.isNotBlank()) {
    "arcDistributionAssets must not be empty"
}
require(distributionAssetsArgument == null || (projectRoot != null && sdkRoot != null)) {
    "arcDistributionAssets requires both arcProjectRoot and arcSdkRoot"
}
val distributionAssetsRoot = distributionAssetsArgument?.let { absoluteDirectory(it, "arcDistributionAssets") }
val distributionTargetConfiguration: String? = distributionAssetsRoot?.let { root ->
    val distributionFile = File(root, "arc-distribution.json")
    require(distributionFile.isFile && distributionFile.length() in 1..16777216) {
        "Android distribution stage has no valid arc-distribution.json"
    }
    val distribution = JsonSlurper().parse(distributionFile) as? Map<*, *>
        ?: error("Invalid Android D0 manifest")
    require(distribution["format"] == "arc.distribution" &&
            (distribution["formatVersion"] as? Number)?.toInt() == 1) {
        "Android distribution stage has unsupported format"
    }
    val target = distribution["target"] as? Map<*, *> ?: error("Missing Android D0 target")
    require(target["platform"] == "android" && target["architecture"] == "arm64" &&
            target["renderer"] == "vulkan" && target["id"] == "android-arm64-vulkan") {
        "Android D0 target must be android-arm64-vulkan"
    }
    val configuration = target["configuration"] as? String
    require(configuration in listOf("Debug", "Shipping")) {
        "Android D0 stage requires a Debug or Shipping configuration"
    }
    require(distribution["role"] == "client" && distribution["finalizer"] == "android-gradle") {
        "Android D0 stage must be a client android-gradle distribution"
    }
    val entrypoint = distribution["entrypoint"] as? Map<*, *> ?: error("Missing Android activity entrypoint")
    require(entrypoint["kind"] == "android-activity" && entrypoint["value"] == "android.app.NativeActivity") {
        "Android D0 stage has the wrong NativeActivity entrypoint"
    }
    val installed = JsonSlurper().parse(File(sdkRoot!!, "arc-installation.json")) as? Map<*, *>
        ?: error("Invalid installed ARC SDK")
    require(distribution["engineVersion"] == installed["engineVersion"] &&
            distribution["sdkVersion"] == installed["engineVersion"]) {
        "Android D0 stage engine/SDK versions do not match selected native SDK"
    }
    val projectDescriptors = projectRoot!!.listFiles()
        ?.filter { it.isFile && it.name.endsWith(".arcproject") } ?: emptyList()
    require(projectDescriptors.size == 1) { "Android project root must contain one .arcproject descriptor" }
    val descriptor = JsonSlurper().parse(projectDescriptors.single()) as? Map<*, *>
        ?: error("Invalid ARC project descriptor")
    require(distribution["projectGuid"] == descriptor["guid"]) {
        "Android staged project GUID differs from compiled native project"
    }
    val scenes = distribution["startupScenes"] as? List<*>
    require(!scenes.isNullOrEmpty() && scenes.all { it is String && it.isNotBlank() }) {
        "Android D0 stage must declare startup scenes"
    }
    val configPath = distribution["runtimeConfig"] as? String ?: error("Missing runtime configuration")
    val content = distribution["content"] as? Map<*, *> ?: error("Missing staged content")
    val cookManifests = content["cookManifests"] as? List<*>
    require(!cookManifests.isNullOrEmpty() && cookManifests.all {
        it is String && it.startsWith("Content/") && it.endsWith(".arccookmanifest")
    }) { "Android D0 stage requires cooked-content manifests" }

    val files = distribution["files"] as? List<*> ?: error("Missing Android asset inventory")
    require(files.isNotEmpty()) { "Android D0 stage has no assets" }
    val declared = mutableSetOf<String>()
    val kinds = mutableMapOf<String, String>()
    for (item in files) {
        val record = item as? Map<*, *> ?: error("Invalid Android asset inventory entry")
        val relative = record["path"] as? String ?: error("Missing staged asset path")
        val parts = relative.split("/")
        require(parts.size >= 2 && parts.all { it.isNotEmpty() && it != "." && it != ".." } &&
                !relative.contains('\\') && !relative.contains(':') &&
                (parts.first() == "Content" || parts.first() == "Config")) {
            "Unsafe Android staged asset path: " + relative
        }
        require(declared.add(relative.lowercase())) { "Duplicate Android staged asset path: " + relative }
        val kind = record["kind"] as? String ?: error("Missing staged asset kind")
        require(kind == if (parts.first() == "Config") "configuration" else "content") {
            "Incorrect staged Android asset kind: " + relative
        }
        val expectedSize = record["size"] as? Number ?: error("Missing staged asset size")
        val expectedSha = record["sha256"] as? String ?: error("Missing staged asset hash")
        require(expectedSize.toLong() >= 0 && expectedSha.matches(Regex("[0-9a-f]{64}"))) {
            "Invalid Android asset inventory metadata: " + relative
        }
        val candidate = File(root, relative)
        require(candidate.isFile && !Files.isSymbolicLink(candidate.toPath()) &&
                candidate.length() == expectedSize.toLong()) {
            "Missing, symbolic or incorrect-size Android asset: " + relative
        }
        val digest = MessageDigest.getInstance("SHA-256")
        candidate.inputStream().use { stream ->
            val bytes = ByteArray(64 * 1024)
            while (true) {
                val count = stream.read(bytes)
                if (count < 0) break
                digest.update(bytes, 0, count)
            }
        }
        val actual = digest.digest().joinToString("") { "%02x".format(it) }
        require(actual == expectedSha) { "Android asset SHA-256 mismatch: " + relative }
        kinds[relative] = kind
    }
    require(kinds[configPath] == "configuration" &&
            cookManifests.all { kinds[it] == "content" }) {
        "Android runtime config or cooked-content manifests missing from inventory"
    }
    // A3 stages precisely the hashed D0 file set. Reject accidental authoring
    // sources and stale chunks, including symlinks that could escape staging.
    val present = mutableSetOf<String>()
    Files.walk(root.toPath()).use { paths ->
        paths.forEach { file ->
            if (file == root.toPath()) return@forEach
            require(!Files.isSymbolicLink(file)) { "Android stage cannot contain symlinks" }
            if (Files.isRegularFile(file)) {
                val relative = root.toPath().relativize(file).toString().replace('\\', '/')
                if (relative != "arc-distribution.json") present.add(relative.lowercase())
            }
        }
    }
    require(present == declared) { "Android stage has untracked or missing cooked assets" }
    configuration
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

    if (distributionAssetsRoot != null) {
        sourceSets.getByName("main").assets.srcDir(distributionAssetsRoot)
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

// Validate the final selected variant, not merely whether a D0 stage exists.
// Host Debug expects Debug cooked content; host Release expects Shipping.
tasks.matching { it.name == "mergeDebugAssets" || it.name == "mergeReleaseAssets" }.configureEach {
    doFirst {
        if (distributionAssetsRoot != null) {
            val expected = if (name == "mergeDebugAssets") "Debug" else "Shipping"
            require(distributionTargetConfiguration == expected) {
                "Android D0 stage configuration " + distributionTargetConfiguration +
                    " is incompatible with Gradle " + name + " (expected " + expected + ")"
            }
        }
    }
}
