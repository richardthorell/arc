plugins {
    id("com.android.application")
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

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++20"
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

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
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
