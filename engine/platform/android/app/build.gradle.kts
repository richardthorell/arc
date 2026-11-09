plugins {
    id("com.android.application")
}

android {
    namespace = "org.arcengine.host"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "org.arcengine.host"
        minSdk = 26
        targetSdk = 28

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
