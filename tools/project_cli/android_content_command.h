#pragma once

#include <string>
#include <vector>

// Build and verify the D0 content/configuration payload that will be embedded
// as APK assets. Native APK/JNI assembly is deliberately reserved for A4.
int handle_android_content_command(const std::vector<std::string>& args, bool json_output);
