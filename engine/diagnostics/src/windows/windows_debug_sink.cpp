#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <arc/diagnostics/log.h>

#include <windows.h>

#include <string>

namespace arc
{
namespace
{

class windows_debug_log_sink final : public log_sink
{
public:
    void write(const log_record& record) override
    {
        std::string line;
        line.reserve(record.category.size() + record.message.size() + 16);
        line.append("[");
        line.append(to_string(record.level));
        line.append("]");
        if (!record.category.empty())
        {
            line.append("[");
            line.append(record.category);
            line.append("]");
        }
        line.append(" ");
        line.append(record.message);
        line.append("\n");
        OutputDebugStringA(line.c_str());
    }
};

struct windows_debug_log_sink_installer
{
    windows_debug_log_sink_installer()
    {
        default_logger().add_sink(std::make_shared<windows_debug_log_sink>());
    }
};

const windows_debug_log_sink_installer installer;

} // namespace
} // namespace arc
