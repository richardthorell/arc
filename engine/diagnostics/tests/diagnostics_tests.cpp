#include <arc/diagnostics.h>

#include <catch2/catch_test_macros.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace
{

class memory_sink final : public arc::log_sink
{
public:
    void write(const arc::log_record& record) override
    {
        std::lock_guard lock(mutex);
        records.push_back({ record.level, std::string(record.category), std::string(record.message) });
    }

    struct captured_record
    {
        arc::log_level level{};
        std::string category;
        std::string message;
    };

    std::mutex mutex;
    std::vector<captured_record> records;
};

} // namespace

TEST_CASE("logger filters records by minimum level")
{
    auto log = std::make_shared<arc::logger>();
    auto sink = std::make_shared<memory_sink>();
    log->add_sink(sink);
    log->set_min_level(arc::log_level::warn);

    log->write({ .level = arc::log_level::info, .category = "test", .message = "ignored" });
    log->write({ .level = arc::log_level::error, .category = "test", .message = "kept" });

    REQUIRE(sink->records.size() == 1);
    REQUIRE(sink->records[0].level == arc::log_level::error);
    REQUIRE(sink->records[0].message == "kept");
}

TEST_CASE("process logger fans out through registered sinks")
{
    auto previous = arc::get_logger();
    auto log = std::make_shared<arc::logger>();
    auto first = std::make_shared<memory_sink>();
    auto second = std::make_shared<memory_sink>();
    log->add_sink(first);
    log->add_sink(second);
    arc::set_logger(log);

    arc::info("diagnostics", "hello");

    REQUIRE(first->records.size() == 1);
    REQUIRE(second->records.size() == 1);
    REQUIRE(first->records[0].category == "diagnostics");
    REQUIRE(first->records[0].message == "hello");

    arc::set_logger(previous);
}

TEST_CASE("default logger is available")
{
    REQUIRE(arc::get_logger() != nullptr);
    REQUIRE(arc::to_string(arc::log_level::fatal) == "fatal");
}
