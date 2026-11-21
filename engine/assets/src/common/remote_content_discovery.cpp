#include <arc/assets/remote_content_discovery.h>

#include <exception>
#include <string_view>
#include <utility>

namespace arc::assets
{

jobs::job_future<remote_release_discovery_result>
discover_remote_release(jobs::job_system& jobs, std::string origin, remote_content_expectations expectations,
                        io::content_http_limits limits, jobs::cancellation_token cancellation,
                        remote_release_fetch_executor executor)
{
    // A general worker waits on the dedicated IO worker. Keep cancellation out
    // of the coordinator descriptor so it always returns a structured failure.
    return jobs.submit_future(
        {.name = "assets.remote_content.discover",
         .priority = jobs::job_priority::normal,
         .affinity = jobs::job_affinity::any_worker},
        [&jobs, origin = std::move(origin), expectations = std::move(expectations), limits, cancellation,
         executor = std::move(executor)]() mutable -> remote_release_discovery_result
        {
            remote_release_discovery_result result;
            if (cancellation.stop_requested())
            {
                result.transport_error = io::content_http_error_code::cancelled;
                result.error = "Release lookup cancelled";
                return result;
            }

            io::content_http_release_request request{
                .origin = std::move(origin),
                .project_guid = expectations.project_guid,
                .target_profile = expectations.target.cook_profile};
            const auto preflight = io::validate_content_http_release_request(request, limits);
            if (!preflight.succeeded())
            {
                result.transport_error = preflight.code;
                result.error = preflight.message;
                return result;
            }

            auto future = executor ? executor(jobs, request, limits, cancellation)
                                   : io::fetch_content_http_release(jobs, request, limits, cancellation);
            if (!future.valid())
            {
                result.transport_error = io::content_http_error_code::transport_unavailable;
                result.error = "Release lookup backend returned no job";
                return result;
            }
            const auto waited = future.handle().wait_result();
            if (!waited.succeeded())
            {
                result.transport_error = cancellation.stop_requested()
                                             ? io::content_http_error_code::cancelled
                                             : io::content_http_error_code::network_failure;
                result.error = "Release lookup job did not complete";
                return result;
            }
            io::content_http_fetch_result fetched;
            try
            {
                fetched = future.get();
            }
            catch (const std::exception&)
            {
                result.transport_error = io::content_http_error_code::network_failure;
                result.error = "Release lookup job failed";
                return result;
            }
            if (!fetched.succeeded())
            {
                result.transport_error = fetched.validation.code;
                result.error = fetched.validation.message.empty() ? "Release lookup failed" : fetched.validation.message;
                return result;
            }
            if (cancellation.stop_requested())
            {
                result.transport_error = io::content_http_error_code::cancelled;
                result.error = "Release lookup cancelled";
                return result;
            }

            // A test or future platform backend cannot bypass the same
            // fail-closed response framing required of native WinHTTP.
            const auto verified = io::validate_content_http_release_response(request, *fetched.response, limits);
            if (!verified.succeeded())
            {
                result.transport_error = verified.code;
                result.error = verified.message;
                return result;
            }
            const auto& body = fetched.response->body;
            const std::string_view descriptor{reinterpret_cast<const char*>(body.data()), body.size()};
            auto parsed = parse_remote_content_release(descriptor, expectations);
            if (!parsed.succeeded())
            {
                result.error = "Remote release descriptor rejected: " + parsed.error;
                return result;
            }
            result.release = std::move(parsed.release);
            return result;
        });
}

} // namespace arc::assets
