#include "workflow_command.h"

#include <arc/project/package_workflow.h>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <iostream>
#include <map>

int handle_workflow_command(const std::vector<std::string>& args)
{
    namespace project = arc::project;
    using json = nlohmann::json;
    const auto fail = [](const std::string& field, const std::string& message)
    {
        std::cout << json{{"succeeded", false}, {"field", field}, {"error", message}}.dump() << '\n';
        return 1;
    };
    if (args.size() < 2 || args[1] != "plan") return fail("action", "Expected workflow plan");
    std::map<std::string, std::string> options;
    for (std::size_t index = 2; index < args.size(); ++index)
    {
        const auto& key = args[index];
        if (key == "--json") continue;
        if (key != "--project" && key != "--profile" && key != "--target" && key != "--action" &&
            key != "--destination" && key != "--session-id")
            return fail("arguments", "Unknown option: " + key);
        if (index + 1 == args.size() || args[index + 1].starts_with("--") || options.contains(key))
            return fail("arguments", "Missing or duplicate value: " + key);
        options.emplace(key, args[++index]);
    }
    for (const auto* required : {"--project", "--profile", "--target", "--action"})
        if (!options.contains(required) || options.at(required).empty()) return fail(required, "Required option");
    const std::map<std::string, project::project_workflow_action> actions{
        {"configure", project::project_workflow_action::configure},
        {"buildCode", project::project_workflow_action::build_code},
        {"cook", project::project_workflow_action::cook},
        {"build", project::project_workflow_action::build},
        {"play", project::project_workflow_action::play},
        {"runBuilt", project::project_workflow_action::run_built}};
    const auto action = actions.find(options.at("--action"));
    if (action == actions.end()) return fail("action", "Unknown workflow action");
    const std::map<std::string, project::project_workflow_destination> destinations{
        {"none", project::project_workflow_destination::none},
        {"local", project::project_workflow_destination::local},
        {"device", project::project_workflow_destination::device}};
    const auto destination_name = options.contains("--destination") ? options.at("--destination") : "none";
    const auto destination = destinations.find(destination_name);
    if (destination == destinations.end()) return fail("destination", "Unknown workflow destination");
    const auto descriptor_path = std::filesystem::absolute(options.at("--project"));
    const auto descriptor = project::load_descriptor(descriptor_path);
    if (!descriptor) return fail(descriptor.error().field, descriptor.error().message);
    const auto context = project::resolve_context(descriptor_path, descriptor.value());
    if (!context) return fail(context.error().field, context.error().message);
    const auto plan = project::plan_project_workflow(
        descriptor.value(), context.value(),
        {.action = action->second,
         .profile = options.at("--profile"),
         .target_id = options.at("--target"),
         .destination = destination->second,
         .session_id = options.contains("--session-id") ? options.at("--session-id") : ""});
    if (!plan) return fail(plan.error().field, plan.error().message);
    const auto& output = plan.value().output;
    json stages = json::array();
    for (const auto stage : plan.value().stages)
        stages.push_back(project::workflow_stage_name(stage));
    // A plan describes intent, not executable support. Platform adapters independently preflight every stage.
    std::cout << json{{"succeeded", true},
                      {"version", 1},
                      {"action", action->first},
                      {"destination", destination_name},
                      {"sessionId", plan.value().session_id},
                      {"projectGuid", output.project_guid},
                      {"engineVersion", output.engine_version},
                      {"profile", output.profile},
                      {"targetId", output.target_id},
                      {"configuration", output.configuration},
                      {"platform", output.platform},
                      {"architecture", output.architecture},
                      {"renderer", output.renderer},
                      {"finalizer", output.finalizer},
                      {"cookOutput", output.cook_output.generic_string()},
                      {"packageOutput", output.package_output.generic_string()},
                      {"stages", stages}}
                     .dump()
              << '\n';
    return 0;
}
