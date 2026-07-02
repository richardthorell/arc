#include <arc/editor/arc_host.h>
#include <arc/render/render.h>

#include <iostream>
#include <memory>
#include <string>

int main()
{
    auto host = std::make_shared<arc::editor::arc_host>(std::make_unique<arc::render::renderer>());

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
            continue;

        std::string error;
        if (line.find("\"kind\":\"query\"") != std::string::npos || line.find("\"kind\": \"query\"") != std::string::npos)
        {
            arc::editor::host_query_envelope query;
            if (!arc::editor::from_json(line, query, error))
            {
                std::cerr << "arc_host_process query parse error: " << error << '\n';
                std::cout << arc::editor::to_json(arc::editor::host_response{
                    .request_id = query.request_id,
                    .succeeded = false,
                    .error = error }) << '\n';
                std::cout.flush();
                continue;
            }
            std::cout << arc::editor::to_json(host->query(query)) << '\n';
        }
        else
        {
            arc::editor::host_command_envelope command;
            if (!arc::editor::from_json(line, command, error))
            {
                std::cerr << "arc_host_process command parse error: " << error << '\n';
                std::cout << arc::editor::to_json(arc::editor::host_response{
                    .request_id = command.request_id,
                    .succeeded = false,
                    .error = error }) << '\n';
                std::cout.flush();
                continue;
            }
            std::cout << arc::editor::to_json(host->execute(command)) << '\n';
        }

        for (const auto& event : host->poll_events())
            std::cout << arc::editor::to_json(event) << '\n';
        std::cout.flush();
    }

    return 0;
}
