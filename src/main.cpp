#include "neuro/app.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using neuro::AppConfig;
using neuro::EngineState;
using neuro::KnowledgeEntry;
using neuro::KnowledgeNodeRun;
using neuro::LlmDecision;
using neuro::RuntimeSelfTestResult;
using neuro::SectionNode;

void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " -m /path/to/model.gguf [--topic \"topic\"] [--n-predict 192] [--n-ctx 2048] [--ngl 99]\n"
        << "       [--state-file neuro_state.bin] [--knowledge-node-process] [--knowledge-node-modes all|mode,mode]\n"
        << "       [--no-post-knowledge-node-process]\n"
        << "       [--test-mode] [--test-iterations N] [--test-event \"prompt\"]\n"
        << "\n"
        << "Interactive commands:\n"
        << "  /step <text>   add an event and run one LLM iteration\n"
        << "  /tick <secs>   decay/couple chemistry without LLM\n"
        << "  /knowledge [modes] process in-memory knowledge entries as KnowledgeNodes\n"
        << "  /knowledge-add <title>|<url>|<summary>|<text> add an in-memory knowledge entry\n"
        << "  /show          print engine state\n"
        << "  /save          write body-of-work.md only\n"
        << "  /quit          exit\n";
}

struct LaunchOptions {
    AppConfig config;
    bool test_mode = false;
    int test_iterations = 1;
    std::string test_event = "Produce a short grounded runtime self-test confirmation.";
};

LaunchOptions parse_args(int argc, char ** argv) {
    LaunchOptions launch;
    AppConfig & args = launch.config;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_value = [&](const std::string & flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + flag);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            args.model_path = need_value(a);
        } else if (a == "--topic") {
            args.topic = need_value(a);
        } else if (a == "--state-file") {
            args.state_file = need_value(a);
        } else if (a == "--n-predict") {
            args.n_predict = std::stoi(need_value(a));
        } else if (a == "--n-ctx") {
            args.n_ctx = std::stoi(need_value(a));
        } else if (a == "--ngl") {
            args.n_gpu_layers = std::stoi(need_value(a));
        } else if (a == "--knowledge-node-process") {
            args.knowledge_node_process = true;
        } else if (a == "--no-post-knowledge-node-process") {
            args.post_knowledge_node_process = false;
        } else if (a == "--knowledge-node-modes") {
            args.knowledge_node_modes = need_value(a);
        } else if (a == "--test-mode") {
            launch.test_mode = true;
        } else if (a == "--test-iterations") {
            launch.test_iterations = std::max(1, std::stoi(need_value(a)));
        } else if (a == "--test-event") {
            launch.test_event = need_value(a);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (args.model_path.empty()) {
        throw std::runtime_error("model path is required");
    }
    return launch;
}

std::string trim(const std::string & s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

void print_state(const EngineState & s) {
    std::cout << "\n--- engine state ---\n"
              << "topic: " << s.topic << '\n'
              << "iteration: " << s.iteration << '\n'
              << "current_state: " << s.current_state << '\n'
              << "next_state: " << s.next_state << '\n'
              << "authoring_mode: " << s.authoring_mode << '\n'
              << "chem: " << neuro::render_chemicals(s.chem_values) << '\n';
    if (!s.knowledge_context.empty()) {
        std::cout << "knowledge_context: loaded (" << s.knowledge_context.size() << " chars)\n";
    }
    if (const SectionNode * focused = s.graph.focused()) {
        std::cout << "focused: " << focused->title << "\n"
                  << focused->content << "\n";
    } else {
        std::cout << "focused: <none>\n";
    }
    std::cout << "--------------------\n\n";
}

int run_test_mode(const LaunchOptions & launch) {
    neuro::AppController app(launch.config);
    const RuntimeSelfTestResult result = app.run_self_test(launch.test_event, launch.test_iterations);
    std::cout << "Runtime self-test\n";
    std::cout << "status: " << (result.passed ? (result.degraded ? "DEGRADED" : "PASS") : "FAIL") << "\n";
    std::cout << "iterations: " << result.iterations_completed << "/" << result.iterations_requested << "\n";
    if (!result.json_probe_variant.empty()) {
        std::cout << "json probe variant: " << result.json_probe_variant << "\n";
    }
    std::cout << "json probe total ms: " << result.json_probe_metrics.total_ms << "\n";
    std::cout << "plan total ms: " << result.plan_metrics.total_ms << "\n";
    std::cout << "write total ms: " << result.write_metrics.total_ms << "\n";
    for (const auto & check : result.checks) {
        std::cout << (check.passed ? "[PASS] " : "[FAIL] ") << check.name << ": " << check.detail << "\n";
    }
    if (!result.last_decision.update_title.empty()) {
        std::cout << "last decision title: " << result.last_decision.update_title << "\n";
    }
    if (!result.last_decision.update_text.empty()) {
        std::cout << "last decision text: " << result.last_decision.update_text << "\n";
    }
    if (!result.passed) {
        return 1;
    }
    if (result.degraded) {
        return 2;
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        LaunchOptions launch = parse_args(argc, argv);
        if (launch.test_mode) {
            return run_test_mode(launch);
        }
        AppConfig config = launch.config;
        neuro::AppController app(config);
        app.load_state();
        if (auto run = app.maybe_process_initial_knowledge()) {
            std::cout << run->summary << "\n";
        }

        std::cout << "Loaded model: " << config.model_path << "\n";
        std::cout << "Topic: " << app.snapshot().topic << "\n";
        std::cout << "State file: " << config.state_file << "\n";
        std::cout << "Type /step <text> to run one direct libllama iteration.\n\n";

        std::string line;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, line)) {
                break;
            }
            line = trim(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/show") {
                print_state(app.snapshot());
                continue;
            }
            if (line == "/save") {
                app.save_markdown();
                std::cout << "saved body-of-work.md\n";
                continue;
            }
            if (line.rfind("/knowledge-add", 0) == 0) {
                std::string payload = trim(line.substr(14));
                std::vector<std::string> parts;
                std::string part;
                std::istringstream in(payload);
                while (std::getline(in, part, '|')) {
                    parts.push_back(trim(part));
                }
                KnowledgeEntry entry;
                if (!parts.empty()) entry.title = parts[0];
                if (parts.size() > 1) entry.url = parts[1];
                if (parts.size() > 2) entry.summary = parts[2];
                if (parts.size() > 3) entry.text = parts[3];
                if (entry.summary.empty() && !payload.empty()) {
                    entry.summary = payload;
                }
                app.add_knowledge_entry(std::move(entry));
                std::cout << "added in-memory knowledge entry; count=" << app.snapshot().knowledge_entries.size() << "\n";
                continue;
            }
            if (line.rfind("/knowledge", 0) == 0) {
                std::string raw_modes = trim(line.substr(10));
                if (raw_modes.empty()) {
                    raw_modes = config.knowledge_node_modes;
                }
                const KnowledgeNodeRun run = app.process_knowledge_modes(neuro::parse_knowledge_modes(raw_modes));
                std::cout << run.summary << "\n";
                continue;
            }
            if (line.rfind("/tick", 0) == 0) {
                std::istringstream in(line.substr(5));
                double dt = 0.25;
                in >> dt;
                app.tick(dt);
                print_state(app.snapshot());
                continue;
            }

            std::string event_text = line;
            if (line.rfind("/step", 0) == 0) {
                event_text = trim(line.substr(5));
            }

            const LlmDecision decision = app.run_step(event_text);
            std::cout << "state " << decision.current_state << " -> " << decision.next_state
                      << " | mode=" << decision.authoring_mode << "\n"
                      << "title: " << decision.update_title << "\n"
                      << decision.update_text << "\n\n";
        }

        app.save_state();
        std::cout << "saved binary state: " << config.state_file << "\n";
        app.save_markdown();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
