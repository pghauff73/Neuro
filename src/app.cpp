#include "neuro/app.hpp"

#include "neuro/llama_runtime.hpp"

#include <stdexcept>

namespace neuro {
namespace {

std::string trim(const std::string & s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

} // namespace

AppController::AppController(AppConfig config)
    : config_(std::move(config)),
      engine_(config_.topic),
      runtime_(std::make_unique<LlamaRuntime>(config_.model_path, config_.n_predict, config_.n_ctx, config_.n_gpu_layers)) {}

AppController::~AppController() = default;

const AppConfig & AppController::config() const {
    return config_;
}

void AppController::load_state() {
    auto loaded_state = read_engine_state_binary(config_.state_file);
    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded_state) {
        engine_.state() = std::move(*loaded_state);
    }
    if (engine_.state().knowledge_context.empty()) {
        engine_.state().knowledge_context = build_knowledge_context(engine_.state().knowledge_entries, engine_.state().topic, 4);
    }
}

std::optional<KnowledgeNodeRun> AppController::maybe_process_initial_knowledge() {
    if (!config_.knowledge_node_process) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.process_knowledgebase_modes(parse_knowledge_modes(config_.knowledge_node_modes));
}

void AppController::save_state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    write_engine_state_binary(engine_.state(), config_.state_file);
}

void AppController::save_markdown(const std::string & prefix) const {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.persist(prefix);
}

EngineState AppController::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.state();
}

std::vector<std::string> AppController::configured_knowledge_modes() const {
    return parse_knowledge_modes(config_.knowledge_node_modes);
}

void AppController::tick(double dt) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.tick(dt);
}

void AppController::add_knowledge_entry(KnowledgeEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.add_knowledge_entry(std::move(entry));
}

KnowledgeNodeRun AppController::process_knowledge_modes(const std::vector<std::string> & modes) {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.process_knowledgebase_modes(modes);
}

KnowledgeNodeRun AppController::process_configured_knowledge_modes() {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.process_knowledgebase_modes(parse_knowledge_modes(config_.knowledge_node_modes));
}

LlmDecision AppController::run_step(const std::string & input_text) {
    const std::string trimmed = trim(input_text);
    const std::string event_text = trimmed.empty() ? "Continue developing the topic." : trimmed;

    std::lock_guard<std::mutex> lock(mutex_);
    engine_.add_event(event_text);
    engine_.tick(0.25);
    const LlmDecision decision = engine_.step_with_llm(event_text, *runtime_);
    if (config_.post_knowledge_node_process) {
        engine_.process_knowledgebase_modes(parse_knowledge_modes(config_.knowledge_node_modes));
    }
    return decision;
}

} // namespace neuro
