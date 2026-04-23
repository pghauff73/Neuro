#pragma once

#include "neuro/core.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace neuro {

struct AppConfig {
    std::string model_path;
    std::string topic = "Untitled Topic";
    std::string state_file = "neuro_state.bin";
    int n_predict = 192;
    int n_ctx = 2048;
    int n_gpu_layers = 99;
    bool knowledge_node_process = true;
    bool post_knowledge_node_process = true;
    std::string knowledge_node_modes = "all";
};

class AppController {
public:
    explicit AppController(AppConfig config);
    ~AppController();

    const AppConfig & config() const;
    void load_state();
    std::optional<KnowledgeNodeRun> maybe_process_initial_knowledge();
    void save_state() const;
    void save_markdown(const std::string & prefix = "body-of-work") const;
    EngineState snapshot() const;
    std::vector<std::string> configured_knowledge_modes() const;

    void tick(double dt);
    void add_knowledge_entry(KnowledgeEntry entry);
    KnowledgeNodeRun process_knowledge_modes(const std::vector<std::string> & modes);
    KnowledgeNodeRun process_configured_knowledge_modes();
    LlmDecision run_step(const std::string & input_text);

private:
    AppConfig config_;
    mutable std::mutex mutex_;
    ChemicalEngine engine_;
    std::unique_ptr<class LlamaRuntime> runtime_;
};

} // namespace neuro
