#pragma once

#include <array>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace neuro {

constexpr int kChemCount = 8;

extern const std::array<std::string, kChemCount> kChemNames;
extern const std::array<std::string, 3> kLayerNames;
extern const std::array<std::string, 7> kAuthorModes;
extern const std::array<std::string, 6> kKnowledgeNodeModes;

struct KnowledgeEntry {
    std::string id;
    std::string query;
    std::string title;
    std::string url;
    std::string source_domain;
    std::string summary;
    std::string text;
    std::string mode_summary;
    std::string stabilized_summary;
    std::string processed_at;
    std::string knowledge_node_mode;
    bool retrieval_candidate = false;
    std::vector<std::string> dirty_modes;
    std::string connection_hint;
    std::vector<std::string> structure_path;
    std::vector<std::string> stress_flags;
    std::string action_hint;
    double quality_score = 0.0;
    double trust_score = 0.0;
    double retrieval_score = 0.0;
};

struct SectionNode {
    std::string node_id;
    std::string title;
    std::string content;
    std::string authoring_mode;
    std::string source_state;
    int iteration = 0;
    std::vector<std::string> parent_ids;
    std::vector<std::string> child_ids;
    std::vector<std::string> tags;
    std::string rationale;
};

struct BodyGraph {
    std::unordered_map<std::string, SectionNode> nodes;
    std::vector<std::string> insertion_order;
    std::string focus_node_id;

    void add_node(SectionNode node);
    const SectionNode * focused() const;
    std::string render_markdown() const;
};

struct EngineState {
    std::string topic;
    int iteration = 0;
    std::string current_state = "ACh";
    std::string next_state = "ACh";
    std::string authoring_mode = "structure";
    std::unordered_map<std::string, double> authoring_expression;
    std::array<double, kChemCount> chem_values{};
    std::unordered_map<std::string, std::array<double, kChemCount>> neuro_layers;
    std::vector<std::string> recent_events;
    std::vector<KnowledgeEntry> knowledge_entries;
    std::string knowledge_context;
    std::string last_update_title;
    std::string last_update_text;
    std::string last_raw_response;
    BodyGraph graph;
};

struct LlmDecision {
    std::string current_state;
    std::string authoring_mode;
    std::string update_title;
    std::string update_text;
    std::string next_state;
    std::string rationale;
};

struct KnowledgeNodeRun {
    std::string topic;
    int entry_count = 0;
    std::vector<std::pair<std::string, int>> processed_by_mode;
    std::string summary;
    std::unordered_map<std::string, double> chemical_adjustments;
};

std::string render_chemicals(const std::array<double, kChemCount> & chem_values);
std::vector<std::string> parse_knowledge_modes(const std::string & raw_modes);
std::string build_knowledge_context(const std::vector<KnowledgeEntry> & source_entries, const std::string & topic, int max_entries = 4);
void write_engine_state_binary(const EngineState & state, const std::string & path);
std::optional<EngineState> read_engine_state_binary(const std::string & path);

class LlamaRuntime;

class ChemicalEngine {
public:
    explicit ChemicalEngine(std::string topic);

    EngineState & state();
    const EngineState & state() const;

    void add_event(const std::string & text);
    void add_knowledge_entry(KnowledgeEntry entry);
    void tick(double dt_seconds);
    LlmDecision step_with_llm(const std::string & user_event, const LlamaRuntime & runtime);
    void persist(const std::string & prefix = "body-of-work") const;
    KnowledgeNodeRun process_knowledgebase_modes(const std::vector<std::string> & modes);

private:
    void apply_keyword_signal(const std::string & text);
    void refresh_mode();
    void commit(const LlmDecision & decision);
    static std::string to_lower(std::string s);

    EngineState state_;
};

} // namespace neuro
