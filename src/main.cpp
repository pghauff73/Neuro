#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kChemCount = 8;
constexpr int kDefaultPredict = 192;
constexpr int kDefaultCtx = 2048;
constexpr int kDefaultGpuLayers = 99;
constexpr double kSimulatedTimeScale = 100.0;
constexpr const char * kSystemPrompt =
    "Return exactly one valid JSON object. "
    "No markdown. No explanation. "
    "Use double quotes for all keys and strings.";

const std::array<std::string, kChemCount> kChemNames = {
    "DA", "5HT", "OXY", "CORT", "NE", "GABA", "ACh", "END"
};

const std::array<std::string, 3> kLayerNames = {
    "synaptic", "modulatory", "endocrine"
};

const std::array<std::array<double, kChemCount>, kChemCount> kCoupling = {{
    {{0.0, -0.2, 0.3, -0.1, 0.4, -0.2, 0.2, 0.4}},
    {{-0.3, 0.0, 0.3, -0.2, -0.3, 0.4, 0.2, 0.2}},
    {{0.3, 0.2, 0.0, -0.5, -0.2, 0.3, 0.1, 0.4}},
    {{-0.4, -0.3, -0.5, 0.0, 0.4, -0.4, -0.3, 0.2}},
    {{0.3, -0.2, -0.1, 0.3, 0.0, -0.4, 0.4, 0.1}},
    {{-0.3, 0.2, 0.2, -0.2, -0.5, 0.0, -0.3, 0.2}},
    {{0.2, 0.1, 0.0, -0.1, 0.2, -0.2, 0.0, 0.0}},
    {{0.5, 0.3, 0.4, -0.1, 0.0, 0.3, 0.0, 0.0}},
}};

const std::unordered_map<std::string, std::unordered_map<std::string, double>> kKeywordWeights = {
    {"DA", {{"reward", 1.2}, {"win", 1.0}, {"achievement", 1.1}, {"motivation", 1.0}, {"drive", 1.0}, {"success", 1.0}, {"goal", 0.8}, {"anticipation", 0.8}, {"best", 0.6}}},
    {"5HT", {{"calm", 1.0}, {"satisfaction", 1.1}, {"contentment", 1.2}, {"stable", 0.9}, {"peace", 0.9}, {"social", 0.6}, {"wellbeing", 1.0}, {"positive", 0.7}}},
    {"OXY", {{"trust", 1.2}, {"bond", 1.0}, {"connection", 1.0}, {"greeting", 0.8}, {"greetings", 0.8}, {"human touch", 1.2}, {"warmth", 1.0}, {"affiliation", 1.0}}},
    {"CORT", {{"stress", 1.2}, {"threat", 1.2}, {"pressure", 1.1}, {"fear", 1.1}, {"danger", 1.2}, {"scarcity", 1.0}, {"hierarchy", 0.7}}},
    {"NE", {{"arousal", 1.1}, {"alert", 1.0}, {"urgent", 1.2}, {"competition", 1.0}, {"best", 0.8}, {"win", 0.8}, {"ready", 0.8}, {"activate", 0.9}}},
    {"GABA", {{"inhibit", 1.1}, {"rest", 1.0}, {"relax", 1.1}, {"settled", 0.9}, {"calm", 0.8}, {"quiet", 0.8}}},
    {"ACh", {{"attention", 1.2}, {"focus", 1.2}, {"learning", 1.0}, {"memory", 1.0}, {"signal", 0.8}, {"detail", 0.8}}},
    {"END", {{"pleasure", 1.2}, {"comfort", 1.0}, {"relief", 1.1}, {"satisfaction", 0.8}, {"enjoy", 1.0}, {"positive", 0.7}}},
};

const std::unordered_map<std::string, std::unordered_map<std::string, double>> kSignalWeights = {
    {"DA", {{"synaptic", 0.20}, {"modulatory", 0.72}, {"endocrine", 0.08}}},
    {"5HT", {{"synaptic", 0.08}, {"modulatory", 0.72}, {"endocrine", 0.20}}},
    {"OXY", {{"synaptic", 0.05}, {"modulatory", 0.55}, {"endocrine", 0.40}}},
    {"CORT", {{"synaptic", 0.00}, {"modulatory", 0.20}, {"endocrine", 0.80}}},
    {"NE", {{"synaptic", 0.45}, {"modulatory", 0.50}, {"endocrine", 0.05}}},
    {"GABA", {{"synaptic", 0.82}, {"modulatory", 0.18}, {"endocrine", 0.00}}},
    {"ACh", {{"synaptic", 0.88}, {"modulatory", 0.12}, {"endocrine", 0.00}}},
    {"END", {{"synaptic", 0.08}, {"modulatory", 0.47}, {"endocrine", 0.45}}},
};

const std::unordered_map<std::string, std::unordered_map<std::string, double>> kReadoutWeights = {
    {"DA", {{"synaptic", 0.15}, {"modulatory", 0.75}, {"endocrine", 0.10}}},
    {"5HT", {{"synaptic", 0.05}, {"modulatory", 0.70}, {"endocrine", 0.25}}},
    {"OXY", {{"synaptic", 0.03}, {"modulatory", 0.52}, {"endocrine", 0.45}}},
    {"CORT", {{"synaptic", 0.00}, {"modulatory", 0.10}, {"endocrine", 0.90}}},
    {"NE", {{"synaptic", 0.35}, {"modulatory", 0.60}, {"endocrine", 0.05}}},
    {"GABA", {{"synaptic", 0.90}, {"modulatory", 0.10}, {"endocrine", 0.00}}},
    {"ACh", {{"synaptic", 0.92}, {"modulatory", 0.08}, {"endocrine", 0.00}}},
    {"END", {{"synaptic", 0.05}, {"modulatory", 0.40}, {"endocrine", 0.55}}},
};

const std::unordered_map<std::string, std::unordered_map<std::string, double>> kHalfLives = {
    {"DA", {{"synaptic", 0.07}, {"modulatory", 1.50}, {"endocrine", 12.0}}},
    {"5HT", {{"synaptic", 0.40}, {"modulatory", 4.00}, {"endocrine", 45.0}}},
    {"OXY", {{"synaptic", 0.80}, {"modulatory", 8.00}, {"endocrine", 90.0}}},
    {"CORT", {{"synaptic", 5.00}, {"modulatory", 45.0}, {"endocrine", 240.0}}},
    {"NE", {{"synaptic", 0.08}, {"modulatory", 1.20}, {"endocrine", 10.0}}},
    {"GABA", {{"synaptic", 0.03}, {"modulatory", 0.35}, {"endocrine", 4.0}}},
    {"ACh", {{"synaptic", 0.03}, {"modulatory", 0.25}, {"endocrine", 3.0}}},
    {"END", {{"synaptic", 1.00}, {"modulatory", 12.0}, {"endocrine", 120.0}}},
};

const std::unordered_map<std::string, double> kLayerCouplingGain = {
    {"synaptic", 0.25}, {"modulatory", 0.08}, {"endocrine", 0.02}
};
const std::unordered_map<std::string, double> kLayerTimeCompression = {
    {"synaptic", 1.00}, {"modulatory", 0.40}, {"endocrine", 0.12}
};
const std::unordered_map<std::string, double> kLayerHomeostaticPull = {
    {"synaptic", 0.35}, {"modulatory", 0.12}, {"endocrine", 0.04}
};
const std::unordered_map<std::string, double> kLayerSoftCap = {
    {"synaptic", 1.00}, {"modulatory", 0.85}, {"endocrine", 0.70}
};

const std::array<std::string, 7> kAuthorModes = {
    "explore", "structure", "connect", "stress_test", "stabilize", "action", "activate"
};

const std::array<std::string, 6> kKnowledgeNodeModes = {
    "explore", "connect", "structure", "stress_test", "stabilize", "action"
};

const std::unordered_map<std::string, std::array<std::string, 2>> kModeStates = {
    {"explore", {"DA", "END"}},
    {"structure", {"ACh", "5HT"}},
    {"connect", {"OXY", "5HT"}},
    {"stress_test", {"CORT", "NE"}},
    {"stabilize", {"GABA", "5HT"}},
    {"action", {"NE", "DA"}},
    {"activate", {"NE", "DA"}},
};

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

    void add_node(SectionNode node) {
        const std::string node_id = node.node_id;
        for (const auto & parent_id : node.parent_ids) {
            auto it = nodes.find(parent_id);
            if (it != nodes.end()) {
                if (std::find(it->second.child_ids.begin(), it->second.child_ids.end(), node_id) == it->second.child_ids.end()) {
                    it->second.child_ids.push_back(node_id);
                }
            }
        }
        insertion_order.push_back(node_id);
        focus_node_id = node_id;
        nodes.emplace(node_id, std::move(node));
    }

    const SectionNode * focused() const {
        auto it = nodes.find(focus_node_id);
        if (it != nodes.end()) {
            return &it->second;
        }
        if (!insertion_order.empty()) {
            auto jt = nodes.find(insertion_order.back());
            if (jt != nodes.end()) {
                return &jt->second;
            }
        }
        return nullptr;
    }

    std::string render_markdown() const {
        std::ostringstream out;
        for (const auto & id : insertion_order) {
            auto it = nodes.find(id);
            if (it == nodes.end()) {
                continue;
            }
            const auto & node = it->second;
            out << "## " << node.title << "\n"
                << "[mode=" << node.authoring_mode << " state=" << node.source_state
                << " id=" << node.node_id << "]\n"
                << node.content << "\n\n";
        }
        return out.str();
    }
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

std::string now_iso_like() {
    using system_clock = std::chrono::system_clock;
    const auto now = system_clock::now();
    const std::time_t tt = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return out.str();
}

double clamp(double x, double lo = 0.0, double hi = 1.0) {
    return std::max(lo, std::min(hi, x));
}

bool is_valid_state(const std::string & s) {
    return std::find(kChemNames.begin(), kChemNames.end(), s) != kChemNames.end();
}

bool is_valid_mode(const std::string & s) {
    return std::find(kAuthorModes.begin(), kAuthorModes.end(), s) != kAuthorModes.end();
}

std::string trim(const std::string & s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::string normalize_whitespace(std::string s) {
    for (char & ch : s) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            ch = ' ';
        }
    }
    std::string out;
    bool last_space = false;
    for (char ch : s) {
        const bool is_space = ch == ' ';
        if (is_space && last_space) {
            continue;
        }
        out.push_back(ch);
        last_space = is_space;
    }
    return trim(out);
}

std::string json_escape(const std::string & s) {
    std::ostringstream out;
    for (char ch : s) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    out << ' ';
                } else {
                    out << ch;
                }
        }
    }
    return out.str();
}

std::string clean_json_text(const std::string & raw) {
    std::string text = trim(raw);
    if (text.rfind("```", 0) == 0) {
        const auto first_nl = text.find('\n');
        if (first_nl != std::string::npos) {
            text = text.substr(first_nl + 1);
        }
        const auto last_ticks = text.rfind("```");
        if (last_ticks != std::string::npos) {
            text = text.substr(0, last_ticks);
        }
        text = trim(text);
    }
    const auto start = text.find('{');
    const auto end = text.rfind('}');
    if (start != std::string::npos && end != std::string::npos && end >= start) {
        text = text.substr(start, end - start + 1);
    }
    return text;
}

std::optional<std::string> extract_json_string(const std::string & json, const std::string & key) {
    const std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"])*)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, re)) {
        return std::nullopt;
    }
    std::string value = match[1].str();
    value = std::regex_replace(value, std::regex("\\\\n"), "\n");
    value = std::regex_replace(value, std::regex("\\\\r"), "");
    value = std::regex_replace(value, std::regex("\\\\t"), "\t");
    value = std::regex_replace(value, std::regex("\\\\\""), "\"");
    value = std::regex_replace(value, std::regex("\\\\\\\\"), "\\");
    return value;
}

LlmDecision parse_decision(const std::string & raw, const EngineState & state) {
    const std::string cleaned = clean_json_text(raw);
    LlmDecision out;
    out.current_state = state.current_state;
    out.authoring_mode = state.authoring_mode;
    out.update_title = "Next section";
    out.update_text = "Continue developing the topic.";
    out.next_state = state.current_state;
    out.rationale = "fallback";

    if (const auto v = extract_json_string(cleaned, "current_state"); v && is_valid_state(*v)) out.current_state = *v;
    if (const auto v = extract_json_string(cleaned, "authoring_mode"); v && is_valid_mode(*v)) out.authoring_mode = *v;
    if (const auto v = extract_json_string(cleaned, "update_title"); v) out.update_title = normalize_whitespace(*v);
    if (const auto v = extract_json_string(cleaned, "update_text"); v) out.update_text = normalize_whitespace(*v);
    if (const auto v = extract_json_string(cleaned, "next_state"); v && is_valid_state(*v)) out.next_state = *v;
    if (const auto v = extract_json_string(cleaned, "rationale"); v) out.rationale = normalize_whitespace(*v);

    if (out.update_title.empty()) out.update_title = "Next section";
    if (out.update_text.empty()) out.update_text = "Continue developing the topic.";
    return out;
}

std::string make_node_id(int iteration) {
    std::ostringstream out;
    out << "node-" << iteration << '-' << now_iso_like();
    return out.str();
}

double half_life_to_lambda(double half_life_seconds) {
    return half_life_seconds <= 0.0 ? 0.0 : std::log(2.0) / half_life_seconds;
}

int chem_index(const std::string & chem) {
    auto it = std::find(kChemNames.begin(), kChemNames.end(), chem);
    if (it == kChemNames.end()) {
        throw std::runtime_error("unknown chem: " + chem);
    }
    return static_cast<int>(std::distance(kChemNames.begin(), it));
}

std::array<double, kChemCount> compute_effective(const std::unordered_map<std::string, std::array<double, kChemCount>> & layers) {
    std::array<double, kChemCount> out{};
    for (int i = 0; i < kChemCount; ++i) {
        double total = 0.0;
        for (const auto & layer_name : kLayerNames) {
            const auto & layer_values = layers.at(layer_name);
            total += kReadoutWeights.at(kChemNames[i]).at(layer_name) * layer_values[i];
        }
        out[i] = clamp(total);
    }
    return out;
}

std::unordered_map<std::string, double> compute_authoring_expression(const std::array<double, kChemCount> & c) {
    const double DA = c[0], HT5 = c[1], OXY = c[2], CORT = c[3], NE = c[4], GABA = c[5], ACH = c[6], END = c[7];
    std::unordered_map<std::string, double> expr = {
        {"explore", std::max(0.0, 0.75 * DA + 0.55 * END + 0.25 * NE - 0.20 * GABA - 0.15 * CORT)},
        {"structure", std::max(0.0, 0.90 * ACH + 0.45 * HT5 + 0.10 * GABA - 0.10 * END)},
        {"connect", std::max(0.0, 0.90 * OXY + 0.45 * HT5 + 0.10 * END - 0.20 * CORT)},
        {"stress_test", std::max(0.0, 0.85 * CORT + 0.75 * NE - 0.20 * OXY - 0.10 * HT5)},
        {"stabilize", std::max(0.0, 0.95 * GABA + 0.45 * HT5 - 0.20 * NE - 0.15 * DA)},
        {"action", std::max(0.0, 0.82 * NE + 0.68 * DA + 0.20 * ACH - 0.18 * GABA)},
        {"activate", std::max(0.0, 0.80 * NE + 0.65 * DA + 0.15 * ACH - 0.20 * GABA)},
    };
    double total = 0.0;
    for (const auto & kv : expr) {
        total += kv.second;
    }
    if (total > 0.0) {
        for (auto & kv : expr) {
            kv.second = kv.second / total;
        }
    }
    return expr;
}

std::string choose_mode(const std::unordered_map<std::string, double> & expr, const std::string & fallback) {
    double best = -1.0;
    std::string best_mode = fallback;
    for (const auto & kv : expr) {
        if (kv.second > best) {
            best = kv.second;
            best_mode = kv.first;
        }
    }
    return best_mode;
}

std::string render_chemicals(const std::array<double, kChemCount> & chem_values) {
    std::ostringstream out;
    for (int i = 0; i < kChemCount; ++i) {
        if (i) out << ", ";
        out << kChemNames[i] << '=' << std::fixed << std::setprecision(2) << chem_values[i];
    }
    return out.str();
}

std::string build_prompt(const EngineState & state, const std::string & user_event) {
    const SectionNode * focused = state.graph.focused();
    std::ostringstream runtime;
    runtime << "{"
            << "\"topic\":\"" << json_escape(state.topic) << "\"," 
            << "\"iteration\":" << state.iteration << ','
            << "\"current_state\":\"" << state.current_state << "\"," 
            << "\"next_state\":\"" << state.next_state << "\"," 
            << "\"authoring_mode\":\"" << state.authoring_mode << "\"," 
            << "\"user_event\":\"" << json_escape(user_event) << "\"," 
            << "\"chem\":\"" << json_escape(render_chemicals(state.chem_values)) << "\"";
    if (focused) {
        runtime << ",\"focused_title\":\"" << json_escape(focused->title) << "\"," 
                << "\"focused_content\":\"" << json_escape(focused->content.substr(0, 480)) << "\"";
    }
    if (!state.knowledge_context.empty()) {
        runtime << ",\"knowledge_context\":\"" << json_escape(state.knowledge_context.substr(0, 1800)) << "\"";
    }
    runtime << "}";

    std::ostringstream prompt;
    prompt
        << "Return exactly one compact JSON object with keys "
        << "current_state, authoring_mode, update_title, update_text, next_state, rationale. "
        << "Keep update_title under 8 words and update_text under 90 words. "
        << "Use only states [DA,5HT,OXY,CORT,NE,GABA,ACh,END] and modes [explore,structure,connect,stress_test,stabilize,action,activate]. "
        << "Drive a body-of-work engine forward by creating the next section node. "
        << "When knowledge_context is present, ground the next section in it without inventing citations.\n"
        << "Runtime: " << runtime.str();
    return prompt.str();
}

struct KnowledgeNodeRun {
    std::string topic;
    int entry_count = 0;
    std::vector<std::pair<std::string, int>> processed_by_mode;
    std::string summary;
    std::unordered_map<std::string, double> chemical_adjustments;
};

std::string entry_excerpt(const KnowledgeEntry & entry, size_t max_chars = 900) {
    std::string text = normalize_whitespace(!entry.stabilized_summary.empty() ? entry.stabilized_summary :
        (!entry.mode_summary.empty() ? entry.mode_summary :
        (!entry.summary.empty() ? entry.summary : entry.text)));
    if (text.size() > max_chars) {
        text = text.substr(0, max_chars - 3);
        text = trim(text) + "...";
    }
    return text;
}

std::vector<std::string> tokenize_terms(const std::string & text) {
    std::vector<std::string> tokens;
    std::string token;
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            token.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        } else if (token.size() >= 3) {
            tokens.push_back(token);
            token.clear();
        } else {
            token.clear();
        }
    }
    if (token.size() >= 3) {
        tokens.push_back(token);
    }
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    return tokens;
}

int term_overlap(const std::vector<std::string> & a, const std::vector<std::string> & b) {
    int count = 0;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            ++count;
            ++i;
            ++j;
        } else if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return count;
}

void write_u8(std::ostream & out, uint8_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_u32(std::ostream & out, uint32_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_i32(std::ostream & out, int32_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void write_f64(std::ostream & out, double value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

uint8_t read_u8(std::istream & in) {
    uint8_t value = 0;
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) throw std::runtime_error("state file truncated while reading u8");
    return value;
}

uint32_t read_u32(std::istream & in) {
    uint32_t value = 0;
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) throw std::runtime_error("state file truncated while reading u32");
    return value;
}

int32_t read_i32(std::istream & in) {
    int32_t value = 0;
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) throw std::runtime_error("state file truncated while reading i32");
    return value;
}

double read_f64(std::istream & in) {
    double value = 0.0;
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) throw std::runtime_error("state file truncated while reading f64");
    return value;
}

uint32_t checked_size(size_t size, const std::string & label) {
    if (size > 0xffffffffu) {
        throw std::runtime_error(label + " is too large for binary state format");
    }
    return static_cast<uint32_t>(size);
}

void write_string(std::ostream & out, const std::string & value) {
    write_u32(out, checked_size(value.size(), "string"));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string read_string(std::istream & in) {
    const uint32_t size = read_u32(in);
    std::string value(size, '\0');
    if (size > 0) {
        in.read(&value[0], size);
        if (!in) throw std::runtime_error("state file truncated while reading string");
    }
    return value;
}

void write_string_vector(std::ostream & out, const std::vector<std::string> & values) {
    write_u32(out, checked_size(values.size(), "string vector"));
    for (const auto & value : values) {
        write_string(out, value);
    }
}

std::vector<std::string> read_string_vector(std::istream & in) {
    const uint32_t count = read_u32(in);
    std::vector<std::string> values;
    values.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        values.push_back(read_string(in));
    }
    return values;
}

void write_knowledge_entry(std::ostream & out, const KnowledgeEntry & entry) {
    write_string(out, entry.id);
    write_string(out, entry.query);
    write_string(out, entry.title);
    write_string(out, entry.url);
    write_string(out, entry.source_domain);
    write_string(out, entry.summary);
    write_string(out, entry.text);
    write_string(out, entry.mode_summary);
    write_string(out, entry.stabilized_summary);
    write_string(out, entry.processed_at);
    write_string(out, entry.knowledge_node_mode);
    write_u8(out, entry.retrieval_candidate ? 1 : 0);
    write_string_vector(out, entry.dirty_modes);
    write_string(out, entry.connection_hint);
    write_string_vector(out, entry.structure_path);
    write_string_vector(out, entry.stress_flags);
    write_string(out, entry.action_hint);
    write_f64(out, entry.quality_score);
    write_f64(out, entry.trust_score);
    write_f64(out, entry.retrieval_score);
}

KnowledgeEntry read_knowledge_entry(std::istream & in) {
    KnowledgeEntry entry;
    entry.id = read_string(in);
    entry.query = read_string(in);
    entry.title = read_string(in);
    entry.url = read_string(in);
    entry.source_domain = read_string(in);
    entry.summary = read_string(in);
    entry.text = read_string(in);
    entry.mode_summary = read_string(in);
    entry.stabilized_summary = read_string(in);
    entry.processed_at = read_string(in);
    entry.knowledge_node_mode = read_string(in);
    entry.retrieval_candidate = read_u8(in) != 0;
    entry.dirty_modes = read_string_vector(in);
    entry.connection_hint = read_string(in);
    entry.structure_path = read_string_vector(in);
    entry.stress_flags = read_string_vector(in);
    entry.action_hint = read_string(in);
    entry.quality_score = read_f64(in);
    entry.trust_score = read_f64(in);
    entry.retrieval_score = read_f64(in);
    return entry;
}

void write_section_node(std::ostream & out, const SectionNode & node) {
    write_string(out, node.node_id);
    write_string(out, node.title);
    write_string(out, node.content);
    write_string(out, node.authoring_mode);
    write_string(out, node.source_state);
    write_i32(out, node.iteration);
    write_string_vector(out, node.parent_ids);
    write_string_vector(out, node.child_ids);
    write_string_vector(out, node.tags);
    write_string(out, node.rationale);
}

SectionNode read_section_node(std::istream & in) {
    SectionNode node;
    node.node_id = read_string(in);
    node.title = read_string(in);
    node.content = read_string(in);
    node.authoring_mode = read_string(in);
    node.source_state = read_string(in);
    node.iteration = read_i32(in);
    node.parent_ids = read_string_vector(in);
    node.child_ids = read_string_vector(in);
    node.tags = read_string_vector(in);
    node.rationale = read_string(in);
    return node;
}

void write_engine_state_binary(const EngineState & state, const std::string & path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("unable to write state file: " + path);
    }

    const char magic[8] = {'N', 'A', 'I', 'C', 'P', 'P', 'S', 'T'};
    out.write(magic, sizeof(magic));
    write_u32(out, 1);

    write_string(out, state.topic);
    write_i32(out, state.iteration);
    write_string(out, state.current_state);
    write_string(out, state.next_state);
    write_string(out, state.authoring_mode);

    write_u32(out, checked_size(state.authoring_expression.size(), "authoring expression"));
    for (const auto & kv : state.authoring_expression) {
        write_string(out, kv.first);
        write_f64(out, kv.second);
    }

    for (double value : state.chem_values) {
        write_f64(out, value);
    }

    write_u32(out, checked_size(kLayerNames.size(), "neuro layer names"));
    for (const auto & layer : kLayerNames) {
        write_string(out, layer);
        const auto it = state.neuro_layers.find(layer);
        std::array<double, kChemCount> values{};
        if (it != state.neuro_layers.end()) {
            values = it->second;
        }
        for (double value : values) {
            write_f64(out, value);
        }
    }

    write_string_vector(out, state.recent_events);

    write_u32(out, checked_size(state.knowledge_entries.size(), "knowledge entries"));
    for (const auto & entry : state.knowledge_entries) {
        write_knowledge_entry(out, entry);
    }

    write_string(out, state.knowledge_context);
    write_string(out, state.last_update_title);
    write_string(out, state.last_update_text);
    write_string(out, state.last_raw_response);

    write_string(out, state.graph.focus_node_id);
    write_string_vector(out, state.graph.insertion_order);
    write_u32(out, checked_size(state.graph.nodes.size(), "graph nodes"));
    for (const auto & kv : state.graph.nodes) {
        write_string(out, kv.first);
        write_section_node(out, kv.second);
    }

    if (!out) {
        throw std::runtime_error("failed while writing state file: " + path);
    }
}

std::optional<EngineState> read_engine_state_binary(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }

    char magic[8]{};
    in.read(magic, sizeof(magic));
    const char expected[8] = {'N', 'A', 'I', 'C', 'P', 'P', 'S', 'T'};
    if (!in || std::memcmp(magic, expected, sizeof(magic)) != 0) {
        throw std::runtime_error("invalid state file magic: " + path);
    }
    const uint32_t version = read_u32(in);
    if (version != 1) {
        throw std::runtime_error("unsupported state file version: " + std::to_string(version));
    }

    EngineState state;
    state.topic = read_string(in);
    state.iteration = read_i32(in);
    state.current_state = read_string(in);
    state.next_state = read_string(in);
    state.authoring_mode = read_string(in);

    const uint32_t expr_count = read_u32(in);
    for (uint32_t i = 0; i < expr_count; ++i) {
        const std::string key = read_string(in);
        state.authoring_expression[key] = read_f64(in);
    }

    for (double & value : state.chem_values) {
        value = read_f64(in);
    }

    const uint32_t layer_count = read_u32(in);
    for (uint32_t layer_i = 0; layer_i < layer_count; ++layer_i) {
        const std::string layer = read_string(in);
        std::array<double, kChemCount> values{};
        for (double & value : values) {
            value = read_f64(in);
        }
        state.neuro_layers[layer] = values;
    }
    for (const auto & layer : kLayerNames) {
        if (state.neuro_layers.find(layer) == state.neuro_layers.end()) {
            state.neuro_layers[layer].fill(0.0);
        }
    }

    state.recent_events = read_string_vector(in);

    const uint32_t knowledge_count = read_u32(in);
    state.knowledge_entries.reserve(knowledge_count);
    for (uint32_t i = 0; i < knowledge_count; ++i) {
        state.knowledge_entries.push_back(read_knowledge_entry(in));
    }

    state.knowledge_context = read_string(in);
    state.last_update_title = read_string(in);
    state.last_update_text = read_string(in);
    state.last_raw_response = read_string(in);

    state.graph.focus_node_id = read_string(in);
    state.graph.insertion_order = read_string_vector(in);
    const uint32_t node_count = read_u32(in);
    for (uint32_t i = 0; i < node_count; ++i) {
        const std::string key = read_string(in);
        state.graph.nodes.emplace(key, read_section_node(in));
    }

    if (!in.eof()) {
        in.peek();
        if (!in.eof() && !in) {
            throw std::runtime_error("failed while reading state file: " + path);
        }
    }
    return state;
}

std::string knowledge_mode_summary(const KnowledgeEntry & entry, const std::string & topic, const std::string & mode) {
    const std::string title = normalize_whitespace(!entry.title.empty() ? entry.title : (!entry.query.empty() ? entry.query : entry.url));
    const std::string excerpt = entry_excerpt(entry, 520);
    if (mode == "explore") {
        return normalize_whitespace("Explores " + title + " for topic " + topic + ": " + excerpt);
    }
    if (mode == "connect") {
        return normalize_whitespace("Connects source evidence to " + topic + " by linking " + title + " with the active body of work: " + excerpt);
    }
    if (mode == "structure") {
        return normalize_whitespace("Structures this KnowledgeNode as source -> claim -> implication. Source: " + title + ". Claim basis: " + excerpt);
    }
    if (mode == "stress_test") {
        return normalize_whitespace("Stress test flags for " + title + ": keep claims bounded to the retrieved source, verify scope, and avoid unsupported generalization. Evidence: " + excerpt);
    }
    if (mode == "stabilize") {
        return normalize_whitespace("Stabilized summary for " + title + ": " + excerpt);
    }
    if (mode == "action") {
        return normalize_whitespace("Action cue from " + title + ": use this source as grounded evidence when developing " + topic + ".");
    }
    return excerpt;
}

std::vector<KnowledgeEntry> retrieve_knowledge_entries(const std::vector<KnowledgeEntry> & source_entries, const std::string & topic, int max_entries = 4) {
    std::vector<KnowledgeEntry> entries = source_entries;
    const auto topic_terms = tokenize_terms(topic);
    for (auto & entry : entries) {
        const auto entry_terms = tokenize_terms(entry.query + " " + entry.title + " " + entry.summary + " " + entry.mode_summary + " " + entry.stabilized_summary);
        entry.retrieval_score = term_overlap(topic_terms, entry_terms) + 0.6 * entry.quality_score + 0.35 * entry.trust_score;
    }
    entries.erase(std::remove_if(entries.begin(), entries.end(), [](const KnowledgeEntry & entry) {
        return entry.retrieval_score <= 0.0;
    }), entries.end());
    std::sort(entries.begin(), entries.end(), [](const KnowledgeEntry & a, const KnowledgeEntry & b) {
        return a.retrieval_score > b.retrieval_score;
    });
    if (entries.size() > static_cast<size_t>(std::max(0, max_entries))) {
        entries.resize(static_cast<size_t>(std::max(0, max_entries)));
    }
    return entries;
}

std::string build_knowledge_context(const std::vector<KnowledgeEntry> & source_entries, const std::string & topic, int max_entries = 4) {
    const auto entries = retrieve_knowledge_entries(source_entries, topic, max_entries);
    if (entries.empty()) {
        return "";
    }
    std::ostringstream out;
    out << "Knowledge evidence for " << topic << ":\n";
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto & entry = entries[i];
        out << "R" << (i + 1) << ": "
            << (!entry.title.empty() ? entry.title : (!entry.query.empty() ? entry.query : entry.url))
            << " | " << entry.url
            << " | " << entry_excerpt(entry, 500) << "\n";
    }
    return out.str();
}

std::vector<std::string> parse_knowledge_modes(const std::string & raw_modes) {
    const std::string normalized = normalize_whitespace(raw_modes);
    if (normalized.empty() || normalized == "all") {
        return std::vector<std::string>(kKnowledgeNodeModes.begin(), kKnowledgeNodeModes.end());
    }
    std::vector<std::string> modes;
    std::string item;
    std::istringstream in(normalized);
    while (std::getline(in, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            continue;
        }
        if (std::find(kKnowledgeNodeModes.begin(), kKnowledgeNodeModes.end(), item) == kKnowledgeNodeModes.end()) {
            throw std::runtime_error("unsupported knowledge node mode: " + item);
        }
        if (std::find(modes.begin(), modes.end(), item) == modes.end()) {
            modes.push_back(item);
        }
    }
    if (modes.empty()) {
        modes.push_back("explore");
    }
    return modes;
}

class LlamaRuntime {
public:
    LlamaRuntime(std::string model_path, int n_predict, int n_ctx, int n_gpu_layers)
        : model_path_(std::move(model_path)), n_predict_(n_predict), n_ctx_(n_ctx), n_gpu_layers_(n_gpu_layers) {
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers_;
        model_ = llama_model_load_from_file(model_path_.c_str(), model_params);
        if (!model_) {
            throw std::runtime_error("unable to load model: " + model_path_);
        }
        vocab_ = llama_model_get_vocab(model_);
        if (!vocab_) {
            throw std::runtime_error("unable to get vocab");
        }
    }

    ~LlamaRuntime() {
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }

    std::string complete_json(const std::string & user_prompt) const {
        const std::string full_prompt = std::string(kSystemPrompt) + "\n\n" + user_prompt;
        std::vector<llama_token> prompt_tokens = tokenize(full_prompt);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = std::max(n_ctx_, static_cast<int>(prompt_tokens.size()) + n_predict_ + 16);
        ctx_params.n_batch = static_cast<int>(prompt_tokens.size());
        ctx_params.no_perf = true;

        llama_context * ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            throw std::runtime_error("unable to create llama context");
        }

        auto sampler_params = llama_sampler_chain_default_params();
        sampler_params.no_perf = true;
        llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
        if (!sampler) {
            llama_free(ctx);
            throw std::runtime_error("unable to create sampler");
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.1f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

        std::string output;
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_model_has_encoder(model_)) {
            if (llama_encode(ctx, batch) != 0) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                throw std::runtime_error("llama_encode failed");
            }
            llama_token decoder_start = llama_model_decoder_start_token(model_);
            if (decoder_start == LLAMA_TOKEN_NULL) {
                decoder_start = llama_vocab_bos(vocab_);
            }
            batch = llama_batch_get_one(&decoder_start, 1);
        }

        int n_pos = 0;
        int n_decode = 0;
        while (n_pos + batch.n_tokens < static_cast<int>(prompt_tokens.size()) + n_predict_) {
            if (llama_decode(ctx, batch) != 0) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                throw std::runtime_error("llama_decode failed");
            }
            n_pos += batch.n_tokens;
            llama_token token = llama_sampler_sample(sampler, ctx, -1);
            if (llama_vocab_is_eog(vocab_, token)) {
                break;
            }
            char piece[256];
            const int n = llama_token_to_piece(vocab_, token, piece, sizeof(piece), 0, true);
            if (n < 0) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                throw std::runtime_error("llama_token_to_piece failed");
            }
            output.append(piece, piece + n);
            batch = llama_batch_get_one(&token, 1);
            ++n_decode;

            if (output.find('}') != std::string::npos && output.find('{') != std::string::npos) {
                // The Python script expects one compact JSON object. Stop once one appears complete.
                const auto cleaned = clean_json_text(output);
                if (!cleaned.empty() && cleaned.front() == '{' && cleaned.back() == '}') {
                    output = cleaned;
                    break;
                }
            }
        }

        llama_sampler_free(sampler);
        llama_free(ctx);

        if (output.empty()) {
            throw std::runtime_error("model returned empty output");
        }
        return clean_json_text(output);
    }

private:
    std::vector<llama_token> tokenize(const std::string & text) const {
        const int n_prompt = -llama_tokenize(vocab_, text.c_str(), text.size(), nullptr, 0, true, true);
        if (n_prompt <= 0) {
            throw std::runtime_error("failed to size tokenization");
        }
        std::vector<llama_token> tokens(n_prompt);
        const int rc = llama_tokenize(vocab_, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true);
        if (rc < 0) {
            throw std::runtime_error("tokenization failed");
        }
        return tokens;
    }

    std::string model_path_;
    int n_predict_;
    int n_ctx_;
    int n_gpu_layers_;
    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
};

class ChemicalEngine {
public:
    explicit ChemicalEngine(std::string topic)
        : state_{} {
        state_.topic = std::move(topic);
        for (const auto & layer : kLayerNames) {
            state_.neuro_layers[layer].fill(0.0);
        }
        state_.chem_values.fill(0.0);
        refresh_mode();
    }

    EngineState & state() { return state_; }
    const EngineState & state() const { return state_; }

    void add_event(const std::string & text) {
        state_.recent_events.push_back(text);
        if (state_.recent_events.size() > 20) {
            state_.recent_events.erase(state_.recent_events.begin());
        }
        apply_keyword_signal(text);
    }

    void add_knowledge_entry(KnowledgeEntry entry) {
        if (entry.id.empty()) {
            entry.id = "mem-" + std::to_string(state_.knowledge_entries.size() + 1);
        }
        if (entry.query.empty()) {
            entry.query = state_.topic;
        }
        if (entry.title.empty()) {
            entry.title = entry.query.empty() ? entry.id : entry.query;
        }
        if (entry.summary.empty()) {
            entry.summary = normalize_whitespace(entry.text).substr(0, 800);
        }
        if (entry.quality_score <= 0.0) {
            entry.quality_score = entry.summary.empty() ? 0.1 : std::min(1.0, static_cast<double>(entry.summary.size()) / 800.0);
        }
        state_.knowledge_entries.push_back(std::move(entry));
        state_.knowledge_context = build_knowledge_context(state_.knowledge_entries, state_.topic, 4);
    }

    void tick(double dt_seconds) {
        const double dt = std::max(0.001, dt_seconds) * kSimulatedTimeScale;
        for (const auto & layer : kLayerNames) {
            auto & values = state_.neuro_layers[layer];
            const double time_compression = kLayerTimeCompression.at(layer);
            const double homeo = kLayerHomeostaticPull.at(layer);
            const double gain = kLayerCouplingGain.at(layer);
            const double cap = kLayerSoftCap.at(layer);

            std::array<double, kChemCount> before = values;
            for (int i = 0; i < kChemCount; ++i) {
                const std::string chem = kChemNames[i];
                const double lambda = half_life_to_lambda(kHalfLives.at(chem).at(layer));
                const double decay = std::exp(-lambda * dt * time_compression);
                values[i] *= decay;

                double coupled = 0.0;
                for (int j = 0; j < kChemCount; ++j) {
                    coupled += before[j] * kCoupling[i][j];
                }
                values[i] = clamp(values[i] + gain * coupled * dt - homeo * values[i] * dt, 0.0, cap);
            }
        }
        state_.chem_values = compute_effective(state_.neuro_layers);
        refresh_mode();
    }

    LlmDecision step_with_llm(const std::string & user_event, const LlamaRuntime & runtime) {
        ++state_.iteration;
        const std::string prompt = build_prompt(state_, user_event);
        state_.last_raw_response = runtime.complete_json(prompt);
        LlmDecision decision = parse_decision(state_.last_raw_response, state_);
        commit(decision);
        return decision;
    }

    void persist(const std::string & prefix = "body-of-work") const {
        std::ofstream md(prefix + ".md");
        md << "# Body of Work\n\n"
           << "**Topic:** " << state_.topic << "\n\n"
           << "**Iteration:** " << state_.iteration << "\n\n"
           << "**Current State:** " << state_.current_state << "\n\n"
           << "**Next State:** " << state_.next_state << "\n\n"
           << "**Authoring Mode:** " << state_.authoring_mode << "\n\n"
           << state_.graph.render_markdown();
    }

    KnowledgeNodeRun process_knowledgebase_modes(const std::vector<std::string> & modes) {
        KnowledgeNodeRun run;
        run.topic = state_.topic.empty() ? "Untitled Topic" : state_.topic;
        std::vector<KnowledgeEntry> & entries = state_.knowledge_entries;
        run.entry_count = static_cast<int>(entries.size());
        if (entries.empty()) {
            run.summary = "No in-memory knowledge entries were available for KnowledgeNode processing.";
            return run;
        }

        const auto topic_terms = tokenize_terms(run.topic);
        for (const auto & mode : modes) {
            int processed = 0;
            for (auto & entry : entries) {
                const auto entry_terms = tokenize_terms(entry.query + " " + entry.title + " " + entry.summary + " " + entry.text);
                const int overlap = term_overlap(topic_terms, entry_terms);
                const bool relevant = overlap > 0 || topic_terms.empty();
                if (!relevant && mode != "explore") {
                    continue;
                }

                const std::string summary = knowledge_mode_summary(entry, run.topic, mode);
                entry.processed_at = now_iso_like();
                entry.knowledge_node_mode = mode;
                entry.mode_summary = summary;
                entry.retrieval_candidate = true;

                if (mode == "explore") {
                    entry.dirty_modes = {"connect", "structure", "stress_test", "stabilize", "action"};
                    run.chemical_adjustments["DA"] += 0.02;
                    run.chemical_adjustments["END"] += 0.01;
                } else if (mode == "connect") {
                    entry.connection_hint = "Relate this source to the active topic and latest body node.";
                    run.chemical_adjustments["OXY"] += 0.02;
                    run.chemical_adjustments["5HT"] += 0.01;
                } else if (mode == "structure") {
                    entry.structure_path = {"source", "claim", "implication"};
                    run.chemical_adjustments["ACh"] += 0.02;
                    run.chemical_adjustments["5HT"] += 0.01;
                } else if (mode == "stress_test") {
                    entry.stress_flags = {"scope_check", "source_check", "claim_boundaries"};
                    run.chemical_adjustments["CORT"] += 0.015;
                    run.chemical_adjustments["NE"] += 0.015;
                } else if (mode == "stabilize") {
                    entry.stabilized_summary = summary;
                    run.chemical_adjustments["GABA"] += 0.02;
                    run.chemical_adjustments["5HT"] += 0.01;
                } else if (mode == "action") {
                    entry.action_hint = "Use as grounded evidence in the next section.";
                    run.chemical_adjustments["NE"] += 0.02;
                    run.chemical_adjustments["DA"] += 0.01;
                }
                ++processed;
            }
            run.processed_by_mode.push_back({mode, processed});
        }

        for (const auto & kv : run.chemical_adjustments) {
            const int idx = chem_index(kv.first);
            for (const auto & layer : kLayerNames) {
                state_.neuro_layers[layer][idx] = clamp(state_.neuro_layers[layer][idx] + kv.second, 0.0, kLayerSoftCap.at(layer));
            }
        }
        state_.chem_values = compute_effective(state_.neuro_layers);
        refresh_mode();
        state_.knowledge_context = build_knowledge_context(state_.knowledge_entries, run.topic, 4);

        std::ostringstream summary;
        summary << "KnowledgeNode processing";
        for (const auto & item : run.processed_by_mode) {
            summary << " " << item.first << "=" << item.second;
        }
        run.summary = summary.str();
        return run;
    }

private:
    void apply_keyword_signal(const std::string & text) {
        const std::string lowered = to_lower(text);
        std::array<double, kChemCount> scores{};
        for (int i = 0; i < kChemCount; ++i) {
            const auto & chem = kChemNames[i];
            const auto & map = kKeywordWeights.at(chem);
            for (const auto & kv : map) {
                if (lowered.find(kv.first) != std::string::npos) {
                    scores[i] += kv.second;
                }
            }
        }
        double max_score = 0.0;
        for (double v : scores) max_score = std::max(max_score, v);
        if (max_score <= 0.0) {
            scores[chem_index(state_.current_state)] = 0.25;
            max_score = 0.25;
        }
        for (double & v : scores) v /= max_score;

        for (int i = 0; i < kChemCount; ++i) {
            const auto & chem = kChemNames[i];
            for (const auto & layer : kLayerNames) {
                const double delta = scores[i] * kSignalWeights.at(chem).at(layer) * 0.14;
                state_.neuro_layers[layer][i] = clamp(state_.neuro_layers[layer][i] + delta, 0.0, kLayerSoftCap.at(layer));
            }
        }
        state_.chem_values = compute_effective(state_.neuro_layers);
        refresh_mode();
    }

    void refresh_mode() {
        state_.authoring_expression = compute_authoring_expression(state_.chem_values);
        state_.authoring_mode = choose_mode(state_.authoring_expression, state_.authoring_mode);
    }

    void commit(const LlmDecision & decision) {
        state_.current_state = decision.current_state;
        state_.next_state = decision.next_state;
        state_.authoring_mode = decision.authoring_mode;
        state_.last_update_title = decision.update_title;
        state_.last_update_text = decision.update_text;

        const std::string node_id = make_node_id(state_.iteration);
        SectionNode node;
        node.node_id = node_id;
        node.title = decision.update_title;
        node.content = decision.update_text;
        node.authoring_mode = decision.authoring_mode;
        node.source_state = decision.current_state;
        node.iteration = state_.iteration;
        node.rationale = decision.rationale;
        node.tags = {decision.authoring_mode, decision.current_state};
        if (!state_.graph.insertion_order.empty()) {
            node.parent_ids.push_back(state_.graph.insertion_order.back());
        }
        state_.graph.add_node(std::move(node));

        // Nudge chemistry toward the new mode and state so the engine remains self-propelled.
        const auto mode_states = kModeStates.at(decision.authoring_mode);
        for (const auto & state_name : mode_states) {
            const int idx = chem_index(state_name);
            for (const auto & layer : kLayerNames) {
                state_.neuro_layers[layer][idx] = clamp(state_.neuro_layers[layer][idx] + 0.05, 0.0, kLayerSoftCap.at(layer));
            }
        }
        state_.chem_values = compute_effective(state_.neuro_layers);
        refresh_mode();
    }

    static std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        return s;
    }

    EngineState state_;
};

struct Args {
    std::string model_path;
    std::string topic = "Untitled Topic";
    std::string state_file = "neuro_state.bin";
    int n_predict = kDefaultPredict;
    int n_ctx = kDefaultCtx;
    int n_gpu_layers = kDefaultGpuLayers;
    bool knowledge_node_process = false;
    bool post_knowledge_node_process = true;
    std::string knowledge_node_modes = "all";
};

void print_usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " -m /path/to/model.gguf [--topic \"topic\"] [--n-predict 192] [--n-ctx 2048] [--ngl 99]\n"
        << "       [--state-file neuro_state.bin] [--knowledge-node-process] [--knowledge-node-modes all|mode,mode]\n"
        << "       [--no-post-knowledge-node-process]\n"
        << "\n"
        << "Interactive commands:\n"
        << "  /step <text>   add an event and run one LLM iteration\n"
        << "  /tick <secs>   decay/couple chemistry without LLM\n"
        << "  /knowledge [modes] process in-memory knowledge entries as KnowledgeNodes\n"
        << "  /knowledge-add <title>|<url>|<summary> add an in-memory knowledge entry\n"
        << "  /show          print engine state\n"
        << "  /save          write body-of-work.md only\n"
        << "  /quit          exit\n";
}

Args parse_args(int argc, char ** argv) {
    Args args;
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
    return args;
}

void print_state(const EngineState & s) {
    std::cout << "\n--- engine state ---\n"
              << "topic: " << s.topic << '\n'
              << "iteration: " << s.iteration << '\n'
              << "current_state: " << s.current_state << '\n'
              << "next_state: " << s.next_state << '\n'
              << "authoring_mode: " << s.authoring_mode << '\n'
              << "chem: " << render_chemicals(s.chem_values) << '\n';
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

} // namespace

int main(int argc, char ** argv) {
    try {
        Args args = parse_args(argc, argv);
        LlamaRuntime runtime(args.model_path, args.n_predict, args.n_ctx, args.n_gpu_layers);
        ChemicalEngine engine(args.topic);
        if (auto loaded_state = read_engine_state_binary(args.state_file)) {
            engine.state() = std::move(*loaded_state);
            std::cout << "Loaded state: " << args.state_file << "\n";
        }
        if (engine.state().knowledge_context.empty()) {
            engine.state().knowledge_context = build_knowledge_context(engine.state().knowledge_entries, engine.state().topic, 4);
        }
        const std::vector<std::string> configured_knowledge_modes = parse_knowledge_modes(args.knowledge_node_modes);
        if (args.knowledge_node_process) {
            const KnowledgeNodeRun run = engine.process_knowledgebase_modes(configured_knowledge_modes);
            std::cout << run.summary << "\n";
        }

        std::cout << "Loaded model: " << args.model_path << "\n";
        std::cout << "Topic: " << engine.state().topic << "\n";
        std::cout << "State file: " << args.state_file << "\n";
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
                print_state(engine.state());
                continue;
            }
            if (line == "/save") {
                engine.persist();
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
                engine.add_knowledge_entry(std::move(entry));
                std::cout << "added in-memory knowledge entry; count=" << engine.state().knowledge_entries.size() << "\n";
                continue;
            }
            if (line.rfind("/knowledge", 0) == 0) {
                std::string raw_modes = trim(line.substr(10));
                if (raw_modes.empty()) {
                    raw_modes = args.knowledge_node_modes;
                }
                const KnowledgeNodeRun run = engine.process_knowledgebase_modes(parse_knowledge_modes(raw_modes));
                std::cout << run.summary << "\n";
                continue;
            }
            if (line.rfind("/tick", 0) == 0) {
                std::istringstream in(line.substr(5));
                double dt = 0.25;
                in >> dt;
                engine.tick(dt);
                print_state(engine.state());
                continue;
            }

            std::string event_text = line;
            if (line.rfind("/step", 0) == 0) {
                event_text = trim(line.substr(5));
            }
            if (event_text.empty()) {
                event_text = "Continue developing the topic.";
            }

            engine.add_event(event_text);
            engine.tick(0.25);
            const LlmDecision decision = engine.step_with_llm(event_text, runtime);
            if (args.post_knowledge_node_process) {
                engine.process_knowledgebase_modes(configured_knowledge_modes);
            }

            std::cout << "state " << decision.current_state << " -> " << decision.next_state
                      << " | mode=" << decision.authoring_mode << "\n"
                      << "title: " << decision.update_title << "\n"
                      << decision.update_text << "\n\n";
        }

        write_engine_state_binary(engine.state(), args.state_file);
        std::cout << "saved binary state: " << args.state_file << "\n";
        engine.persist();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
