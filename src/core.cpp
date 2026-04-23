#include "neuro/core.hpp"

#include "neuro/llama_runtime.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace neuro {
namespace {

using json = nlohmann::ordered_json;

constexpr double kSimulatedTimeScale = 100.0;
constexpr size_t kMaxRuntimeLogEntries = 200;
std::mutex g_runtime_log_mutex;
std::string g_runtime_log_path = "neuro_runtime.log";

const std::array<std::array<double, kChemCount>, kChemCount> kCoupling = {{
    {{ 0.00,  0.10,  0.00, -0.35,  0.28, -0.18,  0.12,  0.18 }},
    {{-0.08,  0.00,  0.18, -0.22, -0.16,  0.20,  0.08,  0.22 }},
    {{ 0.00,  0.18,  0.00, -0.28, -0.10,  0.10,  0.00,  0.16 }},
    {{-0.20, -0.16, -0.22,  0.00,  0.32, -0.12,  0.10, -0.18 }},
    {{ 0.20, -0.10, -0.08,  0.28,  0.00, -0.24,  0.24, -0.08 }},
    {{-0.16,  0.18,  0.06, -0.24, -0.30,  0.00, -0.10,  0.10 }},
    {{ 0.10,  0.06,  0.00,  0.08,  0.22, -0.12,  0.00,  0.00 }},
    {{ 0.14,  0.18,  0.12, -0.30, -0.12,  0.14, -0.04,  0.00 }},
}};

struct SignalPattern {
    std::array<double, kChemCount> values{};
};

const std::unordered_map<std::string, SignalPattern> kEventPatterns = {
    {"reward",   {{{0.22, 0.04, 0.00, 0.00, 0.08, 0.00, 0.02, 0.08}}}},
    {"progress", {{{0.18, 0.08, 0.00, 0.00, 0.04, 0.00, 0.08, 0.10}}}},
    {"novelty",  {{{0.14, 0.00, 0.00, 0.00, 0.16, 0.00, 0.10, 0.00}}}},
    {"threat",   {{{-0.06,-0.04,-0.08, 0.24, 0.18, 0.00, 0.04,-0.04}}}},
    {"conflict", {{{0.00,-0.02,-0.04, 0.16, 0.14, 0.00, 0.10,-0.02}}}},
    {"detail",   {{{0.00, 0.04, 0.00, 0.00, 0.06, 0.00, 0.20, 0.00}}}},
    {"social",   {{{0.00, 0.10, 0.22,-0.04, 0.00, 0.02, 0.00, 0.06}}}},
    {"closure",  {{{0.06, 0.14, 0.04,-0.10,-0.08, 0.08, 0.00, 0.20}}}},
    {"rest",     {{{0.00, 0.10, 0.00,-0.06,-0.12, 0.18,-0.04, 0.10}}}},
    {"urgency",  {{{0.04,-0.02, 0.00, 0.06, 0.22,-0.04, 0.08, 0.00}}}},
};

const std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> kEventKeywordGroups = {
    {"reward", {{"reward", 1.0}, {"win", 0.9}, {"achievement", 1.0}, {"success", 0.9}, {"motivation", 0.8}, {"goal", 0.7}}},
    {"progress", {{"progress", 1.0}, {"advance", 0.9}, {"improve", 0.8}, {"resolve", 0.9}, {"solved", 0.9}, {"develop", 0.7}}},
    {"novelty", {{"novel", 1.0}, {"new", 0.7}, {"discover", 0.9}, {"explore", 0.8}, {"unexpected", 0.8}}},
    {"threat", {{"threat", 1.0}, {"danger", 1.0}, {"risk", 0.9}, {"fear", 0.8}, {"pressure", 0.8}, {"failure", 0.9}}},
    {"conflict", {{"conflict", 1.0}, {"contradiction", 0.9}, {"problem", 0.7}, {"issue", 0.7}, {"tension", 0.8}, {"uncertain", 0.8}}},
    {"detail", {{"detail", 1.0}, {"precision", 1.0}, {"focus", 0.9}, {"attention", 0.9}, {"analyze", 0.8}, {"structure", 0.7}, {"revise", 0.8}}},
    {"social", {{"trust", 1.0}, {"connect", 0.9}, {"relationship", 1.0}, {"audience", 0.8}, {"voice", 0.7}, {"social", 0.8}, {"bond", 0.9}}},
    {"closure", {{"complete", 1.0}, {"done", 0.9}, {"finished", 0.9}, {"settled", 0.8}, {"relief", 1.0}, {"resolved", 1.0}}},
    {"rest", {{"rest", 1.0}, {"calm", 0.8}, {"relax", 0.9}, {"quiet", 0.7}, {"pause", 0.8}, {"recover", 0.9}}},
    {"urgency", {{"urgent", 1.0}, {"immediately", 1.0}, {"now", 0.7}, {"deadline", 0.9}, {"activate", 0.8}, {"ready", 0.7}}},
};

const std::unordered_map<std::string, double> kLayerSignalInjection = {
    {"synaptic", 1.00}, {"modulatory", 0.35}, {"endocrine", 0.10}
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
    {"DA",   {{"synaptic", 0.20}, {"modulatory", 6.0},  {"endocrine", 90.0}}},
    {"5HT",  {{"synaptic", 0.80}, {"modulatory", 18.0}, {"endocrine", 180.0}}},
    {"OXY",  {{"synaptic", 1.20}, {"modulatory", 24.0}, {"endocrine", 240.0}}},
    {"CORT", {{"synaptic", 8.00}, {"modulatory", 45.0}, {"endocrine", 360.0}}},
    {"NE",   {{"synaptic", 0.10}, {"modulatory", 4.0},  {"endocrine", 60.0}}},
    {"GABA", {{"synaptic", 0.12}, {"modulatory", 5.0},  {"endocrine", 80.0}}},
    {"ACh",  {{"synaptic", 0.10}, {"modulatory", 3.5},  {"endocrine", 45.0}}},
    {"END",  {{"synaptic", 2.50}, {"modulatory", 20.0}, {"endocrine", 300.0}}},
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
const std::unordered_map<std::string, std::array<double, kChemCount>> kLayerSetPoint = {
    {"synaptic",   {0.16, 0.08, 0.08, 0.03, 0.18, 0.05, 0.08, 0.12}},
    {"modulatory", {0.24, 0.16, 0.16, 0.08, 0.22, 0.10, 0.12, 0.18}},
    {"endocrine",  {0.18, 0.14, 0.14, 0.10, 0.16, 0.10, 0.08, 0.18}},
};
constexpr double kEarlyLifeExploreIterations = 6.0;
const std::unordered_map<std::string, double> kEarlyLifeExploreGain = {
    {"synaptic", 0.55}, {"modulatory", 0.80}, {"endocrine", 0.35}
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

const std::unordered_map<std::string, std::array<double, kChemCount>> kModeSetPointBias = {
    {"explore",     { 0.08, -0.02,  0.00, -0.04,  0.04, -0.03,  0.03,  0.04 }},
    {"structure",   { 0.00,  0.04,  0.00, -0.02, -0.02,  0.03,  0.10,  0.00 }},
    {"connect",     { 0.00,  0.05,  0.10, -0.04, -0.02,  0.02,  0.00,  0.04 }},
    {"stress_test", {-0.03, -0.03, -0.04,  0.10,  0.08, -0.03,  0.03, -0.02 }},
    {"stabilize",   {-0.02,  0.06,  0.02, -0.05, -0.05,  0.10, -0.02,  0.06 }},
    {"action",      { 0.07, -0.02,  0.00,  0.00,  0.08, -0.04,  0.04,  0.00 }},
    {"activate",    { 0.04, -0.02,  0.00,  0.02,  0.11, -0.05,  0.03,  0.00 }},
};

const std::unordered_map<std::string, double> kLayerModeBiasGain = {
    {"synaptic", 0.35}, {"modulatory", 1.00}, {"endocrine", 0.45}
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

void trim_runtime_log(std::vector<std::string> & entries) {
    if (entries.size() > kMaxRuntimeLogEntries) {
        entries.erase(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(entries.size() - kMaxRuntimeLogEntries));
    }
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

std::optional<json> parse_json_object(const std::string & raw) {
    const std::string cleaned = clean_json_text(raw);
    if (cleaned.empty()) {
        return std::nullopt;
    }
    try {
        json parsed = json::parse(cleaned);
        if (!parsed.is_object()) {
            return std::nullopt;
        }
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> extract_json_string(const json & obj, const char * key) {
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_string()) {
        return std::nullopt;
    }
    return it->get<std::string>();
}

std::optional<std::string> extract_json_string_any(const json & obj, std::initializer_list<const char *> keys) {
    for (const char * key : keys) {
        if (auto value = extract_json_string(obj, key)) {
            return value;
        }
    }
    return std::nullopt;
}

std::optional<double> extract_json_number(const json & obj, const char * key) {
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_number()) {
        return std::nullopt;
    }
    return it->get<double>();
}

std::optional<double> extract_json_number_any(const json & obj, std::initializer_list<const char *> keys) {
    for (const char * key : keys) {
        if (auto value = extract_json_number(obj, key)) {
            return value;
        }
    }
    return std::nullopt;
}

std::optional<bool> extract_json_bool(const json & obj, const char * key) {
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_boolean()) {
        return std::nullopt;
    }
    return it->get<bool>();
}

std::optional<bool> extract_json_bool_any(const json & obj, std::initializer_list<const char *> keys) {
    for (const char * key : keys) {
        if (auto value = extract_json_bool(obj, key)) {
            return value;
        }
    }
    return std::nullopt;
}

std::optional<std::vector<std::string>> extract_json_string_array(const json & obj, const char * key) {
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_array()) {
        return std::nullopt;
    }
    std::vector<std::string> out;
    for (const auto & item : *it) {
        if (!item.is_string()) {
            return std::nullopt;
        }
        out.push_back(item.get<std::string>());
    }
    return out;
}

std::optional<std::vector<std::string>> extract_json_string_array_any(const json & obj, std::initializer_list<const char *> keys) {
    for (const char * key : keys) {
        if (auto value = extract_json_string_array(obj, key)) {
            return value;
        }
    }
    return std::nullopt;
}

std::string compact_excerpt(const std::string & text, size_t max_chars = 200) {
    const std::string normalized = normalize_whitespace(text);
    if (normalized.size() <= max_chars) {
        return normalized;
    }
    return normalized.substr(0, max_chars - 3) + "...";
}

std::string format_runtime_log_entry(const std::string & level, const std::string & message) {
    std::ostringstream out;
    out << "[" << now_iso_like() << "] " << level << " " << compact_excerpt(message, 280);
    return out.str();
}

void write_runtime_log_file(const std::string & entry) {
    std::lock_guard<std::mutex> lock(g_runtime_log_mutex);
    std::ofstream out(g_runtime_log_path, std::ios::app);
    if (out) {
        out << entry << '\n';
    }
}

void write_runtime_log_file_block(const std::string & level, const std::string & label, const std::string & payload) {
    std::lock_guard<std::mutex> lock(g_runtime_log_mutex);
    std::ofstream out(g_runtime_log_path, std::ios::app);
    if (!out) {
        return;
    }
    out << "[" << now_iso_like() << "] " << level << " " << label << " BEGIN\n";
    out << payload << '\n';
    out << "[" << now_iso_like() << "] " << level << " " << label << " END\n";
}

std::vector<KnowledgeEntry> retrieve_knowledge_entries(const std::vector<KnowledgeEntry> & source_entries, const std::string & topic, int max_entries);
std::vector<std::string> allowed_evidence_ids(const std::vector<KnowledgeEntry> & entries) {
    std::vector<std::string> ids;
    ids.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto & entry = entries[i];
        ids.push_back(entry.id.empty() ? ("R" + std::to_string(i + 1)) : entry.id);
    }
    return ids;
}

PlanDecision parse_plan_decision(const std::string & raw, const EngineState & state, const std::vector<std::string> & allowed_ids) {
    const std::string cleaned = clean_json_text(raw);
    const auto parsed = parse_json_object(raw);
    PlanDecision out;
    out.current_state = state.current_state;
    out.authoring_mode = state.authoring_mode;
    out.next_state = state.current_state;
    out.rationale = "fallback";
    if (!parsed) {
        out.rationale = raw.empty() ? "fallback: invalid JSON response" : "fallback: invalid JSON response | raw=" + compact_excerpt(raw);
        return out;
    }

    const auto current_state = extract_json_string_any(*parsed, {"current_state", "cs"});
    const auto authoring_mode = extract_json_string_any(*parsed, {"authoring_mode", "am"});
    const auto next_state = extract_json_string_any(*parsed, {"next_state", "ns"});
    const auto rationale = extract_json_string_any(*parsed, {"rationale", "r"});
    const auto confidence = extract_json_number_any(*parsed, {"confidence", "c"});
    const auto used_evidence_ids = extract_json_string_array_any(*parsed, {"used_evidence_ids", "ue"});
    const auto insufficient_context = extract_json_bool_any(*parsed, {"insufficient_context", "ic"});
    const auto conflicts_detected = extract_json_string_array_any(*parsed, {"conflicts_detected", "cd"});

    std::vector<std::string> issues;
    if (current_state && is_valid_state(*current_state)) {
        out.current_state = *current_state;
    } else {
        issues.push_back("current_state");
    }
    if (authoring_mode && is_valid_mode(*authoring_mode)) {
        out.authoring_mode = *authoring_mode;
    } else {
        issues.push_back("authoring_mode");
    }
    if (next_state && is_valid_state(*next_state)) {
        out.next_state = *next_state;
    } else {
        issues.push_back("next_state");
    }
    if (rationale) {
        out.rationale = normalize_whitespace(*rationale);
    }
    if (confidence) {
        out.confidence = clamp(*confidence, 0.0, 1.0);
    }
    if (used_evidence_ids) {
        for (const auto & id : *used_evidence_ids) {
            const std::string normalized = normalize_whitespace(id);
            if (!normalized.empty() &&
                std::find(allowed_ids.begin(), allowed_ids.end(), normalized) != allowed_ids.end()) {
                out.used_evidence_ids.push_back(normalized);
            } else if (!normalized.empty()) {
                issues.push_back("used_evidence_ids");
            }
        }
    }
    if (insufficient_context) {
        out.insufficient_context = *insufficient_context;
    }
    if (conflicts_detected) {
        for (const auto & item : *conflicts_detected) {
            const std::string normalized = normalize_whitespace(item);
            if (!normalized.empty()) {
                out.conflicts_detected.push_back(normalized);
            }
        }
    }

    if (!issues.empty()) {
        std::ostringstream msg;
        msg << "fallback: missing or invalid ";
        for (size_t i = 0; i < issues.size(); ++i) {
            if (i > 0) {
                msg << ", ";
            }
            msg << issues[i];
        }
        if (!cleaned.empty()) {
            msg << " | raw=" << compact_excerpt(cleaned);
        } else if (!raw.empty()) {
            msg << " | raw=" << compact_excerpt(raw);
        }
        out.rationale = msg.str();
    } else {
        if (out.rationale.empty()) {
            out.rationale = "parsed";
        }
        out.accepted = true;
    }
    return out;
}

LlmDecision parse_write_decision(const std::string & raw) {
    const std::string cleaned = clean_json_text(raw);
    const auto parsed = parse_json_object(raw);
    LlmDecision out;
    out.update_title = "Next section";
    out.update_text = "Continue developing the topic.";
    out.rationale = "fallback";
    if (!parsed) {
        out.rationale = raw.empty() ? "fallback: invalid JSON response" : "fallback: invalid JSON response | raw=" + compact_excerpt(raw);
        return out;
    }

    const auto update_title = extract_json_string_any(*parsed, {"update_title", "ut"});
    const auto update_text = extract_json_string_any(*parsed, {"update_text", "ux"});
    const auto rationale = extract_json_string_any(*parsed, {"rationale", "r"});
    std::vector<std::string> issues;
    if (update_title) {
        out.update_title = normalize_whitespace(*update_title);
    } else {
        issues.push_back("update_title");
    }
    if (update_text) {
        out.update_text = normalize_whitespace(*update_text);
    } else {
        issues.push_back("update_text");
    }
    if (rationale) {
        out.rationale = normalize_whitespace(*rationale);
    }
    if (!issues.empty()) {
        std::ostringstream msg;
        msg << "fallback: missing or invalid ";
        for (size_t i = 0; i < issues.size(); ++i) {
            if (i > 0) {
                msg << ", ";
            }
            msg << issues[i];
        }
        if (!cleaned.empty()) {
            msg << " | raw=" << compact_excerpt(cleaned);
        } else if (!raw.empty()) {
            msg << " | raw=" << compact_excerpt(raw);
        }
        out.rationale = msg.str();
    } else {
        if (out.rationale.empty()) {
            out.rationale = "parsed";
        }
        out.accepted = true;
    }
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
        {"explore", std::max(0.0, 0.80 * DA + 0.35 * NE + 0.20 * END - 0.20 * GABA - 0.20 * CORT)},
        {"structure", std::max(0.0, 0.95 * ACH + 0.45 * HT5 + 0.15 * GABA - 0.18 * NE)},
        {"connect", std::max(0.0, 0.85 * OXY + 0.40 * HT5 + 0.20 * END - 0.25 * CORT)},
        {"stress_test", std::max(0.0, 0.90 * CORT + 0.70 * NE - 0.20 * OXY - 0.15 * END)},
        {"stabilize", std::max(0.0, 0.85 * GABA + 0.55 * HT5 + 0.30 * END - 0.25 * NE - 0.15 * DA)},
        {"action", std::max(0.0, 0.70 * NE + 0.65 * DA + 0.25 * ACH - 0.20 * GABA - 0.10 * HT5)},
        {"activate", std::max(0.0, 0.85 * NE + 0.40 * DA + 0.10 * ACH - 0.20 * GABA)},
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

std::unordered_map<std::string, double> compute_mode_viability(const EngineState & state) {
    const double DA = state.chem_values[0];
    const double HT5 = state.chem_values[1];
    const double OXY = state.chem_values[2];
    const double CORT = state.chem_values[3];
    const double NE = state.chem_values[4];
    const double GABA = state.chem_values[5];
    const double ACH = state.chem_values[6];
    const double END = state.chem_values[7];
    const bool grounded = !state.last_used_evidence_ids.empty() || !state.knowledge_entries.empty();
    const bool insufficient = state.last_insufficient_context;
    const bool conflicted = !state.last_conflicts_detected.empty();
    const double confidence = state.last_confidence;
    const double fallback_pressure = std::min(1.0, 0.20 * static_cast<double>(state.fallback_step_streak));
    const double productive_pressure = std::min(1.0, 0.15 * static_cast<double>(state.productive_step_streak));
    const double stagnation_pressure = std::min(1.0, 0.16 * static_cast<double>(state.stagnation_step_streak));
    const double knowledge_body_drive = state.knowledge_entries.empty()
        ? clamp(1.0 - 0.08 * static_cast<double>(state.graph.insertion_order.size()), 0.0, 1.0)
        : 0.0;

    std::unordered_map<std::string, double> viability = {
        {"explore", 1.0},
        {"structure", 1.0},
        {"connect", 1.0},
        {"stress_test", 1.0},
        {"stabilize", 1.0},
        {"action", 1.0},
        {"activate", 1.0},
    };

    if (!grounded) {
        viability["connect"] *= 0.55;
        viability["action"] *= 0.60;
        viability["activate"] *= 0.65;
        viability["explore"] *= 0.85;
    }
    if (insufficient) {
        viability["explore"] *= 0.80;
        viability["connect"] *= 0.70;
        viability["action"] *= 0.68;
        viability["activate"] *= 0.72;
    }
    if (!conflicted) {
        viability["stress_test"] *= 0.55;
    }
    if (confidence < 0.45) {
        viability["explore"] *= 0.85;
        viability["action"] *= 0.70;
        viability["activate"] *= 0.72;
        viability["stress_test"] *= 0.70;
        viability["stabilize"] *= 1.10;
    }
    if (fallback_pressure > 0.0) {
        viability["explore"] *= std::max(0.35, 1.0 - 0.55 * fallback_pressure);
        viability["action"] *= std::max(0.30, 1.0 - 0.75 * fallback_pressure);
        viability["activate"] *= std::max(0.30, 1.0 - 0.80 * fallback_pressure);
        viability["stress_test"] *= std::max(0.35, 1.0 - 0.65 * fallback_pressure);
        viability["stabilize"] *= 1.0 + 0.45 * fallback_pressure;
        viability["structure"] *= 1.0 + 0.20 * fallback_pressure;
    }
    if (productive_pressure > 0.0) {
        viability["explore"] *= 1.0 + 0.18 * productive_pressure;
        viability["connect"] *= 1.0 + 0.22 * productive_pressure;
        viability["action"] *= 1.0 + 0.18 * productive_pressure;
        viability["activate"] *= 1.0 + 0.10 * productive_pressure;
    }
    if (knowledge_body_drive > 0.0) {
        viability["explore"] *= 1.0 + 0.55 * knowledge_body_drive;
        viability["connect"] *= 1.0 + 0.35 * knowledge_body_drive;
        viability["structure"] *= 1.0 + 0.25 * knowledge_body_drive;
        viability["action"] *= 1.0 + 0.18 * knowledge_body_drive;
        viability["stabilize"] *= 1.0 - 0.25 * knowledge_body_drive;
        viability["stress_test"] *= 1.0 - 0.30 * knowledge_body_drive;
    }
    if (stagnation_pressure > 0.0) {
        viability["explore"] *= 1.0 + 0.60 * stagnation_pressure;
        viability["connect"] *= 1.0 + 0.32 * stagnation_pressure;
        viability["action"] *= 1.0 + 0.26 * stagnation_pressure;
        viability["activate"] *= 1.0 + 0.22 * stagnation_pressure;
        viability["structure"] *= 1.0 + 0.12 * stagnation_pressure;
        viability["stabilize"] *= std::max(0.40, 1.0 - 0.45 * stagnation_pressure);
        viability["stress_test"] *= std::max(0.45, 1.0 - 0.40 * stagnation_pressure);
    }
    if (ACH > 0.88 && HT5 < 0.45) {
        viability["structure"] *= 0.68;
    }
    if (CORT > 0.62 && NE > 0.72) {
        viability["stress_test"] *= 0.72;
        viability["action"] *= 0.78;
        viability["activate"] *= 0.72;
        viability["stabilize"] *= 1.10;
    }
    if (GABA > 0.82 && END < 0.28) {
        viability["stabilize"] *= 0.72;
        viability["explore"] *= 0.88;
    }
    if (DA < 0.22 && NE > 0.68) {
        viability["activate"] *= 0.60;
        viability["action"] *= 0.65;
    }
    if (OXY < 0.18) {
        viability["connect"] *= 0.72;
    }

    for (auto & kv : viability) {
        kv.second = clamp(kv.second, 0.15, 1.35);
    }
    return viability;
}

std::vector<ModeForecast> compute_mode_forecasts(const EngineState & state) {
    const double DA = state.chem_values[0];
    const double HT5 = state.chem_values[1];
    const double OXY = state.chem_values[2];
    const double CORT = state.chem_values[3];
    const double NE = state.chem_values[4];
    const double GABA = state.chem_values[5];
    const double ACH = state.chem_values[6];
    const double END = state.chem_values[7];
    const double evidence_support = clamp(0.25 * static_cast<double>(std::min<size_t>(4, state.knowledge_entries.size())));
    const double last_grounding = state.last_used_evidence_ids.empty() ? 0.0 : clamp(0.35 + 0.20 * static_cast<double>(state.last_used_evidence_ids.size()));
    const double grounding = clamp(std::max(evidence_support, last_grounding));
    const double conflict_load = clamp(0.25 * static_cast<double>(state.last_conflicts_detected.size()) + state.chem_values[3] * 0.35);
    const double fallback_pressure = clamp(0.22 * static_cast<double>(state.fallback_step_streak));
    const double productive_pressure = clamp(0.18 * static_cast<double>(state.productive_step_streak));
    const double stagnation_pressure = clamp(0.18 * static_cast<double>(state.stagnation_step_streak));
    const double confidence = clamp(state.last_confidence);
    const double insufficiency = state.last_insufficient_context ? 1.0 : 0.0;
    const double knowledge_body_drive = state.knowledge_entries.empty()
        ? clamp(1.0 - 0.08 * static_cast<double>(state.graph.insertion_order.size()), 0.0, 1.0)
        : 0.0;

    auto make = [&](const std::string & mode, double expected_grounding, double expected_conflict,
                    double expected_progress, double expected_risk) {
        ModeForecast forecast;
        forecast.mode = mode;
        forecast.expected_grounding = clamp(expected_grounding);
        forecast.expected_conflict = clamp(expected_conflict);
        forecast.expected_progress = clamp(expected_progress);
        forecast.expected_risk = clamp(expected_risk);
        forecast.expected_value = clamp(
            0.50 * forecast.expected_progress +
            0.25 * forecast.expected_grounding +
            0.10 * productive_pressure +
            0.10 * confidence -
            0.30 * forecast.expected_risk -
            0.12 * forecast.expected_conflict,
            0.0, 1.0);
        return forecast;
    };

    std::vector<ModeForecast> forecasts;
    forecasts.push_back(make(
        "explore",
        0.45 * grounding + 0.20 * DA,
        0.15 * conflict_load,
        0.55 * DA + 0.20 * NE + 0.15 * END + 0.28 * knowledge_body_drive + 0.30 * stagnation_pressure - 0.25 * fallback_pressure,
        0.18 * CORT + 0.15 * insufficiency + 0.20 * fallback_pressure));
    forecasts.push_back(make(
        "structure",
        0.65 * grounding + 0.15 * HT5,
        0.12 * conflict_load,
        0.50 * ACH + 0.20 * HT5 + 0.15 * grounding + 0.12 * knowledge_body_drive + 0.10 * stagnation_pressure,
        0.18 * (ACH > 0.88 && HT5 < 0.45 ? 1.0 : 0.0) + 0.08 * fallback_pressure));
    forecasts.push_back(make(
        "connect",
        0.75 * grounding + 0.15 * OXY + 0.20 * knowledge_body_drive,
        0.18 * conflict_load,
        0.45 * OXY + 0.20 * HT5 + 0.15 * grounding + 0.18 * knowledge_body_drive + 0.18 * stagnation_pressure,
        0.18 * insufficiency + 0.12 * (OXY < 0.18 ? 1.0 : 0.0)));
    forecasts.push_back(make(
        "stress_test",
        0.55 * grounding,
        0.75 * conflict_load + 0.15 * CORT,
        0.35 * conflict_load + 0.18 * NE,
        0.32 * CORT + 0.18 * fallback_pressure + 0.15 * (!state.last_conflicts_detected.empty() ? 0.0 : 1.0)));
    forecasts.push_back(make(
        "stabilize",
        0.40 * grounding + 0.18 * HT5,
        0.10 * conflict_load,
        0.35 * GABA + 0.30 * HT5 + 0.20 * END + 0.20 * fallback_pressure - 0.18 * stagnation_pressure,
        0.16 * (GABA > 0.82 && END < 0.28 ? 1.0 : 0.0) + 0.08 * stagnation_pressure));
    forecasts.push_back(make(
        "action",
        0.60 * grounding,
        0.20 * conflict_load,
        0.45 * NE + 0.30 * DA + 0.10 * grounding + 0.10 * knowledge_body_drive + 0.14 * stagnation_pressure - 0.20 * fallback_pressure,
        0.20 * insufficiency + 0.15 * fallback_pressure + 0.12 * (DA < 0.22 && NE > 0.68 ? 1.0 : 0.0)));
    forecasts.push_back(make(
        "activate",
        0.45 * grounding,
        0.22 * conflict_load,
        0.50 * NE + 0.20 * DA + 0.12 * stagnation_pressure - 0.18 * fallback_pressure,
        0.22 * insufficiency + 0.18 * fallback_pressure + 0.18 * (DA < 0.22 && NE > 0.68 ? 1.0 : 0.0)));
    return forecasts;
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

void apply_overload(std::array<double, kChemCount> & values) {
    const double DA = values[0];
    const double HT5 = values[1];
    const double CORT = values[3];
    const double NE = values[4];
    const double GABA = values[5];
    const double ACH = values[6];

    const double stress_over = std::max(0.0, CORT - 0.72);
    values[0] = clamp(values[0] - 0.20 * stress_over);
    values[2] = clamp(values[2] - 0.25 * stress_over);
    values[7] = clamp(values[7] - 0.18 * stress_over);

    const double arousal_over = std::max(0.0, NE - 0.78);
    values[2] = clamp(values[2] - 0.18 * arousal_over);
    values[7] = clamp(values[7] - 0.12 * arousal_over);
    values[6] = clamp(values[6] - 0.10 * arousal_over);

    const double impulsive_drive = std::max(0.0, (DA + NE) * 0.5 - (HT5 + GABA) * 0.5 - 0.18);
    values[3] = clamp(values[3] + 0.12 * impulsive_drive);
    values[4] = clamp(values[4] + 0.08 * impulsive_drive);

    const double inhibition_over = std::max(0.0, GABA - 0.82);
    values[0] = clamp(values[0] - 0.16 * inhibition_over);
    values[4] = clamp(values[4] - 0.18 * inhibition_over);
    values[6] = clamp(values[6] - 0.10 * inhibition_over);

    const double brittle_focus = std::max(0.0, ACH - 0.80) * std::max(0.0, 0.55 - HT5);
    values[3] = clamp(values[3] + 0.10 * brittle_focus);
}

std::string entry_excerpt(const KnowledgeEntry & entry, size_t max_chars);

bool is_fallback_like_node(const SectionNode & node) {
    std::string title = node.title;
    std::string rationale = node.rationale;
    std::transform(title.begin(), title.end(), title.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::transform(rationale.begin(), rationale.end(), rationale.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return title == "next section" ||
           title == "stabilizing note" ||
           rationale.find("fallback") != std::string::npos ||
           rationale.find("insufficient_context") != std::string::npos;
}

std::string build_focus_packet(const SectionNode * focused) {
    std::ostringstream focus;
    focus << "{";
    if (focused) {
        const bool fallback_like = is_fallback_like_node(*focused);
        const std::string summary = fallback_like
            ? "Fallback placeholder node. Ignore prior prose and re-anchor from current event and evidence."
            : compact_excerpt(focused->content, 280);
        focus << "\"node_id\":\"" << json_escape(focused->node_id) << "\","
              << "\"title\":\"" << json_escape(normalize_whitespace(focused->title)) << "\","
              << "\"mode\":\"" << json_escape(focused->authoring_mode) << "\","
              << "\"state\":\"" << json_escape(focused->source_state) << "\","
              << "\"summary\":\"" << json_escape(summary) << "\","
              << "\"rationale\":\"" << json_escape(fallback_like ? "" : compact_excerpt(focused->rationale, 180)) << "\"";
    } else {
        focus << "\"node_id\":\"\","
              << "\"title\":\"\","
              << "\"mode\":\"\","
              << "\"state\":\"\","
              << "\"summary\":\"\","
              << "\"rationale\":\"\"";
    }
    focus << "}";
    return focus.str();
}

std::string build_plan_packet(const EngineState & state, const StepContext & context) {
    const SectionNode * focused = state.graph.focused();
    std::ostringstream chemistry;
    chemistry << "{";
    for (int i = 0; i < kChemCount; ++i) {
        if (i > 0) {
            chemistry << ",";
        }
        chemistry << "\"" << kChemNames[i] << "\":" << std::fixed << std::setprecision(3) << state.chem_values[i];
    }
    chemistry << "}";

    std::ostringstream recent_events;
    recent_events << "[";
    const size_t recent_limit = std::min<size_t>(1, context.recent_events.size());
    for (size_t i = 0; i < recent_limit; ++i) {
        if (i > 0) {
            recent_events << ",";
        }
        recent_events << "\"" << json_escape(normalize_whitespace(context.recent_events[i])) << "\"";
    }
    recent_events << "]";

    std::ostringstream evidence;
    evidence << "[";
    const size_t evidence_limit = std::min<size_t>(2, context.evidence_entries.size());
    for (size_t i = 0; i < evidence_limit; ++i) {
        const auto & entry = context.evidence_entries[i];
        if (i > 0) {
            evidence << ",";
        }
        const std::string evidence_id = entry.id.empty() ? ("R" + std::to_string(i + 1)) : entry.id;
        evidence << "{"
                 << "\"id\":\"" << json_escape(evidence_id) << "\","
                 << "\"title\":\"" << json_escape(normalize_whitespace(entry.title.empty() ? entry.query : entry.title)) << "\","
                 << "\"url\":\"" << json_escape(entry.url) << "\","
                 << "\"source_domain\":\"" << json_escape(entry.source_domain) << "\","
                 << "\"summary\":\"" << json_escape(entry_excerpt(entry, 280)) << "\","
                 << "\"retrieval_score\":" << std::fixed << std::setprecision(3) << entry.retrieval_score << ","
                 << "\"quality_score\":" << std::fixed << std::setprecision(3) << entry.quality_score << ","
                 << "\"trust_score\":" << std::fixed << std::setprecision(3) << entry.trust_score
                 << "}";
    }
    evidence << "]";

    std::ostringstream packet;
    packet << "{"
           << "\"task\":{"
           << "\"kind\":\"body_node_decision\","
           << "\"goal\":\"choose the next grounded section update\""
           << "},"
           << "\"state\":{"
           << "\"topic\":\"" << json_escape(state.topic) << "\","
           << "\"iteration\":" << state.iteration << ","
           << "\"current_state\":\"" << state.current_state << "\","
            << "\"next_state\":\"" << state.next_state << "\","
            << "\"authoring_mode\":\"" << state.authoring_mode << "\","
            << "\"chemistry\":" << chemistry.str()
           << "},"
           << "\"focus\":" << build_focus_packet(focused) << ","
           << "\"recent_events\":" << recent_events.str() << ","
           << "\"evidence\":" << evidence.str() << ","
           << "\"constraints\":{"
           << "\"allow_citations\":false,"
            << "\"valid_states\":[\"DA\",\"5HT\",\"OXY\",\"CORT\",\"NE\",\"GABA\",\"ACh\",\"END\"],"
            << "\"valid_modes\":[\"explore\",\"structure\",\"connect\",\"stress_test\",\"stabilize\",\"action\",\"activate\"]"
           << "},"
           << "\"operator_input\":{"
           << "\"user_event\":\"" << json_escape(context.user_event) << "\""
           << "}"
           << "}";
    return packet.str();
}

std::string build_plan_prompt(const std::string & packet, bool compact_json_schema) {
    std::ostringstream prompt;
    if (compact_json_schema) {
        prompt
            << "Return exactly one compact JSON object. "
            << "Use short keys only: "
            << "cs=current_state, am=authoring_mode, ns=next_state, r=rationale. "
            << "Optional: c=confidence, ue=used_evidence_ids, ic=insufficient_context, cd=conflicts_detected. "
            << "Confidence is 0..1. ue and cd are arrays of strings. "
            << "Use only provided evidence ids. "
            << "Do not invent citations or sources. "
            << "If context is insufficient, set ic=true and keep the plan conservative.\n"
            << "Packet: " << packet;
    } else {
        prompt
            << "Return exactly one compact JSON object. "
            << "Required keys: current_state, authoring_mode, next_state, rationale. "
            << "Optional keys: confidence, used_evidence_ids, insufficient_context, conflicts_detected. "
            << "Confidence must be a number from 0 to 1. "
            << "used_evidence_ids and conflicts_detected must be arrays of strings. "
            << "Use only provided evidence ids. "
            << "Do not invent citations or sources. "
            << "If context is insufficient, set insufficient_context=true and keep the plan conservative.\n"
            << "Packet: " << packet;
    }
    return prompt.str();
}

std::string build_probe_grammar(bool compact_json_schema) {
    if (compact_json_schema) {
        return R"GBNF(
root ::= "{" ws "\"p\"" ws ":" ws "\"h\"" ws "," ws "\"s\"" ws ":" ws "\"o\"" ws "}" ws
ws ::= [ \t\n]{0,8}
)GBNF";
    }
    return R"GBNF(
root ::= "{" ws "\"ping\"" ws ":" ws "\"hello\"" ws "," ws "\"status\"" ws ":" ws "\"ok\"" ws "}" ws
ws ::= [ \t\n]{0,8}
)GBNF";
}

std::string build_plan_grammar(bool compact_json_schema) {
    if (compact_json_schema) {
        return R"GBNF(
root ::= "{" ws "\"cs\"" ws ":" ws state ws "," ws "\"am\"" ws ":" ws mode ws "," ws "\"ns\"" ws ":" ws state ws "," ws "\"r\"" ws ":" ws string ws "}" ws
state ::= "\"DA\"" | "\"5HT\"" | "\"OXY\"" | "\"CORT\"" | "\"NE\"" | "\"GABA\"" | "\"ACh\"" | "\"END\""
mode ::= "\"explore\"" | "\"structure\"" | "\"connect\"" | "\"stress_test\"" | "\"stabilize\"" | "\"action\"" | "\"activate\""
string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]{0,8}
)GBNF";
    }
    return R"GBNF(
root ::= "{" ws "\"current_state\"" ws ":" ws state ws "," ws "\"authoring_mode\"" ws ":" ws mode ws "," ws "\"next_state\"" ws ":" ws state ws "," ws "\"rationale\"" ws ":" ws string ws "}" ws
state ::= "\"DA\"" | "\"5HT\"" | "\"OXY\"" | "\"CORT\"" | "\"NE\"" | "\"GABA\"" | "\"ACh\"" | "\"END\""
mode ::= "\"explore\"" | "\"structure\"" | "\"connect\"" | "\"stress_test\"" | "\"stabilize\"" | "\"action\"" | "\"activate\""
string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]{0,8}
)GBNF";
}

std::string build_write_packet(const EngineState & state, const StepContext & context, const StepPlan & plan) {
    const SectionNode * focused = state.graph.focused();
    std::ostringstream evidence;
    evidence << "[";
    bool first = true;
    for (size_t i = 0; i < context.evidence_entries.size(); ++i) {
        const auto & entry = context.evidence_entries[i];
        const std::string evidence_id = entry.id.empty() ? ("R" + std::to_string(i + 1)) : entry.id;
        if (!plan.decision.used_evidence_ids.empty() &&
            std::find(plan.decision.used_evidence_ids.begin(), plan.decision.used_evidence_ids.end(), evidence_id) == plan.decision.used_evidence_ids.end()) {
            continue;
        }
        if (!first) {
            evidence << ",";
        }
        first = false;
        evidence << "{"
                 << "\"id\":\"" << json_escape(evidence_id) << "\","
                 << "\"title\":\"" << json_escape(normalize_whitespace(entry.title.empty() ? entry.query : entry.title)) << "\","
                 << "\"summary\":\"" << json_escape(entry_excerpt(entry, 220)) << "\""
                 << "}";
    }
    evidence << "]";

    std::ostringstream packet;
    packet << "{"
           << "\"task\":{"
           << "\"kind\":\"body_node_write\","
           << "\"goal\":\"write the next grounded section\""
           << "},"
           << "\"approved_plan\":{"
           << "\"current_state\":\"" << json_escape(plan.decision.current_state) << "\","
           << "\"authoring_mode\":\"" << json_escape(plan.decision.authoring_mode) << "\","
           << "\"next_state\":\"" << json_escape(plan.decision.next_state) << "\","
           << "\"confidence\":" << std::fixed << std::setprecision(3) << plan.decision.confidence << ","
           << "\"rationale\":\"" << json_escape(plan.decision.rationale) << "\""
           << "},"
           << "\"focus\":" << build_focus_packet(focused) << ","
           << "\"operator_input\":{"
           << "\"user_event\":\"" << json_escape(context.user_event) << "\""
           << "},"
           << "\"evidence\":" << evidence.str() << ","
           << "\"constraints\":{"
           << "\"max_title_words\":8,"
           << "\"max_text_words\":90,"
           << "\"allow_citations\":false"
           << "}"
           << "}";
    return packet.str();
}

std::string build_write_prompt(const std::string & packet, bool compact_json_schema) {
    std::ostringstream prompt;
    if (compact_json_schema) {
        prompt
            << "Return exactly one compact JSON object. "
            << "Use short keys only: ut=update_title, ux=update_text, r=rationale. "
            << "Keep ut under 8 words and ux under 90 words. "
            << "Do not invent citations or sources. "
            << "Write only the section text described by the approved plan.\n"
            << "Packet: " << packet;
    } else {
        prompt
            << "Return exactly one compact JSON object. "
            << "Required keys: update_title, update_text, rationale. "
            << "Keep update_title under 8 words and update_text under 90 words. "
            << "Do not invent citations or sources. "
            << "Write only the section text described by the approved plan.\n"
            << "Packet: " << packet;
    }
    return prompt.str();
}

std::string build_write_grammar(bool compact_json_schema) {
    if (compact_json_schema) {
        return R"GBNF(
root ::= "{" ws "\"ut\"" ws ":" ws string ws "," ws "\"ux\"" ws ":" ws string ws "," ws "\"r\"" ws ":" ws string ws "}" ws
string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]{0,8}
)GBNF";
    }
    return R"GBNF(
root ::= "{" ws "\"update_title\"" ws ":" ws string ws "," ws "\"update_text\"" ws ":" ws string ws "," ws "\"rationale\"" ws ":" ws string ws "}" ws
string ::= "\"" char* "\"" ws
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
ws ::= [ \t\n]{0,8}
)GBNF";
}

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
    size_t i = 0;
    size_t j = 0;
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

} // namespace

const std::array<std::string, kChemCount> kChemNames = {
    "DA", "5HT", "OXY", "CORT", "NE", "GABA", "ACh", "END"
};

const std::array<std::string, 3> kLayerNames = {
    "synaptic", "modulatory", "endocrine"
};

const std::array<std::string, 7> kAuthorModes = {
    "explore", "structure", "connect", "stress_test", "stabilize", "action", "activate"
};

const std::array<std::string, 6> kKnowledgeNodeModes = {
    "explore", "connect", "structure", "stress_test", "stabilize", "action"
};

void BodyGraph::add_node(SectionNode node) {
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

const SectionNode * BodyGraph::focused() const {
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

std::string BodyGraph::render_markdown() const {
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

std::string render_chemicals(const std::array<double, kChemCount> & chem_values) {
    std::ostringstream out;
    for (int i = 0; i < kChemCount; ++i) {
        if (i) out << ", ";
        out << kChemNames[i] << '=' << std::fixed << std::setprecision(2) << chem_values[i];
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

void write_engine_state_binary(const EngineState & state, const std::string & path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("unable to write state file: " + path);
    }

    const char magic[8] = {'N', 'A', 'I', 'C', 'P', 'P', 'S', 'T'};
    out.write(magic, sizeof(magic));
    write_u32(out, 4);

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
    write_string(out, state.last_prompt_packet);
    write_string(out, state.last_clean_response);
    write_string(out, state.last_raw_response);
    write_f64(out, state.last_confidence);
    write_string_vector(out, state.last_used_evidence_ids);
    write_u8(out, state.last_insufficient_context ? 1 : 0);
    write_string_vector(out, state.last_conflicts_detected);
    write_i32(out, state.fallback_step_streak);
    write_i32(out, state.productive_step_streak);
    write_string_vector(out, state.runtime_log);

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
    if (version != 1 && version != 2 && version != 3 && version != 4) {
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
    if (version >= 2) {
        state.last_prompt_packet = read_string(in);
        state.last_clean_response = read_string(in);
    }
    state.last_raw_response = read_string(in);
    if (version >= 2) {
        state.last_confidence = read_f64(in);
        state.last_used_evidence_ids = read_string_vector(in);
        state.last_insufficient_context = read_u8(in) != 0;
        state.last_conflicts_detected = read_string_vector(in);
        if (version >= 4) {
            state.fallback_step_streak = read_i32(in);
            state.productive_step_streak = read_i32(in);
        }
        if (version >= 3) {
            state.runtime_log = read_string_vector(in);
            trim_runtime_log(state.runtime_log);
        }
    }

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

void set_runtime_log_file(std::string path) {
    const std::string normalized = trim(path);
    std::lock_guard<std::mutex> lock(g_runtime_log_mutex);
    g_runtime_log_path = normalized.empty() ? "neuro_runtime.log" : normalized;
}

void append_runtime_log(EngineState & state, const std::string & level, const std::string & message) {
    const std::string entry = format_runtime_log_entry(level, message);
    state.runtime_log.push_back(entry);
    trim_runtime_log(state.runtime_log);
    std::cerr << entry << '\n';
    write_runtime_log_file(entry);
}

void append_runtime_log_detail(EngineState & state, const std::string & level, const std::string & label, const std::string & payload) {
    append_runtime_log(state, level, label + " bytes=" + std::to_string(payload.size()));
    write_runtime_log_file_block(level, label, payload);
}

std::string build_knowledge_context(const std::vector<KnowledgeEntry> & source_entries, const std::string & topic, int max_entries) {
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

ChemicalEngine::ChemicalEngine(std::string topic)
    : state_{} {
    state_.topic = std::move(topic);
    state_.current_state = "DA";
    state_.next_state = "END";
    state_.authoring_mode = "explore";
    for (const auto & layer : kLayerNames) {
        auto & values = state_.neuro_layers[layer];
        const auto & set_point = kLayerSetPoint.at(layer);
        const auto & explore_bias = kModeSetPointBias.at("explore");
        const double gain = kLayerModeBiasGain.at(layer) * 1.15;
        const double developmental_gain = kEarlyLifeExploreGain.at(layer);
        for (int i = 0; i < kChemCount; ++i) {
            values[i] = clamp(set_point[i] + explore_bias[i] * gain + explore_bias[i] * developmental_gain, 0.0, kLayerSoftCap.at(layer));
        }
    }
    state_.chem_values = compute_effective(state_.neuro_layers);
    refresh_mode();
}

EngineState & ChemicalEngine::state() {
    return state_;
}

const EngineState & ChemicalEngine::state() const {
    return state_;
}

void ChemicalEngine::log_runtime(const std::string & level, const std::string & message) {
    append_runtime_log(state_, level, message);
}

void ChemicalEngine::add_event(const std::string & text) {
    state_.recent_events.push_back(text);
    if (state_.recent_events.size() > 20) {
        state_.recent_events.erase(state_.recent_events.begin());
    }
    apply_keyword_signal(text);
}

void ChemicalEngine::add_knowledge_entry(KnowledgeEntry entry) {
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
    const auto & added = state_.knowledge_entries.back();
    log_runtime("INFO", "knowledge entry added id=" + added.id + " title=" + compact_excerpt(added.title, 80));
}

void ChemicalEngine::tick(double dt_seconds) {
    const double dt = std::max(0.001, dt_seconds) * kSimulatedTimeScale;
    const double stagnation_pressure = std::min(1.0, 0.16 * static_cast<double>(state_.stagnation_step_streak));
    const double knowledge_body_drive = state_.knowledge_entries.empty()
        ? clamp(1.0 - 0.08 * static_cast<double>(state_.graph.insertion_order.size()), 0.0, 1.0)
        : 0.0;
    for (const auto & layer : kLayerNames) {
        auto & values = state_.neuro_layers[layer];
        const double time_compression = kLayerTimeCompression.at(layer);
        const double homeo = kLayerHomeostaticPull.at(layer);
        const double gain = kLayerCouplingGain.at(layer);
        const double cap = kLayerSoftCap.at(layer);
        const auto & set_point = kLayerSetPoint.at(layer);
        const auto & mode_bias = kModeSetPointBias.at(state_.authoring_mode);
        const double mode_bias_gain = kLayerModeBiasGain.at(layer);
        const double developmental_scale = std::max(0.0, 1.0 - static_cast<double>(state_.iteration) / kEarlyLifeExploreIterations);
        const auto & explore_bias = kModeSetPointBias.at("explore");
        const double developmental_gain = kEarlyLifeExploreGain.at(layer) * developmental_scale;

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
            const double target = clamp(set_point[i] + mode_bias[i] * mode_bias_gain + explore_bias[i] * developmental_gain, 0.0, cap);
            const double homeostatic_pull = (target - values[i]) * homeo * dt;
            values[i] = clamp(values[i] + gain * coupled * dt + homeostatic_pull, 0.0, cap);
        }
        if (knowledge_body_drive > 0.0) {
            const double drive_gain = 0.025 * dt * (layer == "modulatory" ? 1.0 : (layer == "endocrine" ? 0.55 : 0.35));
            values[0] = clamp(values[0] + drive_gain * 1.05 * knowledge_body_drive, 0.0, cap); // DA
            values[2] = clamp(values[2] + drive_gain * 0.55 * knowledge_body_drive, 0.0, cap); // OXY
            values[4] = clamp(values[4] + drive_gain * 1.10 * knowledge_body_drive, 0.0, cap); // NE
            values[6] = clamp(values[6] + drive_gain * 0.85 * knowledge_body_drive, 0.0, cap); // ACh
            values[7] = clamp(values[7] + drive_gain * 0.35 * knowledge_body_drive, 0.0, cap); // END
            values[3] = clamp(values[3] - drive_gain * 0.40 * knowledge_body_drive, 0.0, cap); // CORT
            values[5] = clamp(values[5] - drive_gain * 0.45 * knowledge_body_drive, 0.0, cap); // GABA
        }
        if (stagnation_pressure > 0.0) {
            const double growth_gain = 0.018 * dt * (layer == "modulatory" ? 1.0 : (layer == "endocrine" ? 0.45 : 0.30));
            values[0] = clamp(values[0] + growth_gain * 1.00 * stagnation_pressure, 0.0, cap); // DA
            values[2] = clamp(values[2] + growth_gain * 0.50 * stagnation_pressure, 0.0, cap); // OXY
            values[4] = clamp(values[4] + growth_gain * 0.85 * stagnation_pressure, 0.0, cap); // NE
            values[6] = clamp(values[6] + growth_gain * 0.55 * stagnation_pressure, 0.0, cap); // ACh
            values[7] = clamp(values[7] + growth_gain * 0.22 * stagnation_pressure, 0.0, cap); // END
            values[3] = clamp(values[3] - growth_gain * 0.30 * stagnation_pressure, 0.0, cap); // CORT
            values[5] = clamp(values[5] - growth_gain * 0.35 * stagnation_pressure, 0.0, cap); // GABA
        }
        apply_overload(values);
    }
    state_.chem_values = compute_effective(state_.neuro_layers);
    refresh_mode();
}

StepContext ChemicalEngine::prepare_step_context(const std::string & user_event) const {
    StepContext context;
    context.user_event = user_event;

    const size_t recent_count = std::min<size_t>(1, state_.recent_events.size());
    const size_t recent_start = state_.recent_events.size() > recent_count ? state_.recent_events.size() - recent_count : 0;
    context.recent_events.reserve(recent_count);
    for (size_t i = recent_start; i < state_.recent_events.size(); ++i) {
        context.recent_events.push_back(state_.recent_events[i]);
    }

    context.evidence_entries = retrieve_knowledge_entries(state_.knowledge_entries, state_.topic, 2);
    return context;
}

StepPlan ChemicalEngine::plan_step_with_llm(const StepContext & context, const LlamaRuntime & runtime, const RuntimeTuning & tuning) const {
    StepPlan plan;
    plan.context = context;
    plan.plan_packet = build_plan_packet(state_, context);
    const std::string prompt = build_plan_prompt(plan.plan_packet, tuning.compact_json_schema);
    LlamaCompletionMetrics metrics;
    LlamaCompletionOptions options;
    options.max_tokens = tuning.plan_max_tokens;
    options.max_attempts = tuning.plan_max_attempts;
    options.temperature = 0.05f;
    options.retry_temperature = 0.10f;
    options.top_p = 0.90f;
    options.top_k = 24;
    options.seed = 1337;
    options.max_elapsed_ms = tuning.plan_max_elapsed_ms;
    options.prompt_chunk_tokens = tuning.plan_prompt_chunk_tokens;
    options.response_prefix = "{\"";
    options.grammar = build_plan_grammar(tuning.compact_json_schema);
    if (tuning.compact_json_schema) {
        options.temperature = 0.0f;
        options.retry_temperature = 0.0f;
        options.top_p = 1.0f;
        options.top_k = 1;
    }
    append_runtime_log_detail(const_cast<EngineState &>(state_), "INFO", "llm.plan.packet", plan.plan_packet);
    append_runtime_log_detail(const_cast<EngineState &>(state_), "INFO", "llm.plan.prompt", prompt);
    {
        std::ostringstream opts;
        opts << "{"
             << "\"max_tokens\":" << options.max_tokens << ","
             << "\"max_attempts\":" << options.max_attempts << ","
             << "\"temperature\":" << options.temperature << ","
             << "\"retry_temperature\":" << options.retry_temperature << ","
             << "\"top_p\":" << options.top_p << ","
             << "\"top_k\":" << options.top_k << ","
             << "\"seed\":" << options.seed << ","
             << "\"max_elapsed_ms\":" << options.max_elapsed_ms << ","
             << "\"prompt_chunk_tokens\":" << options.prompt_chunk_tokens
             << "}";
        append_runtime_log_detail(const_cast<EngineState &>(state_), "INFO", "llm.plan.options", opts.str());
    }
    try {
        plan.clean_response = runtime.complete_json(prompt, options, &plan.raw_response, &metrics);
    } catch (const std::exception & e) {
        if (plan.raw_response.empty()) {
            plan.clean_response = std::string("{\"rationale\":\"fallback: ") + json_escape(e.what()) + "\"}";
            plan.raw_response = plan.clean_response;
        } else {
            plan.clean_response = plan.raw_response;
        }
    }
    {
        std::ostringstream runtime_metrics;
        runtime_metrics << "{"
                        << "\"attempts_used\":" << metrics.attempts_used << ","
                        << "\"prompt_tokens\":" << metrics.prompt_tokens << ","
                        << "\"output_tokens\":" << metrics.output_tokens << ","
                        << "\"tokenize_ms\":" << metrics.tokenize_ms << ","
                        << "\"context_reset_ms\":" << metrics.context_reset_ms << ","
                        << "\"prompt_decode_ms\":" << metrics.prompt_decode_ms << ","
                        << "\"decode_ms\":" << metrics.decode_ms << ","
                        << "\"first_token_ms\":" << metrics.first_token_ms << ","
                        << "\"total_ms\":" << metrics.total_ms << ","
                        << "\"timed_out\":" << (metrics.timed_out ? "true" : "false") << ","
                        << "\"no_first_token\":" << (metrics.no_first_token ? "true" : "false") << ","
                        << "\"timed_out_stage\":\"" << json_escape(metrics.timed_out_stage) << "\""
                        << "}";
        append_runtime_log_detail(const_cast<EngineState &>(state_), "INFO", "llm.plan.metrics", runtime_metrics.str());
    }
    auto & mutable_state = const_cast<EngineState &>(state_);
    mutable_state.last_plan_metrics.attempts_used = metrics.attempts_used;
    mutable_state.last_plan_metrics.prompt_tokens = metrics.prompt_tokens;
    mutable_state.last_plan_metrics.output_tokens = metrics.output_tokens;
    mutable_state.last_plan_metrics.tokenize_ms = metrics.tokenize_ms;
    mutable_state.last_plan_metrics.context_reset_ms = metrics.context_reset_ms;
    mutable_state.last_plan_metrics.prompt_decode_ms = metrics.prompt_decode_ms;
    mutable_state.last_plan_metrics.decode_ms = metrics.decode_ms;
    mutable_state.last_plan_metrics.first_token_ms = metrics.first_token_ms;
    mutable_state.last_plan_metrics.total_ms = metrics.total_ms;
    mutable_state.last_plan_metrics.timed_out = metrics.timed_out;
    mutable_state.last_plan_metrics.no_first_token = metrics.no_first_token;
    mutable_state.last_plan_metrics.timed_out_stage = metrics.timed_out_stage;
    append_runtime_log_detail(const_cast<EngineState &>(state_), "INFO", "llm.plan.response.raw",
        plan.raw_response.empty() ? plan.clean_response : plan.raw_response);
    append_runtime_log_detail(const_cast<EngineState &>(state_), "INFO", "llm.plan.response.clean", plan.clean_response);
    plan.decision = parse_plan_decision(plan.clean_response, state_, allowed_evidence_ids(context.evidence_entries));
    {
        std::ostringstream parsed;
        parsed << "{"
               << "\"accepted\":" << (plan.decision.accepted ? "true" : "false") << ","
               << "\"fallback_used\":" << (plan.decision.fallback_used ? "true" : "false") << ","
               << "\"current_state\":\"" << json_escape(plan.decision.current_state) << "\","
               << "\"authoring_mode\":\"" << json_escape(plan.decision.authoring_mode) << "\","
               << "\"next_state\":\"" << json_escape(plan.decision.next_state) << "\","
               << "\"confidence\":" << plan.decision.confidence << ","
               << "\"insufficient_context\":" << (plan.decision.insufficient_context ? "true" : "false") << ","
               << "\"rationale\":\"" << json_escape(plan.decision.rationale) << "\""
               << "}";
        append_runtime_log_detail(const_cast<EngineState &>(state_), "INFO", "llm.plan.parsed", parsed.str());
    }
    return plan;
}

StepPlan ChemicalEngine::build_fallback_plan(const StepContext & context, const std::string & reason) const {
    StepPlan plan;
    plan.context = context;
    plan.plan_packet = build_plan_packet(state_, context);
    plan.clean_response = "{\"fallback\":true}";
    plan.raw_response = plan.clean_response;
    plan.decision.current_state = state_.current_state;
    plan.decision.authoring_mode = state_.authoring_mode == "stress_test" ? "stabilize" : state_.authoring_mode;
    plan.decision.next_state = plan.decision.authoring_mode == "stabilize" ? "5HT" : state_.current_state;
    plan.decision.rationale = "fallback plan: " + reason;
    plan.decision.confidence = 0.20;
    plan.decision.insufficient_context = true;
    if (!context.evidence_entries.empty()) {
        plan.decision.used_evidence_ids.push_back(context.evidence_entries.front().id.empty() ? "R1" : context.evidence_entries.front().id);
    }
    plan.decision.accepted = true;
    plan.decision.fallback_used = true;
    return plan;
}

LlmDecision ChemicalEngine::write_step_with_llm(const StepContext & context, const StepPlan & plan, const LlamaRuntime & runtime, const RuntimeTuning & tuning) {
    const std::string packet = build_write_packet(state_, context, plan);
    std::string clean_response;
    std::string raw_response;
    LlamaCompletionMetrics metrics;
    LlamaCompletionOptions options;
    options.max_tokens = tuning.write_max_tokens;
    options.max_attempts = tuning.write_max_attempts;
    options.temperature = 0.08f;
    options.retry_temperature = 0.12f;
    options.top_p = 0.90f;
    options.top_k = 24;
    options.seed = 2337;
    options.max_elapsed_ms = tuning.write_max_elapsed_ms;
    options.prompt_chunk_tokens = tuning.write_prompt_chunk_tokens;
    options.response_prefix = "{\"";
    options.grammar = build_write_grammar(tuning.compact_json_schema);
    if (tuning.compact_json_schema) {
        options.temperature = 0.0f;
        options.retry_temperature = 0.0f;
        options.top_p = 1.0f;
        options.top_k = 1;
    }
    const std::string prompt = build_write_prompt(packet, tuning.compact_json_schema);
    append_runtime_log_detail(state_, "INFO", "llm.write.packet", packet);
    append_runtime_log_detail(state_, "INFO", "llm.write.prompt", prompt);
    {
        std::ostringstream opts;
        opts << "{"
             << "\"max_tokens\":" << options.max_tokens << ","
             << "\"max_attempts\":" << options.max_attempts << ","
             << "\"temperature\":" << options.temperature << ","
             << "\"retry_temperature\":" << options.retry_temperature << ","
             << "\"top_p\":" << options.top_p << ","
             << "\"top_k\":" << options.top_k << ","
             << "\"seed\":" << options.seed << ","
             << "\"max_elapsed_ms\":" << options.max_elapsed_ms << ","
             << "\"prompt_chunk_tokens\":" << options.prompt_chunk_tokens
             << "}";
        append_runtime_log_detail(state_, "INFO", "llm.write.options", opts.str());
    }
    try {
        clean_response = runtime.complete_json(prompt, options, &raw_response, &metrics);
    } catch (const std::exception & e) {
        if (raw_response.empty()) {
            clean_response = std::string("{\"rationale\":\"fallback: ") + json_escape(e.what()) + "\"}";
            raw_response = clean_response;
        } else {
            clean_response = raw_response;
        }
    }
    {
        std::ostringstream runtime_metrics;
        runtime_metrics << "{"
                        << "\"attempts_used\":" << metrics.attempts_used << ","
                        << "\"prompt_tokens\":" << metrics.prompt_tokens << ","
                        << "\"output_tokens\":" << metrics.output_tokens << ","
                        << "\"tokenize_ms\":" << metrics.tokenize_ms << ","
                        << "\"context_reset_ms\":" << metrics.context_reset_ms << ","
                        << "\"prompt_decode_ms\":" << metrics.prompt_decode_ms << ","
                        << "\"decode_ms\":" << metrics.decode_ms << ","
                        << "\"first_token_ms\":" << metrics.first_token_ms << ","
                        << "\"total_ms\":" << metrics.total_ms << ","
                        << "\"timed_out\":" << (metrics.timed_out ? "true" : "false") << ","
                        << "\"no_first_token\":" << (metrics.no_first_token ? "true" : "false") << ","
                        << "\"timed_out_stage\":\"" << json_escape(metrics.timed_out_stage) << "\""
                        << "}";
        append_runtime_log_detail(state_, "INFO", "llm.write.metrics", runtime_metrics.str());
    }
    state_.last_write_metrics.attempts_used = metrics.attempts_used;
    state_.last_write_metrics.prompt_tokens = metrics.prompt_tokens;
    state_.last_write_metrics.output_tokens = metrics.output_tokens;
    state_.last_write_metrics.tokenize_ms = metrics.tokenize_ms;
    state_.last_write_metrics.context_reset_ms = metrics.context_reset_ms;
    state_.last_write_metrics.prompt_decode_ms = metrics.prompt_decode_ms;
    state_.last_write_metrics.decode_ms = metrics.decode_ms;
    state_.last_write_metrics.first_token_ms = metrics.first_token_ms;
    state_.last_write_metrics.total_ms = metrics.total_ms;
    state_.last_write_metrics.timed_out = metrics.timed_out;
    state_.last_write_metrics.no_first_token = metrics.no_first_token;
    state_.last_write_metrics.timed_out_stage = metrics.timed_out_stage;
    append_runtime_log_detail(state_, "INFO", "llm.write.response.raw", raw_response.empty() ? clean_response : raw_response);
    append_runtime_log_detail(state_, "INFO", "llm.write.response.clean", clean_response);

    LlmDecision decision = parse_write_decision(clean_response);
    decision.current_state = plan.decision.current_state;
    decision.authoring_mode = plan.decision.authoring_mode;
    decision.next_state = plan.decision.next_state;
    decision.confidence = plan.decision.confidence;
    decision.used_evidence_ids = plan.decision.used_evidence_ids;
    decision.insufficient_context = plan.decision.insufficient_context;
    decision.conflicts_detected = plan.decision.conflicts_detected;
    if (!raw_response.empty()) {
        decision.rationale = normalize_whitespace(plan.decision.rationale + " | " + decision.rationale);
    } else if (decision.rationale.empty()) {
        decision.rationale = plan.decision.rationale;
    }
    state_.last_prompt_packet = packet;
    state_.last_raw_response = raw_response.empty() ? clean_response : raw_response;
    state_.last_clean_response = clean_response;
    {
        std::ostringstream parsed;
        parsed << "{"
               << "\"accepted\":" << (decision.accepted ? "true" : "false") << ","
               << "\"current_state\":\"" << json_escape(decision.current_state) << "\","
               << "\"authoring_mode\":\"" << json_escape(decision.authoring_mode) << "\","
               << "\"next_state\":\"" << json_escape(decision.next_state) << "\","
               << "\"confidence\":" << decision.confidence << ","
               << "\"insufficient_context\":" << (decision.insufficient_context ? "true" : "false") << ","
               << "\"update_title\":\"" << json_escape(decision.update_title) << "\","
               << "\"rationale\":\"" << json_escape(decision.rationale) << "\""
               << "}";
        append_runtime_log_detail(state_, "INFO", "llm.write.parsed", parsed.str());
    }
    return decision;
}

LlmDecision ChemicalEngine::build_fallback_write(const StepContext & context, const StepPlan & plan, const std::string & reason) const {
    LlmDecision decision;
    decision.current_state = plan.decision.current_state;
    decision.authoring_mode = plan.decision.authoring_mode;
    decision.next_state = plan.decision.next_state;
    decision.confidence = plan.decision.confidence;
    decision.used_evidence_ids = plan.decision.used_evidence_ids;
    decision.insufficient_context = true;
    decision.conflicts_detected = plan.decision.conflicts_detected;
    decision.update_title = plan.decision.authoring_mode == "stabilize" ? "Stabilizing Note" : "Next Section";
    const SectionNode * focused = state_.graph.focused();
    const std::string anchor = focused ? compact_excerpt(focused->content, 110) : state_.topic;
    decision.update_text = normalize_whitespace("Conservatively continue from " + anchor + ". Event: " + context.user_event + ".");
    decision.rationale = normalize_whitespace(plan.decision.rationale + " | fallback write: " + reason);
    decision.accepted = true;
    return decision;
}

LlmDecision ChemicalEngine::step_with_llm(const std::string & user_event, const LlamaRuntime & runtime, const RuntimeTuning & tuning) {
    using steady_clock = std::chrono::steady_clock;
    const auto step_started = steady_clock::now();
    ++state_.iteration;
    log_runtime("INFO", "step start iteration=" + std::to_string(state_.iteration) +
        " mode=" + state_.authoring_mode + " state=" + state_.current_state +
        " event=" + compact_excerpt(user_event, 140));
    const StepContext context = prepare_step_context(user_event);
    log_runtime("INFO", "step context prepared recent_events=" + std::to_string(context.recent_events.size()) +
        " evidence_entries=" + std::to_string(context.evidence_entries.size()));
    const auto plan_started = steady_clock::now();
    StepPlan plan = plan_step_with_llm(context, runtime, tuning);
    const auto plan_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - plan_started).count();

    if (!plan.raw_response.empty()) {
        log_runtime("INFO", "plan response received raw_bytes=" + std::to_string(plan.raw_response.size()) +
            " clean_bytes=" + std::to_string(plan.clean_response.size()) +
            " packet_bytes=" + std::to_string(plan.plan_packet.size()) +
            " elapsed_ms=" + std::to_string(plan_elapsed_ms));
    } else {
        log_runtime("INFO", "plan stage completed packet_bytes=" + std::to_string(plan.plan_packet.size()) +
            " elapsed_ms=" + std::to_string(plan_elapsed_ms));
    }

    if (!plan.decision.accepted) {
        log_runtime("WARN", "plan rejected iteration=" + std::to_string(state_.iteration) +
            " reason=" + plan.decision.rationale);
        plan = build_fallback_plan(context, plan.decision.rationale.empty() ? "invalid plan response" : plan.decision.rationale);
    }

    state_.last_prompt_packet = plan.plan_packet;
    state_.last_raw_response = plan.raw_response.empty() ? plan.clean_response : plan.raw_response;
    state_.last_clean_response = plan.clean_response;

    if (plan.decision.fallback_used) {
        log_runtime("WARN", "using local fallback plan iteration=" + std::to_string(state_.iteration));
    }

    const auto write_started = steady_clock::now();
    LlmDecision decision = write_step_with_llm(context, plan, runtime, tuning);
    const auto write_elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - write_started).count();
    state_.last_step_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - step_started).count();
    log_runtime("INFO", "write stage completed packet_bytes=" + std::to_string(state_.last_prompt_packet.size()) +
        " raw_bytes=" + std::to_string(state_.last_raw_response.size()) +
        " clean_bytes=" + std::to_string(state_.last_clean_response.size()) +
        " elapsed_ms=" + std::to_string(write_elapsed_ms));
    bool write_fallback_used = false;
    if (!decision.accepted) {
        log_runtime("WARN", "write rejected iteration=" + std::to_string(state_.iteration) +
            " reason=" + decision.rationale);
        decision = build_fallback_write(context, plan, decision.rationale.empty() ? "invalid write response" : decision.rationale);
        write_fallback_used = true;
    }

    state_.last_confidence = decision.confidence;
    state_.last_used_evidence_ids = decision.used_evidence_ids;
    state_.last_insufficient_context = decision.insufficient_context;
    state_.last_conflicts_detected = decision.conflicts_detected;

    if (decision.confidence < 0.50) {
        std::ostringstream msg;
        msg << "low-confidence decision confidence=" << std::fixed << std::setprecision(2) << decision.confidence
            << " iteration=" << state_.iteration;
        log_runtime("WARN", msg.str());
    }
    if (decision.insufficient_context) {
        log_runtime("WARN", "decision flagged insufficient_context iteration=" + std::to_string(state_.iteration));
    }
    if (!decision.conflicts_detected.empty()) {
        log_runtime("WARN", "decision conflicts detected count=" + std::to_string(decision.conflicts_detected.size()));
    }

    apply_outcome_feedback(plan, decision, write_fallback_used);
    const bool zero_token_failure =
        state_.last_plan_metrics.output_tokens == 0 ||
        state_.last_write_metrics.output_tokens == 0 ||
        state_.last_plan_metrics.first_token_ms < 0 ||
        state_.last_write_metrics.first_token_ms < 0;
    const bool hard_runtime_failure =
        zero_token_failure ||
        state_.last_plan_metrics.no_first_token ||
        state_.last_write_metrics.no_first_token;
    if (hard_runtime_failure) {
        log_runtime("WARN", "step commit skipped due to hard runtime failure iteration=" +
            std::to_string(state_.iteration) +
            " plan_tokens=" + std::to_string(state_.last_plan_metrics.output_tokens) +
            " write_tokens=" + std::to_string(state_.last_write_metrics.output_tokens) +
            " plan_timeout_stage=" + (state_.last_plan_metrics.timed_out_stage.empty() ? "none" : state_.last_plan_metrics.timed_out_stage) +
            " write_timeout_stage=" + (state_.last_write_metrics.timed_out_stage.empty() ? "none" : state_.last_write_metrics.timed_out_stage));
        return decision;
    }
    commit(decision);
    log_runtime("INFO", "step commit complete iteration=" + std::to_string(state_.iteration) +
        " title=" + compact_excerpt(decision.update_title, 80) +
        " next_mode=" + decision.authoring_mode +
        " next_state=" + decision.next_state);
    return decision;
}

void ChemicalEngine::persist(const std::string & prefix) const {
    std::ofstream md(prefix + ".md");
    md << "# Body of Work\n\n"
       << "**Topic:** " << state_.topic << "\n\n"
       << "**Iteration:** " << state_.iteration << "\n\n"
       << "**Current State:** " << state_.current_state << "\n\n"
       << "**Next State:** " << state_.next_state << "\n\n"
       << "**Authoring Mode:** " << state_.authoring_mode << "\n\n"
       << state_.graph.render_markdown();
}

KnowledgeNodeRun ChemicalEngine::process_knowledgebase_modes(const std::vector<std::string> & modes) {
    KnowledgeNodeRun run;
    run.topic = state_.topic.empty() ? "Untitled Topic" : state_.topic;
    std::vector<KnowledgeEntry> & entries = state_.knowledge_entries;
    run.entry_count = static_cast<int>(entries.size());
    if (entries.empty()) {
        run.summary = "No in-memory knowledge entries were available for KnowledgeNode processing.";
        log_runtime("WARN", "knowledge processing skipped with no entries");
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
    log_runtime("INFO", run.summary);
    return run;
}

void ChemicalEngine::apply_keyword_signal(const std::string & text) {
    const std::string lowered = to_lower(text);
    std::array<double, kChemCount> scores{};
    double total_signal = 0.0;

    for (const auto & group : kEventKeywordGroups) {
        double group_score = 0.0;
        for (const auto & kv : group.second) {
            if (lowered.find(kv.first) != std::string::npos) {
                group_score += kv.second;
            }
        }
        if (group_score <= 0.0) {
            continue;
        }
        total_signal += group_score;
        const auto & pattern = kEventPatterns.at(group.first).values;
        for (int i = 0; i < kChemCount; ++i) {
            scores[i] += pattern[i] * group_score;
        }
    }

    if (total_signal <= 0.0) {
        scores[chem_index(state_.current_state)] += 0.10;
        scores[chem_index(state_.authoring_mode == "structure" ? "ACh" : state_.current_state)] += 0.05;
        total_signal = 1.0;
    }

    for (double & v : scores) {
        v /= total_signal;
    }

    for (int i = 0; i < kChemCount; ++i) {
        for (const auto & layer : kLayerNames) {
            const double delta = scores[i] * kLayerSignalInjection.at(layer) * 0.14;
            state_.neuro_layers[layer][i] = clamp(state_.neuro_layers[layer][i] + delta, 0.0, kLayerSoftCap.at(layer));
        }
    }
    state_.chem_values = compute_effective(state_.neuro_layers);
    refresh_mode();
}

void ChemicalEngine::refresh_mode() {
    state_.authoring_expression = compute_authoring_expression(state_.chem_values);
    const auto viability = compute_mode_viability(state_);
    state_.last_mode_forecasts = compute_mode_forecasts(state_);
    std::unordered_map<std::string, double> forecast_values;
    for (const auto & forecast : state_.last_mode_forecasts) {
        forecast_values[forecast.mode] = forecast.expected_value;
    }
    double total = 0.0;
    for (auto & kv : state_.authoring_expression) {
        const double forecast = forecast_values.count(kv.first) ? forecast_values.at(kv.first) : 0.5;
        const double forecast_gain = 0.55 + 0.90 * forecast;
        kv.second *= viability.at(kv.first) * forecast_gain;
        total += kv.second;
    }
    if (total > 0.0) {
        for (auto & kv : state_.authoring_expression) {
            kv.second /= total;
        }
    }
    state_.authoring_mode = choose_mode(state_.authoring_expression, state_.authoring_mode);
}

void ChemicalEngine::apply_outcome_feedback(const StepPlan & plan, const LlmDecision & decision, bool write_fallback_used) {
    const bool grounded = !decision.used_evidence_ids.empty();
    const bool knowledge_deprived = state_.knowledge_entries.empty();
    const bool repeated_same_mode =
        !state_.graph.insertion_order.empty() &&
        state_.graph.focused() != nullptr &&
        state_.graph.focused()->authoring_mode == state_.authoring_mode;
    const bool productive =
        !plan.decision.fallback_used &&
        !write_fallback_used &&
        decision.accepted &&
        decision.confidence >= 0.55 &&
        !decision.insufficient_context &&
        !decision.update_text.empty() &&
        decision.conflicts_detected.empty() &&
        grounded;
    const bool low_value =
        plan.decision.fallback_used ||
        write_fallback_used ||
        decision.insufficient_context ||
        decision.confidence < 0.45 ||
        !decision.conflicts_detected.empty() ||
        !grounded;
    const bool stagnant =
        !productive &&
        (low_value ||
         (knowledge_deprived && !grounded) ||
         (repeated_same_mode && state_.productive_step_streak == 0));

    if (productive) {
        state_.productive_step_streak += 1;
        state_.fallback_step_streak = 0;
        state_.stagnation_step_streak = std::max(0, state_.stagnation_step_streak - 2);
    } else if (low_value) {
        state_.fallback_step_streak += 1;
        state_.productive_step_streak = 0;
        if (stagnant) {
            state_.stagnation_step_streak += 1;
        }
    } else {
        state_.fallback_step_streak = std::max(0, state_.fallback_step_streak - 1);
        state_.productive_step_streak = std::max(0, state_.productive_step_streak - 1);
        state_.stagnation_step_streak = std::max(0, state_.stagnation_step_streak - 1);
    }

    const double failure_scale = std::min(1.0, 0.35 + 0.20 * state_.fallback_step_streak);
    const double reward_scale = std::min(1.0, 0.30 + 0.15 * state_.productive_step_streak);
    const double stagnation_scale = std::min(1.0, 0.20 + 0.14 * state_.stagnation_step_streak);
    const std::array<double, kChemCount> failure_adjust = {
        -0.035 * failure_scale,  // DA
         0.025 * failure_scale,  // 5HT
        -0.015 * failure_scale,  // OXY
         0.040 * failure_scale,  // CORT
        -0.020 * failure_scale,  // NE
         0.035 * failure_scale,  // GABA
        -0.025 * failure_scale,  // ACh
        -0.020 * failure_scale   // END
    };
    const std::array<double, kChemCount> reward_adjust = {
         0.030 * reward_scale,   // DA
         0.020 * reward_scale,   // 5HT
         0.015 * reward_scale,   // OXY
        -0.025 * reward_scale,   // CORT
         0.015 * reward_scale,   // NE
        -0.010 * reward_scale,   // GABA
         0.015 * reward_scale,   // ACh
         0.030 * reward_scale    // END
    };
    const std::array<double, kChemCount> stagnation_adjust = {
         0.025 * stagnation_scale,   // DA
        -0.008 * stagnation_scale,   // 5HT
         0.014 * stagnation_scale,   // OXY
        -0.016 * stagnation_scale,   // CORT
         0.020 * stagnation_scale,   // NE
        -0.022 * stagnation_scale,   // GABA
         0.018 * stagnation_scale,   // ACh
         0.010 * stagnation_scale    // END
    };

    for (const auto & layer : kLayerNames) {
        auto & values = state_.neuro_layers[layer];
        for (int i = 0; i < kChemCount; ++i) {
            if (productive) {
                values[i] = clamp(values[i] + reward_adjust[i], 0.0, kLayerSoftCap.at(layer));
            } else if (low_value) {
                values[i] = clamp(values[i] + failure_adjust[i], 0.0, kLayerSoftCap.at(layer));
            }
            if (stagnant) {
                values[i] = clamp(values[i] + stagnation_adjust[i], 0.0, kLayerSoftCap.at(layer));
            }
        }
    }

    state_.chem_values = compute_effective(state_.neuro_layers);
    refresh_mode();

    std::ostringstream msg;
    msg << "outcome feedback productive=" << (productive ? "yes" : "no")
        << " low_value=" << (low_value ? "yes" : "no")
        << " grounded=" << (grounded ? "yes" : "no")
        << " plan_fallback=" << (plan.decision.fallback_used ? "yes" : "no")
        << " write_fallback=" << (write_fallback_used ? "yes" : "no")
        << " confidence=" << std::fixed << std::setprecision(2) << decision.confidence
        << " fallback_streak=" << state_.fallback_step_streak
        << " productive_streak=" << state_.productive_step_streak
        << " stagnation_streak=" << state_.stagnation_step_streak
        << " mode_now=" << state_.authoring_mode;
    log_runtime(low_value ? "WARN" : "INFO", msg.str());
}

void ChemicalEngine::commit(const LlmDecision & decision) {
    state_.current_state = decision.current_state;
    state_.next_state = decision.next_state;
    state_.authoring_mode = decision.authoring_mode;
    state_.last_update_title = decision.update_title;
    state_.last_update_text = decision.update_text;

    SectionNode node;
    node.node_id = make_node_id(state_.iteration);
    node.title = decision.update_title;
    node.content = decision.update_text;
    node.authoring_mode = decision.authoring_mode;
    node.source_state = decision.current_state;
    node.iteration = state_.iteration;
    node.rationale = decision.rationale;
    node.tags = {decision.authoring_mode, decision.current_state};
    if (decision.insufficient_context) {
        node.tags.push_back("insufficient_context");
    }
    if (decision.confidence > 0.0) {
        std::ostringstream confidence_tag;
        confidence_tag << "confidence=" << std::fixed << std::setprecision(2) << decision.confidence;
        node.tags.push_back(confidence_tag.str());
    }
    for (const auto & evidence_id : decision.used_evidence_ids) {
        node.tags.push_back("evidence:" + evidence_id);
    }
    for (const auto & conflict : decision.conflicts_detected) {
        node.tags.push_back("conflict:" + conflict);
    }
    if (!state_.graph.insertion_order.empty()) {
        node.parent_ids.push_back(state_.graph.insertion_order.back());
    }
    state_.graph.add_node(std::move(node));

    double reinforcement = 0.05;
    reinforcement *= std::max(0.20, decision.confidence > 0.0 ? decision.confidence : 0.60);
    if (decision.insufficient_context) {
        reinforcement *= 0.50;
    }
    if (!decision.conflicts_detected.empty()) {
        reinforcement *= 0.75;
    }

    const auto mode_states = kModeStates.at(decision.authoring_mode);
    for (const auto & state_name : mode_states) {
        const int idx = chem_index(state_name);
        for (const auto & layer : kLayerNames) {
            state_.neuro_layers[layer][idx] = clamp(state_.neuro_layers[layer][idx] + reinforcement, 0.0, kLayerSoftCap.at(layer));
        }
    }
    state_.chem_values = compute_effective(state_.neuro_layers);
    refresh_mode();
    std::ostringstream msg;
    msg << "node committed id=" << state_.graph.focus_node_id
        << " reinforcement=" << std::fixed << std::setprecision(3) << reinforcement
        << " chem=" << render_chemicals(state_.chem_values);
    log_runtime("INFO", msg.str());
}

std::string ChemicalEngine::to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return s;
}

} // namespace neuro
