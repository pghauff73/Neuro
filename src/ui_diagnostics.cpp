#include "neuro/ui.hpp"

#include "imgui.h"

namespace neuro {

namespace {

std::string join_strings(const std::vector<std::string> & values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += values[i];
    }
    return out;
}

} // namespace

bool render_diagnostics_panel(const EngineState & state, const GuiState & gui, const AsyncStepRunner & runner) {
    ImGui::Begin("Diagnostics", nullptr);
    if (ImGui::IsWindowCollapsed()) {
        ImGui::End();
        return true;
    }
    ImGui::Text("Busy: %s", runner.busy() ? "yes" : "no");
    ImGui::Text("Confidence: %.2f", state.last_confidence);
    ImGui::Text("Insufficient Context: %s", state.last_insufficient_context ? "yes" : "no");
    ImGui::Text("Fallback Streak: %d", state.fallback_step_streak);
    ImGui::Text("Productive Streak: %d", state.productive_step_streak);
    ImGui::Text("Last Step ms: %lld", static_cast<long long>(state.last_step_total_ms));
    if (!state.last_used_evidence_ids.empty()) {
        ImGui::TextWrapped("Evidence Used: %s", join_strings(state.last_used_evidence_ids).c_str());
    }
    if (!state.last_conflicts_detected.empty()) {
        ImGui::TextWrapped("Conflicts: %s", join_strings(state.last_conflicts_detected).c_str());
    }
    if (ImGui::CollapsingHeader("Forecasts", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!state.last_mode_forecasts.empty()) {
            if (ImGui::BeginTable("mode_forecasts", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Mode");
                ImGui::TableSetupColumn("Value");
                ImGui::TableSetupColumn("Ground");
                ImGui::TableSetupColumn("Progress");
                ImGui::TableSetupColumn("Conflict");
                ImGui::TableSetupColumn("Risk");
                ImGui::TableHeadersRow();
                for (const auto & forecast : state.last_mode_forecasts) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    if (forecast.mode == state.authoring_mode) {
                        ImGui::TextColored(ImVec4(0.55f, 0.85f, 1.0f, 1.0f), "%s", forecast.mode.c_str());
                    } else {
                        ImGui::TextUnformatted(forecast.mode.c_str());
                    }
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%.2f", forecast.expected_value);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%.2f", forecast.expected_grounding);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text("%.2f", forecast.expected_progress);
                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%.2f", forecast.expected_conflict);
                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%.2f", forecast.expected_risk);
                }
                ImGui::EndTable();
            }
        } else {
            ImGui::TextDisabled("No mode forecasts available yet.");
        }
    }
    if (ImGui::CollapsingHeader("Last Output", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextWrapped("Last Title: %s", state.last_update_title.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("%s", state.last_update_text.c_str());
    }
    if (ImGui::CollapsingHeader("Timing", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::BeginTable("timing_metrics", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Stage");
            ImGui::TableSetupColumn("Total");
            ImGui::TableSetupColumn("Prompt Tok");
            ImGui::TableSetupColumn("Output Tok");
            ImGui::TableSetupColumn("First Tok");
            ImGui::TableSetupColumn("Timeout");
            ImGui::TableHeadersRow();

            const RuntimeStageMetrics * metrics[2] = {&state.last_plan_metrics, &state.last_write_metrics};
            const char * labels[2] = {"Plan", "Write"};
            for (int i = 0; i < 2; ++i) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(labels[i]);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%lld ms", static_cast<long long>(metrics[i]->total_ms));
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%d", metrics[i]->prompt_tokens);
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%d", metrics[i]->output_tokens);
                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%lld ms", static_cast<long long>(metrics[i]->first_token_ms));
                ImGui::TableSetColumnIndex(5);
                ImGui::TextUnformatted(metrics[i]->timed_out ? "yes" : "no");
            }
            ImGui::EndTable();
        }
        ImGui::Text("Plan tokenize/context/decode: %lld / %lld / %lld ms",
            static_cast<long long>(state.last_plan_metrics.tokenize_ms),
            static_cast<long long>(state.last_plan_metrics.context_reset_ms),
            static_cast<long long>(state.last_plan_metrics.decode_ms));
        ImGui::Text("Write tokenize/context/decode: %lld / %lld / %lld ms",
            static_cast<long long>(state.last_write_metrics.tokenize_ms),
            static_cast<long long>(state.last_write_metrics.context_reset_ms),
            static_cast<long long>(state.last_write_metrics.decode_ms));
    }
    if (ImGui::CollapsingHeader("Model I/O", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!state.last_clean_response.empty()) {
            ImGui::TextUnformatted("Clean Response");
            ImGui::BeginChild("clean_response", ImVec2(0.0f, 110.0f), true);
            ImGui::TextWrapped("%s", state.last_clean_response.c_str());
            ImGui::EndChild();
        }
        if (!state.last_prompt_packet.empty()) {
            ImGui::Separator();
            ImGui::TextUnformatted("Prompt Packet");
            ImGui::BeginChild("prompt_packet", ImVec2(0.0f, 140.0f), true);
            ImGui::TextWrapped("%s", state.last_prompt_packet.c_str());
            ImGui::EndChild();
        }
        ImGui::Separator();
        ImGui::TextUnformatted("Raw Response");
        ImGui::BeginChild("raw_response", ImVec2(0.0f, 170.0f), true);
        ImGui::TextWrapped("%s", state.last_raw_response.c_str());
        ImGui::EndChild();
    }
    if (ImGui::CollapsingHeader("Runtime Log", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!state.runtime_log.empty()) {
            ImGui::BeginChild("runtime_log", ImVec2(0.0f, 180.0f), true);
            for (auto it = state.runtime_log.rbegin(); it != state.runtime_log.rend(); ++it) {
                const bool is_error = it->find(" ERROR ") != std::string::npos;
                const bool is_warn = !is_error && it->find(" WARN ") != std::string::npos;
                if (is_error) {
                    ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.45f, 1.0f), "%s", it->c_str());
                } else if (is_warn) {
                    ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.45f, 1.0f), "%s", it->c_str());
                } else {
                    ImGui::TextWrapped("%s", it->c_str());
                }
            }
            ImGui::EndChild();
        } else {
            ImGui::TextDisabled("Runtime log is empty.");
        }
    }
    if (ImGui::CollapsingHeader("Knowledge Context")) {
        if (!state.knowledge_context.empty()) {
            ImGui::TextWrapped("%s", state.knowledge_context.c_str());
        } else {
            ImGui::TextDisabled("No knowledge context available.");
        }
    }
    if (!gui.error_message.empty() && ImGui::CollapsingHeader("Errors", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", gui.error_message.c_str());
    }
    ImGui::End();
    return false;
}

} // namespace neuro
