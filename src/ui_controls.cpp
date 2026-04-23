#include "neuro/ui.hpp"

#include "neuro/app.hpp"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

namespace neuro {

namespace {

void render_section_caption(const char * label, const char * hint = nullptr) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.82f, 0.88f, 0.95f, 1.0f));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    if (hint != nullptr && hint[0] != '\0') {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.58f, 0.66f, 0.76f, 1.0f));
        ImGui::TextWrapped("%s", hint);
        ImGui::PopStyleColor();
    }
    ImGui::Separator();
}

} // namespace

bool render_controls_panel(AppController & app, AsyncStepRunner & runner, GuiState & gui) {
    ImGui::Begin("Run Controls", nullptr);
    if (ImGui::IsWindowCollapsed()) {
        ImGui::End();
        return true;
    }
    render_section_caption("Event Prompt", "Describe the next operation or scene update for the model.");
    ImGui::InputTextMultiline("Event", &gui.event_input, ImVec2(-FLT_MIN, 68.0f));

    if (runner.busy()) {
        ImGui::TextColored(ImVec4(0.60f, 0.82f, 0.98f, 1.0f), "LLM step in progress...");
    }
    if (gui.auto_run) {
        ImGui::TextColored(ImVec4(0.54f, 0.88f, 0.75f, 1.0f), "Auto-run active");
    }

    ImGui::BeginDisabled(runner.busy());
    if (ImGui::Button("Step")) {
        if (runner.start(app, gui.event_input)) {
            gui.status_message = "Started LLM step.";
            gui.error_message.clear();
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    const bool stop_button_active = gui.auto_run;
    if (stop_button_active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.50f, 0.22f, 0.22f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.62f, 0.28f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.40f, 0.17f, 0.17f, 1.0f));
    }
    if (ImGui::Button(gui.auto_run ? "Stop" : "Play")) {
        if (gui.auto_run) {
            gui.auto_run = false;
            gui.status_message = runner.busy() ? "Stopping after current step." : "Auto-run stopped.";
        } else {
            gui.auto_run = true;
            gui.status_message = runner.busy() ? "Auto-run armed." : "Auto-run started.";
            gui.error_message.clear();
        }
    }
    if (stop_button_active) {
        ImGui::PopStyleColor(3);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::BeginDisabled(runner.busy());
    ImGui::InputFloat("Tick Seconds", &gui.tick_seconds, 0.05f, 0.25f, "%.2f");
    ImGui::SameLine();
    if (ImGui::Button("Tick")) {
        try {
            app.tick(gui.tick_seconds);
            gui.cached_state = app.snapshot();
            gui.status_message = "Advanced chemistry.";
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::EndDisabled();

    render_section_caption("Knowledge Processing", "Run selected knowledge-node modes against the current state.");
    ImGui::BeginDisabled(runner.busy());
    ImGui::InputText("Knowledge Modes", &gui.knowledge_modes);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::Button("Process Knowledge", ImVec2(-FLT_MIN, 0.0f))) {
        try {
            const KnowledgeNodeRun run = app.process_knowledge_modes(parse_knowledge_modes(gui.knowledge_modes));
            gui.cached_state = app.snapshot();
            gui.status_message = run.summary;
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::EndDisabled();

    render_section_caption("Performance Tuning", "Adjust stage budgets to minimize iteration time under local runtime limits.");
    ImGui::BeginDisabled(runner.busy());
    ImGui::SetNextItemWidth(110.0f);
    ImGui::InputInt("Plan Tokens", &gui.runtime_tuning.plan_max_tokens);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.0f);
    ImGui::InputInt("Plan Attempts", &gui.runtime_tuning.plan_max_attempts);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Plan Budget ms", &gui.runtime_tuning.plan_max_elapsed_ms);
    ImGui::SetNextItemWidth(110.0f);
    ImGui::InputInt("Write Tokens", &gui.runtime_tuning.write_max_tokens);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.0f);
    ImGui::InputInt("Write Attempts", &gui.runtime_tuning.write_max_attempts);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Write Budget ms", &gui.runtime_tuning.write_max_elapsed_ms);
    if (ImGui::Button("Apply Tuning")) {
        try {
            app.set_runtime_tuning(gui.runtime_tuning);
            gui.runtime_tuning = app.runtime_tuning();
            gui.status_message = "Updated runtime tuning.";
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Tuning")) {
        gui.runtime_tuning = app.config().runtime_tuning;
    }
    ImGui::EndDisabled();

    ImGui::Separator();
    if (!gui.status_message.empty()) {
        ImGui::TextWrapped("%s", gui.status_message.c_str());
    }
    if (!gui.error_message.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", gui.error_message.c_str());
    }
    ImGui::End();
    return false;
}

} // namespace neuro
