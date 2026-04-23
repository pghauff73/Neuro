#include "neuro/ui.hpp"

#include "neuro/app.hpp"

#include "imgui.h"
#include "misc/cpp/imgui_stdlib.h"

namespace neuro {

void render_controls_panel(AppController & app, AsyncStepRunner & runner, GuiState & gui) {
    ImGui::Begin("Controls");
    ImGui::InputTextMultiline("Event", &gui.event_input, ImVec2(-FLT_MIN, 90.0f));

    if (runner.busy()) {
        ImGui::TextUnformatted("LLM step in progress...");
    }

    ImGui::BeginDisabled(runner.busy());
    if (ImGui::Button("Step")) {
        if (runner.start(app, gui.event_input)) {
            gui.status_message = "Started LLM step.";
            gui.error_message.clear();
        }
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("Tick Seconds", &gui.tick_seconds, 0.05f, 0.25f, "%.2f");
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

    ImGui::InputText("Knowledge Modes", &gui.knowledge_modes);
    if (ImGui::Button("Process Knowledge")) {
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

    ImGui::Separator();
    if (!gui.status_message.empty()) {
        ImGui::TextWrapped("%s", gui.status_message.c_str());
    }
    if (!gui.error_message.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", gui.error_message.c_str());
    }
    ImGui::End();
}

} // namespace neuro
