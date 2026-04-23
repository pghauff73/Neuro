#include "neuro/ui.hpp"

#include "imgui.h"

namespace neuro {

void render_diagnostics_panel(const EngineState & state, const GuiState & gui, const AsyncStepRunner & runner) {
    ImGui::Begin("Diagnostics");
    ImGui::Text("Busy: %s", runner.busy() ? "yes" : "no");
    ImGui::TextWrapped("Last Title: %s", state.last_update_title.c_str());
    ImGui::Separator();
    ImGui::TextWrapped("%s", state.last_update_text.c_str());
    ImGui::Separator();
    ImGui::BeginChild("raw_response", ImVec2(0.0f, 220.0f), true);
    ImGui::TextWrapped("%s", state.last_raw_response.c_str());
    ImGui::EndChild();
    if (!state.knowledge_context.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", state.knowledge_context.c_str());
    }
    if (!gui.error_message.empty()) {
        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", gui.error_message.c_str());
    }
    ImGui::End();
}

} // namespace neuro
