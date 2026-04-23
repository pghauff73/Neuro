#include "neuro/ui.hpp"

#include "neuro/app.hpp"

#include "imgui.h"

namespace neuro {

void render_session_panel(AppController & app, const EngineState & state, GuiState & gui) {
    ImGui::Begin("Session");
    const AppConfig & config = app.config();
    ImGui::TextWrapped("Model: %s", config.model_path.c_str());
    ImGui::TextWrapped("State File: %s", config.state_file.c_str());
    ImGui::Separator();
    ImGui::Text("Topic");
    ImGui::TextWrapped("%s", state.topic.c_str());

    if (ImGui::Button("Load State")) {
        try {
            app.load_state();
            gui.cached_state = app.snapshot();
            gui.status_message = "Reloaded state from disk.";
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save State")) {
        try {
            app.save_state();
            gui.status_message = "Saved binary state.";
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save Markdown")) {
        try {
            app.save_markdown();
            gui.status_message = "Wrote body-of-work.md.";
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::End();
}

} // namespace neuro
