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

bool render_session_panel(AppController & app, const EngineState & state, GuiState & gui, bool busy) {
    ImGui::Begin("Session Control", nullptr);
    if (ImGui::IsWindowCollapsed()) {
        ImGui::End();
        return true;
    }
    const AppConfig & config = app.config();
    render_section_caption("Runtime Assets", "Persistent session operations and export targets.");
    ImGui::TextWrapped("Model: %s", config.model_path.c_str());
    ImGui::TextWrapped("State File: %s", config.state_file.c_str());
    render_section_caption("Topic");
    ImGui::SetNextItemWidth(-FLT_MIN);
    const bool submitted = ImGui::InputText("##topic_input", &gui.topic_input, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Apply Topic")) {
        try {
            app.set_topic(gui.topic_input);
            gui.cached_state = app.snapshot();
            gui.topic_input = gui.cached_state.topic;
            gui.status_message = "Updated topic.";
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset Topic")) {
        gui.topic_input = state.topic;
    }
    ImGui::EndDisabled();
    if (submitted && !busy) {
        try {
            app.set_topic(gui.topic_input);
            gui.cached_state = app.snapshot();
            gui.topic_input = gui.cached_state.topic;
            gui.status_message = "Updated topic.";
            gui.error_message.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    if (busy) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.58f, 0.66f, 0.76f, 1.0f));
        ImGui::TextWrapped("You can draft a new topic now. Applying it is locked until the inference pipeline is idle.");
        ImGui::PopStyleColor();
    }

    if (busy) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.78f, 0.38f, 1.0f));
        ImGui::TextWrapped("Session writes are locked while an inference step is running.");
        ImGui::PopStyleColor();
    }

    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Load State")) {
        try {
            app.load_state();
            gui.cached_state = app.snapshot();
            gui.topic_input = gui.cached_state.topic;
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
    ImGui::EndDisabled();
    ImGui::End();
    return false;
}

} // namespace neuro
