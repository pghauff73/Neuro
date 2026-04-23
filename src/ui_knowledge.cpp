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

bool render_knowledge_panel(AppController & app, const EngineState & state, GuiState & gui, bool busy) {
    ImGui::Begin("Knowledge Base", nullptr);
    if (ImGui::IsWindowCollapsed()) {
        ImGui::End();
        return true;
    }
    render_section_caption("Capture Entry", "Manually ingest source notes and structured context.");

    ImGui::BeginDisabled(busy);
    ImGui::InputText("Title", &gui.knowledge_title);
    ImGui::InputText("URL", &gui.knowledge_url);
    ImGui::InputTextMultiline("Summary", &gui.knowledge_summary, ImVec2(-FLT_MIN, 54.0f));
    ImGui::InputTextMultiline("Text", &gui.knowledge_text, ImVec2(-FLT_MIN, 54.0f));
    if (ImGui::Button("Add Entry")) {
        try {
            KnowledgeEntry entry;
            entry.title = gui.knowledge_title;
            entry.url = gui.knowledge_url;
            entry.summary = gui.knowledge_summary;
            entry.text = gui.knowledge_text;
            app.add_knowledge_entry(std::move(entry));
            gui.cached_state = app.snapshot();
            gui.status_message = "Added knowledge entry.";
            gui.error_message.clear();
            gui.knowledge_title.clear();
            gui.knowledge_url.clear();
            gui.knowledge_summary.clear();
            gui.knowledge_text.clear();
        } catch (const std::exception & e) {
            gui.error_message = e.what();
        }
    }
    ImGui::EndDisabled();

    ImGui::Separator();
    if (busy) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.78f, 0.38f, 1.0f));
        ImGui::TextWrapped("Knowledge entry capture is locked while an inference step is running.");
        ImGui::PopStyleColor();
    }
    render_section_caption("Indexed Entries", "Current retrieval corpus and ranking data.");
    if (ImGui::BeginTable("knowledge_table", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable, ImVec2(0.0f, 220.0f))) {
        ImGui::TableSetupColumn("Title");
        ImGui::TableSetupColumn("Mode");
        ImGui::TableSetupColumn("Score");
        ImGui::TableSetupColumn("URL");
        ImGui::TableHeadersRow();
        for (const auto & entry : state.knowledge_entries) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextWrapped("%s", entry.title.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(entry.knowledge_node_mode.c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.2f", entry.retrieval_score);
            ImGui::TableSetColumnIndex(3);
            ImGui::TextWrapped("%s", entry.url.c_str());
        }
        ImGui::EndTable();
    }
    ImGui::End();
    return false;
}

} // namespace neuro
