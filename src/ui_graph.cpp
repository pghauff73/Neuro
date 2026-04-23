#include "neuro/ui.hpp"

#include "imgui.h"

namespace neuro {

namespace {

ImVec4 mode_color(const std::string & mode) {
    if (mode == "structure") return ImVec4(0.55f, 0.82f, 0.96f, 1.0f);
    if (mode == "explore") return ImVec4(0.96f, 0.79f, 0.36f, 1.0f);
    if (mode == "connect") return ImVec4(0.52f, 0.88f, 0.72f, 1.0f);
    if (mode == "stress_test") return ImVec4(0.95f, 0.47f, 0.40f, 1.0f);
    if (mode == "stabilize") return ImVec4(0.72f, 0.67f, 0.96f, 1.0f);
    if (mode == "action") return ImVec4(0.98f, 0.59f, 0.28f, 1.0f);
    if (mode == "activate") return ImVec4(0.98f, 0.59f, 0.28f, 1.0f);
    return ImVec4(0.68f, 0.76f, 0.88f, 1.0f);
}

std::string compact_text(const std::string & text, size_t max_chars) {
    if (text.size() <= max_chars) {
        return text;
    }
    return text.substr(0, max_chars - 3) + "...";
}

void draw_node_card(const SectionNode & node, bool selected) {
    const ImVec4 accent = mode_color(node.authoring_mode);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, selected ? ImVec4(0.13f, 0.18f, 0.25f, 1.0f) : ImVec4(0.10f, 0.13f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(accent.x, accent.y, accent.z, selected ? 0.90f : 0.55f));
    ImGui::BeginChild(node.node_id.c_str(), ImVec2(220.0f, 150.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PushStyleColor(ImGuiCol_Text, accent);
    ImGui::Text("Iteration %d", node.iteration);
    ImGui::PopStyleColor();
    ImGui::TextWrapped("%s", node.title.c_str());
    ImGui::Separator();
    ImGui::Text("Mode: %s", node.authoring_mode.c_str());
    ImGui::Text("State: %s", node.source_state.c_str());
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.76f, 0.82f, 0.90f, 1.0f));
    const std::string preview = compact_text(node.content, 120);
    ImGui::TextWrapped("%s", preview.c_str());
    ImGui::PopStyleColor();
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
}

} // namespace

bool render_graph_panel(const EngineState & state, GuiState & gui) {
    ImGui::Begin("Body Graph", nullptr);
    if (ImGui::IsWindowCollapsed()) {
        ImGui::End();
        return true;
    }

    if (gui.selected_node_id.empty()) {
        gui.selected_node_id = state.graph.focus_node_id;
    }

    ImGui::TextUnformatted("Narrative Timeline");
    ImGui::Separator();
    ImGui::BeginChild("graph_timeline", ImVec2(0.0f, 190.0f), true, ImGuiWindowFlags_HorizontalScrollbar);
    for (const auto & node_id : state.graph.insertion_order) {
        auto it = state.graph.nodes.find(node_id);
        if (it == state.graph.nodes.end()) {
            continue;
        }
        const bool selected = gui.selected_node_id == node_id;
        if (ImGui::InvisibleButton((node_id + "_button").c_str(), ImVec2(220.0f, 150.0f))) {
            gui.selected_node_id = node_id;
        }
        const ImVec2 min = ImGui::GetItemRectMin();
        ImGui::SetCursorScreenPos(min);
        draw_node_card(it->second, selected);
        if (&node_id != &state.graph.insertion_order.back()) {
            ImGui::SameLine();
        }
    }
    ImGui::EndChild();

    ImGui::Spacing();
    ImGui::TextUnformatted("Selected Node");
    ImGui::Separator();
    ImGui::BeginChild("graph_detail", ImVec2(0.0f, 0.0f), true);
    auto it = state.graph.nodes.find(gui.selected_node_id);
    if (it == state.graph.nodes.end()) {
        ImGui::TextUnformatted("No node selected.");
    } else {
        const SectionNode & node = it->second;
        ImGui::PushStyleColor(ImGuiCol_Text, mode_color(node.authoring_mode));
        ImGui::TextWrapped("%s", node.title.c_str());
        ImGui::PopStyleColor();
        ImGui::Separator();
        ImGui::Text("Mode: %s", node.authoring_mode.c_str());
        ImGui::Text("State: %s", node.source_state.c_str());
        ImGui::Text("Iteration: %d", node.iteration);
        ImGui::Text("Parents: %d", static_cast<int>(node.parent_ids.size()));
        ImGui::SameLine();
        ImGui::Text("Children: %d", static_cast<int>(node.child_ids.size()));
        ImGui::TextWrapped("Rationale: %s", node.rationale.c_str());
        ImGui::Separator();
        ImGui::TextWrapped("%s", node.content.c_str());
    }
    ImGui::EndChild();
    ImGui::End();
    return false;
}

} // namespace neuro
