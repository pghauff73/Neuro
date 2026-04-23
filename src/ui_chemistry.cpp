#include "neuro/ui.hpp"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace neuro {

namespace {

constexpr float kPi = 3.1415926535f;

ImU32 dim_color(const ImVec4 & color, float alpha_scale = 1.0f) {
    return ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, color.w * alpha_scale));
}

ImVec4 chem_color(int index) {
    static const ImVec4 kColors[kChemCount] = {
        ImVec4(0.96f, 0.74f, 0.28f, 1.0f),
        ImVec4(0.42f, 0.80f, 0.61f, 1.0f),
        ImVec4(0.49f, 0.78f, 0.95f, 1.0f),
        ImVec4(0.95f, 0.42f, 0.37f, 1.0f),
        ImVec4(0.94f, 0.58f, 0.22f, 1.0f),
        ImVec4(0.72f, 0.65f, 0.96f, 1.0f),
        ImVec4(0.57f, 0.88f, 0.88f, 1.0f),
        ImVec4(0.98f, 0.57f, 0.74f, 1.0f),
    };
    return kColors[index % kChemCount];
}

void render_state_transition_strip(const EngineState & state) {
    const char * states[3] = {
        state.current_state.c_str(),
        state.authoring_mode.c_str(),
        state.next_state.c_str(),
    };
    const char * labels[3] = {"Current", "Mode", "Next"};
    const ImVec4 accents[3] = {
        ImVec4(0.56f, 0.84f, 0.95f, 1.0f),
        ImVec4(0.55f, 0.88f, 0.74f, 1.0f),
        ImVec4(0.98f, 0.80f, 0.44f, 1.0f),
    };

    if (ImGui::BeginTable("transition_strip", 3, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();
        for (int i = 0; i < 3; ++i) {
            ImGui::TableSetColumnIndex(i);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.13f, 0.18f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(accents[i].x, accents[i].y, accents[i].z, 0.55f));
            ImGui::BeginChild(labels[i], ImVec2(0.0f, 48.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.58f, 0.66f, 0.76f, 1.0f));
            ImGui::TextUnformatted(labels[i]);
            ImGui::PopStyleColor();
            ImGui::PushStyleColor(ImGuiCol_Text, accents[i]);
            ImGui::TextWrapped("%s", states[i]);
            ImGui::PopStyleColor();
            ImGui::EndChild();
            ImGui::PopStyleColor(2);
        }
        ImGui::EndTable();
    }
}

void render_radar_chart(const EngineState & state) {
    const float chart_height = 250.0f;
    const ImVec2 canvas_size(ImGui::GetContentRegionAvail().x, chart_height);
    if (canvas_size.x <= 40.0f || canvas_size.y <= 40.0f) {
        return;
    }

    ImGui::BeginChild("chem_radar", canvas_size, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImDrawList * draw = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 size = ImGui::GetContentRegionAvail();
    const ImVec2 center(origin.x + size.x * 0.5f, origin.y + size.y * 0.52f);
    const float radius = std::min(size.x, size.y) * 0.30f;
    const int rings = 4;

    for (int ring = 1; ring <= rings; ++ring) {
        const float r = radius * static_cast<float>(ring) / static_cast<float>(rings);
        for (int i = 0; i < kChemCount; ++i) {
            const float a0 = -kPi * 0.5f + (2.0f * kPi * static_cast<float>(i) / static_cast<float>(kChemCount));
            const float a1 = -kPi * 0.5f + (2.0f * kPi * static_cast<float>((i + 1) % kChemCount) / static_cast<float>(kChemCount));
            const ImVec2 p0(center.x + std::cos(a0) * r, center.y + std::sin(a0) * r);
            const ImVec2 p1(center.x + std::cos(a1) * r, center.y + std::sin(a1) * r);
            draw->AddLine(p0, p1, dim_color(ImVec4(0.24f, 0.29f, 0.36f, 0.55f)), 1.0f);
        }
    }

    ImVec2 value_points[kChemCount];
    for (int i = 0; i < kChemCount; ++i) {
        const float angle = -kPi * 0.5f + (2.0f * kPi * static_cast<float>(i) / static_cast<float>(kChemCount));
        const float line_radius = radius + 18.0f;
        const ImVec2 axis_end(center.x + std::cos(angle) * line_radius, center.y + std::sin(angle) * line_radius);
        draw->AddLine(center, axis_end, dim_color(ImVec4(0.30f, 0.36f, 0.44f, 0.75f)), 1.0f);

        const float chem_radius = radius * static_cast<float>(state.chem_values[i]);
        value_points[i] = ImVec2(center.x + std::cos(angle) * chem_radius, center.y + std::sin(angle) * chem_radius);

        const ImVec2 label_pos(center.x + std::cos(angle) * (radius + 26.0f), center.y + std::sin(angle) * (radius + 26.0f));
        draw->AddText(ImVec2(label_pos.x - 12.0f, label_pos.y - 6.0f), dim_color(chem_color(i)), kChemNames[i].c_str());
    }

    draw->AddConvexPolyFilled(value_points, kChemCount, dim_color(ImVec4(0.37f, 0.70f, 0.94f, 0.20f)));
    draw->AddPolyline(value_points, kChemCount, dim_color(ImVec4(0.55f, 0.84f, 0.98f, 0.95f)), true, 2.0f);

    for (int i = 0; i < kChemCount; ++i) {
        draw->AddCircleFilled(value_points[i], 4.0f, dim_color(chem_color(i)));
    }

    ImGui::Dummy(canvas_size);
    ImGui::EndChild();
}

} // namespace

bool render_chemistry_panel(const EngineState & state) {
    ImGui::Begin("Chemistry Monitor", nullptr);
    if (ImGui::IsWindowCollapsed()) {
        ImGui::End();
        return true;
    }
    ImGui::TextUnformatted("State Flow");
    ImGui::Separator();
    render_state_transition_strip(state);

    ImGui::Separator();
    ImGui::TextUnformatted("Chemical Posture");
    ImGui::Separator();
    render_radar_chart(state);

    ImGui::TextUnformatted("Signal Intensities");
    ImGui::Separator();
    for (int i = 0; i < kChemCount; ++i) {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, chem_color(i));
        ImGui::Text("%s", kChemNames[i].c_str());
        ImGui::SameLine(96.0f);
        ImGui::ProgressBar(static_cast<float>(state.chem_values[i]), ImVec2(-FLT_MIN, 0.0f));
        ImGui::PopStyleColor();
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Mode Pressure");
    ImGui::Separator();
    std::vector<std::pair<std::string, double>> modes(state.authoring_expression.begin(), state.authoring_expression.end());
    std::sort(modes.begin(), modes.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });
    for (const auto & mode : modes) {
        ImGui::Text("%s", mode.first.c_str());
        ImGui::SameLine(96.0f);
        ImGui::ProgressBar(static_cast<float>(mode.second), ImVec2(-FLT_MIN, 0.0f));
    }
    ImGui::End();
    return false;
}

} // namespace neuro
