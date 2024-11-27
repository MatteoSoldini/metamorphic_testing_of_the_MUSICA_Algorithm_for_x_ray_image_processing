#include "../include/app.h"

#include <GLFW/glfw3.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_glfw.h>
#include <imgui.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "../include/vk_imgui.h"
#include "../include/file.h"

App::App() {
    vkState = new VulkanState();
    vkImGui = nullptr;
    vkWindow = nullptr;
    window = nullptr;
    vkProcessing = nullptr;

    currentFrame = 0;
}

void App::init() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // disable OpenGL context
    window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Maverick", nullptr,
                              nullptr);
    glfwSetWindowUserPointer(window, this);

    std::vector<const char*> extensions;
    uint32_t extensions_count = 0;
    const char** glfw_extensions =
        glfwGetRequiredInstanceExtensions(&extensions_count);
    for (uint32_t i = 0; i < extensions_count; i++)
        extensions.push_back(glfw_extensions[i]);

    vkState->init(extensions);

    vkWindow = new VulkanWindow(vkState, window, framesInFlight);
    vkWindow->init();

    vkImGui = new VulkanImGui(vkState, window, framesInFlight, vkWindow->getRenderPass());
    vkImGui->init();

    vkProcessing = new VulkanProcessing(vkState);

    std::vector<VkImageView> procImageViews;
    vkProcessing->init(imageSize, &procImageViews);

    imageDescriptorSets.resize(procImageViews.size());
    for (uint32_t i = 0; i < procImageViews.size(); i++) {
        imageDescriptorSets[i] = (ImTextureID)ImGui_ImplVulkan_AddTexture(
            vkState->getTextureSampler(),
            procImageViews[i],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }
}

void processRun(VulkanProcessing) {}

void App::draw() {
    auto startTime = std::chrono::high_resolution_clock::now();
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    static bool showDemoWindow = false;
    static std::vector<uint8_t> testImage;
    static float mean = 0.0f;
    static float sdev = 0.0f;
    static float snr = 0.0f;

    ImVec4 clearColor = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);

    if (showDemoWindow) ImGui::ShowDemoWindow(&showDemoWindow);

    {
        static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                        ImGuiWindowFlags_NoMove |
                                        ImGuiWindowFlags_NoSavedSettings;

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);

        ImGui::Begin("main_window", nullptr, flags);

        ImGui::Checkbox("ImGui Demo Window", &showDemoWindow);

        ImGui::SameLine();

        if (ImGui::Button("debug process (last image)")) {
            vkProcessing->debugProcess();
        }

        ImGui::SameLine();

        ImVec2 size = ImGui::GetContentRegionAvail();
        ImGui::Image(imageDescriptorSets[vkProcessing->getCurrentIndex()], ImVec2(size.y, size.y));

        ImGui::SameLine();

        ImGui::BeginChild("debug");
        // ImGui::Image(
        //     vkProcessing->getGradationHistogram(),
        //     ImVec2(VulkanProcessing::histRenderWidth, VulkanProcessing::histRenderHeight)
        // );
        // ImGui::Image(
        //     vkProcessing->getNoiseHistogram(),
        //     ImVec2(VulkanProcessing::histRenderWidth, VulkanProcessing::histRenderHeight)
        // );
        ImGui::EndChild();

        ImGui::End();
    }

    // Rendering
    ImGui::Render();

    ImDrawData* drawData = ImGui::GetDrawData();
    const bool isMinimized =
        (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f);
    if (!isMinimized) {
        auto recordingFrame = vkWindow->startFrameRecord(currentFrame);
        VkCommandBuffer commandBuffer = vkImGui->recordFrame(
            currentFrame, recordingFrame.framebuffer, recordingFrame.extent);
        vkWindow->submitFrame(currentFrame, {commandBuffer});
        currentFrame = (currentFrame + 1) & framesInFlight;
    }

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time =
        std::chrono::duration<float, std::chrono::milliseconds::period>(
            currentTime - startTime)
            .count();

    if (time > 5.0f)
        printf("draw time: %fms\n", time);
}

void App::run() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        draw();
    }
}

void App::cleanup() {
    vkImGui->cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();
}
