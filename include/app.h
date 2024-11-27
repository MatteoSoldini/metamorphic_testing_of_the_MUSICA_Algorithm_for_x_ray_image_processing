#pragma once
#include "vk_imgui.h"
#include "vk_processing.h"
#include "vk_state.h"
#include "vk_window.h"

#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

class App {
private:
    const int framesInFlight = 2;

    GLFWwindow* window;

    VulkanState* vkState;
    VulkanWindow* vkWindow;
    VulkanImGui* vkImGui;
    VulkanProcessing* vkProcessing;

    uint32_t currentFrame;

    const uint32_t imageSize = 1792;
    const std::string filePath = 
        "..\\..\\..\\raw_images\\torax_1.raw";
        //"..\\..\\..\\test\\metamorphic_test\\ra\\head_2.raw_qn_0.001.raw";
    std::vector<uint16_t> imageData;

    std::vector<ImTextureID> imageDescriptorSets;

    void draw();

public:
    App();
    void init();
    void run();
    void cleanup();
};