#include "../include/vk_imgui.h"
#include "../include/vk_utils.h"
#include "../include/vk_processing.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <imgui_impl_glfw.h>
#include <stdexcept>

bool VulkanImGui::init() {
    // create descriptor pool
    // the size of the pool is very oversize, but it's copied from imgui demo itself.
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };

    VkDescriptorPoolCreateInfo descriptorInfo = {};
    descriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptorInfo.maxSets = 1000;
    descriptorInfo.poolSizeCount = std::size(poolSizes);
    descriptorInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(vkState->getDevice(), &descriptorInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        return false;
    }

    // command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = vkState->getGraphicsFamily();

    if (vkCreateCommandPool(vkState->getDevice(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        return false;
    }

    // command buffer
    commandBuffers.resize(framesInFlight);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = commandBuffers.size();

    if (vkAllocateCommandBuffers(vkState->getDevice(), &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        return false;
    }

    // imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    ImGui_ImplVulkan_LoadFunctions([](const char* function_name, void* vulkan_instance) {
        return vkGetInstanceProcAddr(*(reinterpret_cast<VkInstance*>(vulkan_instance)), function_name);
    }, vkState->getInstance());

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = vkState->getInstance();
    initInfo.PhysicalDevice = vkState->getPhysicalDevice();
    initInfo.Device = vkState->getDevice();
    initInfo.QueueFamily = 0;  // not using
    initInfo.Queue = vkState->getGraphicsQueue();
    initInfo.PipelineCache = nullptr;  // not using
    initInfo.DescriptorPool = descriptorPool;
    initInfo.Subpass = 0;
    initInfo.MinImageCount = framesInFlight;
    initInfo.ImageCount = framesInFlight;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.Allocator = nullptr;
    initInfo.CheckVkResultFn = nullptr;
    initInfo.RenderPass = renderPass;
    ImGui_ImplVulkan_Init(&initInfo);

    // Use any command queue
    VkCommandBuffer commandBuffer = commandBuffers[0];    // any

    if (vkResetCommandPool(vkState->getDevice(), commandPool, 0)) {
        return false;
    }
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        return false;
    }

    VkSubmitInfo endInfo = {};
    endInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    endInfo.commandBufferCount = 1;
    endInfo.pCommandBuffers = &commandBuffer;
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return false;
    }

    vkQueueSubmit(vkState->getGraphicsQueue(), 1, &endInfo, VK_NULL_HANDLE);
    vkDeviceWaitIdle(vkState->getDevice());

    return true;
}

void VulkanImGui::cleanup() {
    vkDestroyCommandPool(vkState->getDevice(), commandPool, nullptr);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    vkDestroyDescriptorPool(vkState->getDevice(), descriptorPool, nullptr);
}

VkCommandBuffer VulkanImGui::recordFrame(uint32_t currentFrame, VkFramebuffer frameBuffer, VkExtent2D frameBufferExtent) {
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);

    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffers[currentFrame], &info) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin recording command buffer");
    }

    {
        VkRenderPassBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        info.renderPass = renderPass;
        info.framebuffer = frameBuffer;
        info.renderArea.offset = { 0, 0 };
        info.renderArea.extent = frameBufferExtent;

        VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
        info.clearValueCount = 1;
        info.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffers[currentFrame], &info, VK_SUBPASS_CONTENTS_INLINE);
    }

    // Record dear imgui primitives into command buffer
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers[currentFrame]);

    // Submit command buffer
    vkCmdEndRenderPass(commandBuffers[currentFrame]);
    if (vkEndCommandBuffer(commandBuffers[currentFrame])) {
        throw std::runtime_error("Failed to end recording command buffer");
    }

    return commandBuffers[currentFrame];
}

VulkanImGui::VulkanImGui(VulkanState* vkState, GLFWwindow* window, const int framesInFlight, VkRenderPass renderPass) {
    VulkanImGui::vkState = vkState;
    VulkanImGui::window = window;
    VulkanImGui::framesInFlight = framesInFlight;
    VulkanImGui::renderPass = renderPass;

    commandPool = VK_NULL_HANDLE;
    descriptorPool = VK_NULL_HANDLE;
}
