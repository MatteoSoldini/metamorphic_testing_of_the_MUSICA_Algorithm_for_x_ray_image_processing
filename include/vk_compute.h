#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

#include "vk_state.h"

/*
- ? delegate the descriptor set to the user to enable reuse of the object
- ? have the command buffer as an argument to the execute function
*/

class VulkanCompute {
private:
    VulkanState* vkState;

    void createImageSqrtShader();
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSet;
    VkShaderModule shaderModule;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkCommandBuffer commandBuffer;
    VkSemaphore computeFinishedSemaphore;

public:
    VulkanCompute(VulkanState* vkState);

    void init(
        std::string shaderFilePath,
        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings,
        VkDescriptorPool descriptorPool,
        VkCommandPool commandPool
    );

    void bindDescriptors(std::vector<VkImageView> bindImages);
    void bindUniformBufferDescriptor(uint32_t binding, VkBuffer buffer,
                                     VkDeviceSize bufferRange);
    void bindStorageBufferDescriptor(uint32_t binding, VkBuffer buffer,
                                     VkDeviceSize bufferRange,
                                     VkDeviceSize bufferOffset = 0);
    void bindImageDescriptor(uint32_t binding, VkImageView imageView);

    void execute(uint32_t groupCountX, uint32_t groupCountY,
                 VkSemaphore waitSemaphore, VkFence signalFence);

    VkSemaphore getComputeFinishedSemaphore() {
        return computeFinishedSemaphore;
    };

    void cleanup();
};