#include "../include/vk_compute.h"

#include <vector>

#include "../include/file.h"

VulkanCompute::VulkanCompute(VulkanState* vkState) {
    VulkanCompute::vkState = vkState;
}

void VulkanCompute::init(
    std::string shaderFilePath,
    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings,
    VkDescriptorPool descriptorPool, VkCommandPool commandPool) {
    // descriptor set layout
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount =
        (uint32_t)descriptorSetLayoutBindings.size();
    descriptorSetLayoutCreateInfo.pBindings =
        descriptorSetLayoutBindings.data();

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(vkState->getDevice(),
                                                &descriptorSetLayoutCreateInfo,
                                                NULL, &descriptorSetLayout));

    // create pipeline
    auto code = readFile(shaderFilePath);
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = (const uint32_t*)code.data();
    createInfo.codeSize = code.size();

    VK_CHECK_RESULT(vkCreateShaderModule(vkState->getDevice(), &createInfo,
                                         NULL, &shaderModule));

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = shaderModule;
    shaderStageCreateInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(vkState->getDevice(),
                                           &pipelineLayoutCreateInfo, NULL,
                                           &pipelineLayout));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    VK_CHECK_RESULT(
        vkCreateComputePipelines(vkState->getDevice(), VK_NULL_HANDLE, 1,
                                 &pipelineCreateInfo, NULL, &pipeline));

    // allocate descriptors
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool =
        descriptorPool;  // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount =
        1;  // allocate a single descriptor set.
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    VK_CHECK_RESULT(vkAllocateDescriptorSets(
        vkState->getDevice(), &descriptorSetAllocateInfo, &descriptorSet));

    // allocate command buffer
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(
        vkState->getDevice(), &commandBufferAllocateInfo, &commandBuffer));

    VkSemaphoreCreateInfo semaphoreCreateInfo = {};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(vkState->getDevice(), &semaphoreCreateInfo, nullptr,
                      &computeFinishedSemaphore);
}

void VulkanCompute::bindDescriptors(std::vector<VkImageView> bindImages) {
    // bind
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    std::vector<VkDescriptorImageInfo> imageInfos;
    descriptorWrites.resize(bindImages.size());
    imageInfos.resize(bindImages.size());

    for (int i = 0; i < bindImages.size(); i++) {
        imageInfos[i] = {};
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfos[i].imageView = bindImages[i];
        imageInfos[i].sampler = vkState->getTextureSampler();

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[i].pImageInfo = &imageInfos[i];
    }

    vkUpdateDescriptorSets(vkState->getDevice(),
                           (uint32_t)descriptorWrites.size(),
                           descriptorWrites.data(), 0, nullptr);
}

void VulkanCompute::bindUniformBufferDescriptor(uint32_t binding,
                                                VkBuffer buffer,
                                                VkDeviceSize bufferRange) {
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = buffer;
    bufferInfo.range = bufferRange;
    bufferInfo.offset = 0;

    VkWriteDescriptorSet descriptorWrite = {};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = binding;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(vkState->getDevice(), 1, &descriptorWrite, 0,
                           nullptr);
}

void VulkanCompute::bindStorageBufferDescriptor(uint32_t binding,
                                                VkBuffer buffer,
                                                VkDeviceSize bufferRange,
                                                VkDeviceSize bufferOffset) {
    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = buffer;
    bufferInfo.range = bufferRange;
    bufferInfo.offset = bufferOffset;

    VkWriteDescriptorSet descriptorWrite = {};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = binding;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrite.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(vkState->getDevice(), 1, &descriptorWrite, 0,
                           nullptr);
}

void VulkanCompute::bindImageDescriptor(uint32_t binding,
                                        VkImageView imageView) {
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageInfo.imageView = imageView;
    imageInfo.sampler = vkState->getTextureSampler();

    VkWriteDescriptorSet descriptorWrite = {};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = binding;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    descriptorWrite.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(vkState->getDevice(), 1, &descriptorWrite, 0,
                           nullptr);
}

void VulkanCompute::execute(uint32_t groupCountX, uint32_t groupCountY,
                            VkSemaphore waitSemaphore, VkFence signalFence) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

    vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;  // submit a single command buffer
    submitInfo.pCommandBuffers =
        &commandBuffer;  // the command buffer to submit.
    submitInfo.pWaitSemaphores = &waitSemaphore;
    submitInfo.pSignalSemaphores = &computeFinishedSemaphore;

    VK_CHECK_RESULT(vkQueueSubmit(vkState->getComputeQueue(), 1, &submitInfo, signalFence));
}