#include "../include/vk_processing.h"

#include <array>
#include <chrono>
#include <cmath>

#include "../include/bezier_curve.h"
#include "../include/image.h"
#include "../include/file.h"
#include "../include/vk_utils.h"

#include <stb_image_write.h>

#define ASSERT_MSG(cond, msg)                           \
    if (!(cond)) {                                      \
        fprintf(stderr, "VK STATE ERROR: %s\n", msg); \
        return false;                                   \
    }

#define ASSERT(cond)                           \
    if (!(cond)) {                             \
        return false;                          \
    }

bool VulkanProcessing::initMemory() {
    lastRawImage.resize(imageSize * imageSize);

    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
        imageSize,
        imageSize,
        VK_FORMAT_R16_UINT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        &inputImageState
    ));

    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
        imageSize,
        imageSize,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        &sqrtImageState
    ));

    uint32_t currentImageSize = imageSize;
    while (currentImageSize > 1) {
        currentImageSize = std::ceil(currentImageSize / (float)reduceAreaSize);

        VkImageState maxImageState;

        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            currentImageSize,
            currentImageSize,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &maxImageState
        ));

        maxReduceImageStates.push_back(maxImageState);

        VkImageState minImageState;

        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            currentImageSize, currentImageSize, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &minImageState));

        minReduceImageStates.push_back(minImageState);
    }

    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
        imageSize,
        imageSize,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        &normalizedImageState
    ));

    bandpassImageStates.resize(pyramidLevels);
    downsampledImageStates.resize(pyramidLevels);
    lowpassImageStates.resize(pyramidLevels);
    smoothImageStates.resize(pyramidLevels);
    upsampledImageStates.resize(pyramidLevels);

    currentImageSize = imageSize;
    for (uint32_t i = 0; i < pyramidLevels; i++) {
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            currentImageSize,
            currentImageSize,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &bandpassImageStates[i]
        ));
        uint32_t downsampledSize = (uint32_t)std::ceil(currentImageSize / 2.0f);
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            downsampledSize, downsampledSize, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &downsampledImageStates[i]));
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            currentImageSize, currentImageSize, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &smoothImageStates[i]));
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            currentImageSize, currentImageSize, VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &upsampledImageStates[i]));
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            currentImageSize,
            currentImageSize,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &lowpassImageStates[i]
        ));

        currentImageSize = (uint32_t)std::ceil(currentImageSize / 2.0f);
    }

    expandBandpassImageStates.resize(pyramidLevels);
    expandImageStates.resize(pyramidLevels);
    expandUpsampledImageStates.resize(pyramidLevels);
    expandLowpassImageStates.resize(pyramidLevels);

    for (int i = 0; i < pyramidLevels; i++) {
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            bandpassImageStates[pyramidLevels - i - 1].width,
            bandpassImageStates[pyramidLevels - i - 1].height,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &expandImageStates[i]
        ));
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            bandpassImageStates[pyramidLevels - i - 1].width,
            bandpassImageStates[pyramidLevels - i - 1].height,
            VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &expandUpsampledImageStates[i]));
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            bandpassImageStates[pyramidLevels - i - 1].width,
            bandpassImageStates[pyramidLevels - i - 1].height,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &expandLowpassImageStates[i]
        ));
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            bandpassImageStates[pyramidLevels - i - 1].width,
            bandpassImageStates[pyramidLevels - i - 1].height,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &expandBandpassImageStates[i]
        ));
    }

    noiseHistImageStates.resize(pyramidLevels);
    //histogramRenderImageStates.resize(pyramidLevels);
    noiseHistMaxBufferStates.resize(pyramidLevels);
    contrastParametersBufferStates.resize(pyramidLevels);
    contrastCurveBufferStates.resize(pyramidLevels);
    constrastCurveImageStates.resize(pyramidLevels);
    sdevImageStates.resize(pyramidLevels);

    for (int i = 0; i < pyramidLevels; i++) {
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            bandpassImageStates[i].width,
            bandpassImageStates[i].height,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &sdevImageStates[i]
        ));
        ASSERT(vkState->create1DImage2(
            vkState->getComputeQueue(),
            noiseHistogramBins, VK_FORMAT_R32_UINT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT,    // for clearing
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &noiseHistImageStates[i]));
        // vkState->createImage2(
        //     vkState->getComputeQueue(),
        //     histRenderWidth, histRenderHeight, VK_FORMAT_R8G8B8A8_UNORM,
        //     VK_IMAGE_TILING_OPTIMAL,
        //     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        //     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
        //     &histogramRenderImageStates[i]);
        ASSERT(vkState->createBuffer2(
            sizeof(HistogramMaxPoint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &noiseHistMaxBufferStates[i]));
        ASSERT(vkState->createBuffer2(
            sizeof(ContrastCurveObj),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &contrastCurveBufferStates[i]
        ));
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            histRenderWidth, histRenderHeight, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &constrastCurveImageStates[i]));

        ASSERT(vkState->createBuffer2(sizeof(ContrastParameters),
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                               &contrastParametersBufferStates[i]));

        ContrastParameters contrastParameters = {};
        uint32_t coarserLevelsCount = pyramidLevels - coarserLevelsStart;

#ifdef LINEAR_HIGH_CONTRAST_LEVELS_REDUCTION
        contrastParameters.highContrastFactor =
            i < coarserLevelsStart
                ? 1.0f
                : 1.0f - (i - coarserLevelsStart) *
                             (1.0f - highContrastMaxReduction) /
                             (pyramidLevels - coarserLevelsStart - 1);
#else
        contrastParameters.highContrastFactor =
            i < coarserLevelsStart ? 1.0f
                                   : std::pow(highContrastMaxReduction,
                                              ((float)(i - coarserLevelsStart) /
                                               (coarserLevelsCount - 1)));
#endif

        // printf("level %i: %f ^ (%i / %i) = %f\n", i, highContrastMaxReduction,
        //        i - coarserLevelsStart, coarserLevelsCount - 1,
        //        contrastParameters.highContrastFactor);

#ifdef LINEAR_LOW_CONTRAST_LEVELS_REDUCTION
        contrastParameters.lowContrastFactor =
            i < coarserLevelsStart
                ? lowContrastMaxEnhancment -
                      i * ((lowContrastMaxEnhancment - 1) / coarserLevelsStart)
                : 1.0f;
#else
        contrastParameters.lowContrastFactor =
            i < coarserLevelsStart
                ? std::pow(lowContrastMaxEnhancment,
                           1.0f - ((float)i / coarserLevelsStart))
                : 1.0f;
#endif

        vkState->loadDataToBuffer(&contrastParameters,
                                  sizeof(ContrastParameters),
                                  contrastParametersBufferStates[i].buffer);
    }

    // noise reduction
    for (uint32_t i = 0; i < cnrLevel; i++) {
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            expandBandpassImageStates[pyramidLevels - cnrLevel + i].width,
            expandBandpassImageStates[pyramidLevels - cnrLevel + i].height,
            VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &expandBandpassNoiseRedImages[i]
        ));

        ASSERT(vkState->createBuffer2(
            sizeof(NoiseReductionParams),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &noiseReductionParamsBuffers[i]
        ));

        NoiseReductionParams params = {};
        params.highCnr = nrHighCnr;
        params.highFactor = nrMaxHighFactor - (nrMaxHighFactor - 1.0f) * ((float)(i) / cnrLevel);
        params.lowCnr = nrLowCnr;
        params.lowFactor = nrMinLowFactor + (1.0f - nrMinLowFactor) * ((float)(i) / cnrLevel);

        vkState->loadDataToBuffer(
            &params,
            sizeof(NoiseReductionParams),
            noiseReductionParamsBuffers[i].buffer
        );
    }

    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
            histRenderWidth, histRenderHeight, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
            &noiseHistRenderImage));

#ifdef GRAD_WITH_LINEAR_IMAGE
    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
        imageSize,
        imageSize,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        &linearImageState
    ));
#endif

    for (int i = 0; i < rgbImagesCount; i++) {
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            imageSize,
            imageSize,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_GENERAL,
            &rgbImageStates[i]
        ));
    }

    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
        bandpassImageStates[cnrLevel].width,
        bandpassImageStates[cnrLevel].height,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        &cnrImageState
    ));

#ifdef CNR_DEBUG
    vkState->createImage2(
        vkState->getComputeQueue(),
        cnrImageState.width,
        cnrImageState.height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
        &cnrDebugImageState);
#endif

    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
        imageSize,
        imageSize,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        &relevantImageState
    ));

    ASSERT(vkState->create1DImage2(
        vkState->getComputeQueue(),
        gradHistogramBins, VK_FORMAT_R32_UINT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
        &gradHistState));

    ASSERT(vkState->createBuffer2(
        sizeof(HistogramMaxPoint), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &gradHistMaxBufferState));

    ASSERT(vkState->createBuffer2(
        sizeof(GradCurveObj), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &gradCurveBufferState));

    ASSERT(vkState->createImage2(
        vkState->getComputeQueue(),
        imageSize,
        imageSize,
        VK_FORMAT_R32_SFLOAT, 
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        &gradedImageState
    ));

#ifdef ENABLE_CLAHE
    vkState->createImage2(
        vkState->getComputeQueue(),
        imageSize, imageSize, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
        &claheGradedImageState);

    vkState->create3DImage2(
        claheTiles, claheTiles, claheHistogramBins, VK_FORMAT_R32_UINT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_LAYOUT_GENERAL,
        &claheHistogramsImageState);

    vkState->createBuffer2(
        sizeof(GradPoint) * claheHistogramBins * claheTiles * claheTiles,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        &claheGradCurveBuffer);
#endif

    // out image
    for (uint32_t i = 0; i < 2; i++) {
        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            imageSize,
            imageSize,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            &outImages[i]
        ));

        // TODO: move to vkImgui
        // outDescriptorSet[i] = ImGui_ImplVulkan_AddTexture(
        //     vkState->getTextureSampler(),
        //     outImages[i].imageView,
        //     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        // );

        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            histRenderWidth,
            histRenderHeight,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            &gradHistImages[i]
        ));

        // TODO: move to vkImgui
        // gradHistDescriptorSet[i] = ImGui_ImplVulkan_AddTexture(
        //     vkState->getTextureSampler(),
        //     gradHistImages[i].imageView,
        //     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        // );

        ASSERT(vkState->createImage2(
            vkState->getComputeQueue(),
            histRenderWidth,
            histRenderHeight,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            &noiseHistRenderImages[i]
        ));

        // TODO: move to vkImgui
        // noiseHistDescriptorSet[i] = ImGui_ImplVulkan_AddTexture(
        //     vkState->getTextureSampler(),
        //     noiseHistRenderImages[i].imageView,
        //     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        // );
    }

    return true;
}

bool VulkanProcessing::createDescriptorPool() {
    VkDescriptorPoolSize poolSizes[] = {
        //{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000}};

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets =
        1000;  // we only need to allocate one descriptor set from the pool.
    descriptorPoolCreateInfo.poolSizeCount = std::size(poolSizes);
    descriptorPoolCreateInfo.pPoolSizes = poolSizes;

    VK_CHECK_RESULT(vkCreateDescriptorPool(vkState->getDevice(),
                                           &descriptorPoolCreateInfo, NULL,
                                           &descriptorPool));

    return true;
}

bool VulkanProcessing::createCommandBuffer() {
    /*
        We are getting closer to the end. In order to send commands to the
        device(GPU), we must first record commands into a command buffer. To
        allocate a command buffer, we must first create a command pool. So let
        us do that.
        */
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
;
    // the queue family of this command pool. All command buffers allocated
    // from this command pool, must be submitted to queues of this family
    // ONLY.
    commandPoolCreateInfo.queueFamilyIndex =
        queryComputeQueueFamily(vkState->getPhysicalDevice()).value();
    VK_CHECK_RESULT(vkCreateCommandPool(
        vkState->getDevice(), &commandPoolCreateInfo, NULL, &commandPool));

    for (uint32_t i = 0; i < imageToClear; i++) {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;
        VK_CHECK_RESULT(vkAllocateCommandBuffers(
            vkState->getDevice(), &commandBufferAllocateInfo, &clearCommandBuffers[i]));
    }

    return true;
}

void VulkanProcessing::createShaders() {
    {
        vkImageSqrtShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(2);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkImageSqrtShader->init("shaders/img_sqrt.spv",
                                descriptorSetLayoutBindings, descriptorPool,
                                commandPool);
        vkImageSqrtShader->bindImageDescriptor(0, sqrtImageState.imageView);
        vkImageSqrtShader->bindImageDescriptor(1, inputImageState.imageView);
    }

    {
        uint32_t maxReduceImageCount = maxReduceImageStates.size();
        vkImageMaxReduceShaders.resize(maxReduceImageCount);
        for (uint32_t i = 0; i < maxReduceImageCount; i++) {
            VulkanCompute* vkImageMaxReduceShader = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageMaxReduceShader->init("shaders/img_max_reduce.spv",
                                         descriptorSetLayoutBindings,
                                         descriptorPool, commandPool);
            vkImageMaxReduceShader->bindImageDescriptor(
                0, i == 0 ? sqrtImageState.imageView
                          : maxReduceImageStates[i - 1].imageView);
            vkImageMaxReduceShader->bindImageDescriptor(
                1, maxReduceImageStates[i].imageView);

            vkImageMaxReduceShaders[i] = vkImageMaxReduceShader;
        }
    }

    {
        vkImageMinReduceShaders.resize(minReduceImageStates.size());
        for (uint32_t i = 0; i < minReduceImageStates.size(); i++) {
            VulkanCompute* vkImageMinReduceShader = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageMinReduceShader->init("shaders/min_reduce.spv",
                                         descriptorSetLayoutBindings,
                                         descriptorPool, commandPool);
            vkImageMinReduceShader->bindImageDescriptor(
                0, i == 0 ? sqrtImageState.imageView
                          : minReduceImageStates[i - 1].imageView);
            vkImageMinReduceShader->bindImageDescriptor(
                1, minReduceImageStates[i].imageView);

            vkImageMinReduceShaders[i] = vkImageMinReduceShader;
        }
    }

    {
        vkImageNormalizeShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(4);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[3] = {};
        descriptorSetLayoutBindings[3].binding = 3;  // binding = 3
        descriptorSetLayoutBindings[3].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[3].descriptorCount = 1;
        descriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkImageNormalizeShader->init("shaders/img_normalize.spv",
                                     descriptorSetLayoutBindings,
                                     descriptorPool, commandPool);
        vkImageNormalizeShader->bindImageDescriptor(0,
                                                    sqrtImageState.imageView);
        vkImageNormalizeShader->bindImageDescriptor(
            1, maxReduceImageStates[maxReduceImageStates.size() - 1].imageView);
        vkImageNormalizeShader->bindImageDescriptor(
            2, minReduceImageStates[minReduceImageStates.size() - 1].imageView);
        vkImageNormalizeShader->bindImageDescriptor(
            3, normalizedImageState.imageView);
    }

    // image pyramid
    vkImageSmoothShaders.resize(pyramidLevels);
    vkImageDifferenceShaders.resize(pyramidLevels);
    vkImageSmoothUpsampledShaders.resize(pyramidLevels);
    vkImageDifferenceShaders.resize(pyramidLevels);
    vkImageDownsampleShaders.resize(pyramidLevels);
    vkImageUpsampleShaders.resize(pyramidLevels);

    for (int i = 0; i < pyramidLevels; i++) {
        {
            vkImageSmoothShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageSmoothShaders[i]->init("shaders/img_smooth.spv",
                                          descriptorSetLayoutBindings,
                                          descriptorPool, commandPool);
            vkImageSmoothShaders[i]->bindDescriptors(
                {smoothImageStates[i].imageView,
                 i == 0 ? normalizedImageState.imageView
                        : downsampledImageStates[i - 1].imageView});
        }

        {
            vkImageDownsampleShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageDownsampleShaders[i]->init("shaders/img_downsample.spv",
                                              descriptorSetLayoutBindings,
                                              descriptorPool, commandPool);
            vkImageDownsampleShaders[i]->bindDescriptors(
                {downsampledImageStates[i].imageView,
                 smoothImageStates[i].imageView});
        }

        {
            vkImageUpsampleShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageUpsampleShaders[i]->init("shaders/img_upsample.spv",
                                            descriptorSetLayoutBindings,
                                            descriptorPool, commandPool);
            vkImageUpsampleShaders[i]->bindDescriptors(
                {upsampledImageStates[i].imageView,
                 downsampledImageStates[i].imageView});
        }

        {
            vkImageSmoothUpsampledShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageSmoothUpsampledShaders[i]->init(
                "shaders/img_smooth_upsampled.spv", descriptorSetLayoutBindings,
                descriptorPool, commandPool);
            vkImageSmoothUpsampledShaders[i]->bindDescriptors(
                {lowpassImageStates[i].imageView,
                 upsampledImageStates[i].imageView});
        }

        {
            vkImageDifferenceShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(3);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[2] = {};
            descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
            descriptorSetLayoutBindings[2].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[2].descriptorCount = 1;
            descriptorSetLayoutBindings[2].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageDifferenceShaders[i]->init("shaders/img_difference.spv",
                                              descriptorSetLayoutBindings,
                                              descriptorPool, commandPool);
            vkImageDifferenceShaders[i]->bindDescriptors(
                {bandpassImageStates[i].imageView,
                 i == 0 ? normalizedImageState.imageView
                        : downsampledImageStates[i - 1].imageView,
                 lowpassImageStates[i].imageView});
        }
    }

    vkImageExpandAdditionShaders.resize(pyramidLevels);
    vkImageExpandUpsampleShaders.resize(pyramidLevels);
    vkImageExpandLowpassShaders.resize(pyramidLevels);

    for (int i = 0; i < pyramidLevels; i++) {
        {
            vkImageExpandUpsampleShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageExpandUpsampleShaders[i]->init("shaders/img_upsample.spv",
                                                  descriptorSetLayoutBindings,
                                                  descriptorPool, commandPool);
            vkImageExpandUpsampleShaders[i]->bindDescriptors(
                {expandUpsampledImageStates[i].imageView,
                 i == 0
                     ? downsampledImageStates[pyramidLevels - i - 1].imageView
                     : expandImageStates[i - 1].imageView});
        }

        {
            vkImageExpandLowpassShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageExpandLowpassShaders[i]->init(
                "shaders/img_smooth_upsampled.spv", descriptorSetLayoutBindings,
                descriptorPool, commandPool);
            vkImageExpandLowpassShaders[i]->bindDescriptors(
                {expandLowpassImageStates[i].imageView,
                 expandUpsampledImageStates[i].imageView});
        }
        {
            vkImageExpandAdditionShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(3);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[2] = {};
            descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
            descriptorSetLayoutBindings[2].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[2].descriptorCount = 1;
            descriptorSetLayoutBindings[2].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageExpandAdditionShaders[i]->init("shaders/img_addition.spv",
                                                  descriptorSetLayoutBindings,
                                                  descriptorPool, commandPool);
            
            vkImageExpandAdditionShaders[i]->bindImageDescriptor(
                0, expandImageStates[i].imageView
            );
            vkImageExpandAdditionShaders[i]->bindImageDescriptor(
                1, expandLowpassImageStates[i].imageView
            );
            
            uint32_t currentLevel = pyramidLevels - i - 1;
            uint32_t noiseRedPos = i - (pyramidLevels - cnrLevel);
            vkImageExpandAdditionShaders[i]->bindImageDescriptor(
                2,
                currentLevel < cnrLevel - 1 ?
                expandBandpassNoiseRedImages[noiseRedPos].imageView :
                expandBandpassImageStates[i].imageView
            );
        }
    }

    {
        vkImageSDevShaders.resize(pyramidLevels);
        for (int i = 0; i < pyramidLevels; i++) {
            vkImageSDevShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageSDevShaders[i]->init("shaders/img_sdev.spv",
                                        descriptorSetLayoutBindings,
                                        descriptorPool, commandPool);
            vkImageSDevShaders[i]->bindImageDescriptor(
                0, sdevImageStates[i].imageView);
            vkImageSDevShaders[i]->bindImageDescriptor(
                1, bandpassImageStates[i].imageView);
        }
    }

    {
        vkImageApplyContrastCurveShaders.resize(pyramidLevels);
        for (int i = 0; i < pyramidLevels; i++) {
            vkImageApplyContrastCurveShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(4);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[2] = {};
            descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
            descriptorSetLayoutBindings[2].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[2].descriptorCount = 1;
            descriptorSetLayoutBindings[2].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[3] = {};
            descriptorSetLayoutBindings[3].binding = 3;  // binding = 3
            descriptorSetLayoutBindings[3].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[3].descriptorCount = 1;
            descriptorSetLayoutBindings[3].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageApplyContrastCurveShaders[i]->init(
                "shaders/contrast_curve_apply.spv", descriptorSetLayoutBindings,
                descriptorPool, commandPool);
            vkImageApplyContrastCurveShaders[i]->bindImageDescriptor(
                0, expandBandpassImageStates[i].imageView
            );
            vkImageApplyContrastCurveShaders[i]->bindImageDescriptor(
                1, bandpassImageStates[pyramidLevels - i - 1].imageView
            );
            vkImageApplyContrastCurveShaders[i]->bindStorageBufferDescriptor(
                2, contrastCurveBufferStates[pyramidLevels - i - 1].buffer,
                contrastCurveBufferStates[pyramidLevels - i - 1].size
            );
            vkImageApplyContrastCurveShaders[i]->bindImageDescriptor(
                3, sdevImageStates[pyramidLevels - i - 1].imageView
            );
        }
    }

    {
        vkNoiseHistShaders.resize(pyramidLevels);
        for (int i = 0; i < pyramidLevels; i++) {
            vkNoiseHistShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkNoiseHistShaders[i]->init("shaders/noise_hist.spv",
                                        descriptorSetLayoutBindings,
                                        descriptorPool, commandPool);
            vkNoiseHistShaders[i]->bindImageDescriptor(
                0, noiseHistImageStates[i].imageView);
            vkNoiseHistShaders[i]->bindImageDescriptor(
                1, sdevImageStates[i].imageView);
        }
    }

    {
        vkNoiseHistMaxShaders.resize(pyramidLevels);
        for (int i = 0; i < pyramidLevels; i++) {
            vkNoiseHistMaxShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkNoiseHistMaxShaders[i]->init("shaders/img_histogram_max.spv",
                                           descriptorSetLayoutBindings,
                                           descriptorPool, commandPool);
            vkNoiseHistMaxShaders[i]->bindStorageBufferDescriptor(
                0, noiseHistMaxBufferStates[i].buffer,
                noiseHistMaxBufferStates[i].size);
            vkNoiseHistMaxShaders[i]->bindImageDescriptor(
                1, noiseHistImageStates[i].imageView);
        }
    }

#ifdef RENDER_HISTS
    {
        vkNoiseHistRenderShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding>
            descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(3);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags =
            VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags =
            VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags =
            VK_SHADER_STAGE_COMPUTE_BIT;

        vkNoiseHistRenderShader->init(
            "shaders/noise_hist_render.spv", descriptorSetLayoutBindings,
            descriptorPool, commandPool);
        vkNoiseHistRenderShader->bindImageDescriptor(
            0, noiseHistRenderImage.imageView);
        vkNoiseHistRenderShader->bindImageDescriptor(
            1, noiseHistImageStates[cnrLevel].imageView);
        vkNoiseHistRenderShader->bindStorageBufferDescriptor(
            2, noiseHistMaxBufferStates[cnrLevel].buffer,
            noiseHistMaxBufferStates[cnrLevel].size);
    }
#endif

    {
        for (uint32_t i = 0; i < 2; i++) {
            vkNoiseHistRenderShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(3);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[2] = {};
            descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
            descriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[2].descriptorCount = 1;
            descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            vkNoiseHistRenderShaders[i]->init(
                "shaders/noise_hist_render.spv", descriptorSetLayoutBindings,
                descriptorPool, commandPool);
            vkNoiseHistRenderShaders[i]->bindImageDescriptor(
                0, noiseHistRenderImages[i].imageView);
            vkNoiseHistRenderShaders[i]->bindImageDescriptor(
                1, noiseHistImageStates[cnrLevel].imageView);
            vkNoiseHistRenderShaders[i]->bindStorageBufferDescriptor(
                2, noiseHistMaxBufferStates[cnrLevel].buffer,
                noiseHistMaxBufferStates[cnrLevel].size);
        }
        
    }


    // {
    //     vkImageHistRenderShaders.resize(pyramidLevels);
    //     for (int i = 0; i < pyramidLevels; i++) {
    //         vkImageHistRenderShaders[i] = new VulkanCompute(vkState);

    //         std::vector<VkDescriptorSetLayoutBinding>
    //             descriptorSetLayoutBindings;
    //         descriptorSetLayoutBindings.resize(3);

    //         descriptorSetLayoutBindings[0] = {};
    //         descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
    //         descriptorSetLayoutBindings[0].descriptorType =
    //             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    //         descriptorSetLayoutBindings[0].descriptorCount = 1;
    //         descriptorSetLayoutBindings[0].stageFlags =
    //             VK_SHADER_STAGE_COMPUTE_BIT;

    //         descriptorSetLayoutBindings[1] = {};
    //         descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
    //         descriptorSetLayoutBindings[1].descriptorType =
    //             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    //         descriptorSetLayoutBindings[1].descriptorCount = 1;
    //         descriptorSetLayoutBindings[1].stageFlags =
    //             VK_SHADER_STAGE_COMPUTE_BIT;

    //         descriptorSetLayoutBindings[2] = {};
    //         descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
    //         descriptorSetLayoutBindings[2].descriptorType =
    //             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    //         descriptorSetLayoutBindings[2].descriptorCount = 1;
    //         descriptorSetLayoutBindings[2].stageFlags =
    //             VK_SHADER_STAGE_COMPUTE_BIT;

    //         vkImageHistRenderShaders[i]->init(
    //             "shaders/img_histogram_render.spv", descriptorSetLayoutBindings,
    //             descriptorPool, commandPool);
    //         vkImageHistRenderShaders[i]->bindImageDescriptor(
    //             0, histogramRenderImageStates[i].imageView);
    //         vkImageHistRenderShaders[i]->bindImageDescriptor(
    //             1, noiseHistImageStates[i].imageView);
    //         vkImageHistRenderShaders[i]->bindStorageBufferDescriptor(
    //             2, histogramMaxBufferStates[i].buffer,
    //             histogramMaxBufferStates[i].size);
    //     }
    // }

    {
        vkImgGenerateConstrastCurveShaders.resize(pyramidLevels);
        for (int i = 0; i < pyramidLevels; i++) {
            vkImgGenerateConstrastCurveShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(3);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[2] = {};
            descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
            descriptorSetLayoutBindings[2].descriptorType =
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorSetLayoutBindings[2].descriptorCount = 1;
            descriptorSetLayoutBindings[2].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkImgGenerateConstrastCurveShaders[i]->init(
                "shaders/contrast_curve_generate.spv",
                descriptorSetLayoutBindings, descriptorPool, commandPool);
            vkImgGenerateConstrastCurveShaders[i]->bindStorageBufferDescriptor(
                0, contrastCurveBufferStates[i].buffer,
                contrastCurveBufferStates[i].size);
            vkImgGenerateConstrastCurveShaders[i]->bindStorageBufferDescriptor(
                1, noiseHistMaxBufferStates[i].buffer,
                noiseHistMaxBufferStates[i].size);
            vkImgGenerateConstrastCurveShaders[i]->bindUniformBufferDescriptor(
                2, contrastParametersBufferStates[i].buffer,
                contrastParametersBufferStates[i].size);
        }
    }

    {
        VkRenderConstrastCurveShaders.resize(pyramidLevels);
        for (int i = 0; i < pyramidLevels; i++) {
            VkRenderConstrastCurveShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            VkRenderConstrastCurveShaders[i]->init(
                "shaders/contrast_curve_render.spv",
                descriptorSetLayoutBindings, descriptorPool, commandPool);
            VkRenderConstrastCurveShaders[i]->bindImageDescriptor(
                0, constrastCurveImageStates[i].imageView);
            VkRenderConstrastCurveShaders[i]->bindStorageBufferDescriptor(
                1, contrastCurveBufferStates[i].buffer,
                contrastCurveBufferStates[i].size);
        }
    }

#ifdef GRAD_WITH_LINEAR_IMAGE
    {
        vkImageLinearShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(2);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkImageLinearShader->init("shaders/img_linear.spv",
                                  descriptorSetLayoutBindings, descriptorPool,
                                  commandPool);
        vkImageLinearShader->bindImageDescriptor(0, linearImageState.imageView);
        vkImageLinearShader->bindImageDescriptor(
            1, expandImageStates[pyramidLevels - 1].imageView);
    }
#endif

    {
        vkCnrImageShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(3);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 1
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkCnrImageShader->init("shaders/img_cnr.spv",
                               descriptorSetLayoutBindings, descriptorPool,
                               commandPool);
        vkCnrImageShader->bindImageDescriptor(0, cnrImageState.imageView);
        vkCnrImageShader->bindImageDescriptor(
            1, sdevImageStates[cnrLevel].imageView);
        vkCnrImageShader->bindStorageBufferDescriptor(
            2, noiseHistMaxBufferStates[cnrLevel].buffer,
            noiseHistMaxBufferStates[cnrLevel].size);
    }

    {
        for (uint32_t i = 0; i < cnrLevel; i++) {
            vkBandpassNoiseRedShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding> descrSetLayoutBindings;
            descrSetLayoutBindings.resize(4);

            descrSetLayoutBindings[0] = {};
            descrSetLayoutBindings[0].binding = 0;  // binding = 0
            descrSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descrSetLayoutBindings[0].descriptorCount = 1;
            descrSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descrSetLayoutBindings[1] = {};
            descrSetLayoutBindings[1].binding = 1;  // binding = 1
            descrSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descrSetLayoutBindings[1].descriptorCount = 1;
            descrSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descrSetLayoutBindings[2] = {};
            descrSetLayoutBindings[2].binding = 2;  // binding = 2
            descrSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descrSetLayoutBindings[2].descriptorCount = 1;
            descrSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descrSetLayoutBindings[3] = {};
            descrSetLayoutBindings[3].binding = 3;  // binding = 3
            descrSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descrSetLayoutBindings[3].descriptorCount = 1;
            descrSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            vkBandpassNoiseRedShaders[i]->init(
                "shaders/noise_reduction.spv",
                descrSetLayoutBindings,
                descriptorPool,
                commandPool
            );
            
            vkBandpassNoiseRedShaders[i]->bindImageDescriptor(
                0, expandBandpassNoiseRedImages[i].imageView
            );
            uint32_t l = pyramidLevels - cnrLevel + i;
            vkBandpassNoiseRedShaders[i]->bindImageDescriptor(
                1, expandBandpassImageStates[l].imageView
            );
            vkBandpassNoiseRedShaders[i]->bindImageDescriptor(
                2, cnrImageState.imageView
            );
            vkBandpassNoiseRedShaders[i]->bindUniformBufferDescriptor(
                3, noiseReductionParamsBuffers[cnrLevel - i - 1].buffer, noiseReductionParamsBuffers[i].size
            );
        }
    }
    

#ifdef CNR_DEBUG
    {
        vkCnrDebugImageShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(2);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkCnrImageShader->init("shaders/img_cnr.spv",
                               descriptorSetLayoutBindings, descriptorPool,
                               commandPool);
        vkCnrImageShader->bindImageDescriptor(0, cnrDebugImageState.imageView);
        vkCnrImageShader->bindImageDescriptor(
            1, cnrImageState.imageView);
    }
#endif

    {
        vkRelevantImageShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(3);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 1
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkRelevantImageShader->init("shaders/img_relevant.spv",
                                    descriptorSetLayoutBindings, descriptorPool,
                                    commandPool);
        vkRelevantImageShader->bindImageDescriptor(
            0, relevantImageState.imageView);
        vkRelevantImageShader->bindImageDescriptor(
            1, normalizedImageState.imageView);
        vkRelevantImageShader->bindImageDescriptor(2, cnrImageState.imageView);
    }

    {
        vkGradHistShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(3);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkGradHistShader->init("shaders/gradation_histogram.spv",
                               descriptorSetLayoutBindings, descriptorPool,
                               commandPool);
        vkGradHistShader->bindImageDescriptor(0, gradHistState.imageView);
        vkGradHistShader->bindImageDescriptor(1, relevantImageState.imageView);
#ifdef GRAD_WITH_LINEAR_IMAGE
        vkGradHistShader->bindImageDescriptor(2, linearImageState.imageView);
#else
        vkGradHistShader->bindImageDescriptor(2, expandImageStates[pyramidLevels - 1].imageView);
#endif
    }

    {
        for (uint32_t i = 0; i < 2; i++) {
            vkGradHistToRgbShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(4);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[2] = {};
            descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
            descriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[2].descriptorCount = 1;
            descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[3] = {};
            descriptorSetLayoutBindings[3].binding = 3;  // binding = 3
            descriptorSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[3].descriptorCount = 1;
            descriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            vkGradHistToRgbShaders[i]->init(
                "shaders/gradation_curve_debug_render.spv",
                descriptorSetLayoutBindings,
                descriptorPool,
                commandPool
            );
            vkGradHistToRgbShaders[i]->bindImageDescriptor(0, gradHistImages[i].imageView);

            vkGradHistToRgbShaders[i]->bindImageDescriptor(1, gradHistState.imageView);
            vkGradHistToRgbShaders[i]->bindStorageBufferDescriptor(
                2, gradHistMaxBufferState.buffer, gradHistMaxBufferState.size);
            vkGradHistToRgbShaders[i]->bindStorageBufferDescriptor(
                3, gradCurveBufferState.buffer, gradCurveBufferState.size);
        }
    }

    {
        vkGradHistMaxShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(2);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkGradHistMaxShader->init("shaders/img_histogram_max.spv",
                                  descriptorSetLayoutBindings, descriptorPool,
                                  commandPool);
        vkGradHistMaxShader->bindStorageBufferDescriptor(
            0, gradHistMaxBufferState.buffer, gradHistMaxBufferState.size);
        vkGradHistMaxShader->bindImageDescriptor(1, gradHistState.imageView);
    }

    {
        vkGradCurveGenShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(2);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkGradCurveGenShader->init("shaders/gradation_curve_generate.spv",
                                   descriptorSetLayoutBindings, descriptorPool,
                                   commandPool);
        vkGradCurveGenShader->bindStorageBufferDescriptor(
            0, gradCurveBufferState.buffer, gradCurveBufferState.size);
        vkGradCurveGenShader->bindImageDescriptor(1, gradHistState.imageView);
    }

    {
        vkGradCurveApplyShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(3);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkGradCurveApplyShader->init("shaders/img_apply_gradation_curve.spv",
                                     descriptorSetLayoutBindings,
                                     descriptorPool, commandPool);
        vkGradCurveApplyShader->bindImageDescriptor(
            0, gradedImageState.imageView
        );
#ifdef GRAD_WITH_LINEAR_IMAGE
        vkGradCurveApplyShader->bindImageDescriptor(1, linearImageState.imageView);
#else
        vkGradCurveApplyShader->bindImageDescriptor(1, expandImageStates[pyramidLevels - 1].imageView);
#endif
        // vkGradCurveApplyShader->bindImageDescriptor(1,
        //                                             expandImageStates[pyramidLevels - 1].imageView);
        vkGradCurveApplyShader->bindStorageBufferDescriptor(
            2, gradCurveBufferState.buffer, gradCurveBufferState.size);
    }

    {
        for (int i = 0; i < rgbImagesCount; i++) {
            vkImageToRgbShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

            vkImageToRgbShaders[i]->init(
                "shaders/img_to_rgb.spv",
                descriptorSetLayoutBindings,
                descriptorPool,
                commandPool
            );
            vkImageToRgbShaders[i]->bindImageDescriptor(
                0, rgbImageStates[i].imageView
            );
            VkImageView imageView = VK_NULL_HANDLE;
            switch (i) {
                case 0:
                    imageView = normalizedImageState.imageView;
                    break;
                case 1:
                    imageView = expandImageStates[pyramidLevels - 1].imageView;
                    break;
                case 2:
                    imageView = gradedImageState.imageView;
                    break;

                case 3:
#ifdef ENABLE_CLAHE
                    imageView = claheGradedImageState.imageView;
#endif
                    break;
                case 4:
                    imageView = relevantImageState.imageView;
                    break;
                // case 5:
                //     imageView = linearImageState.imageView;
                //     break;

            }
            if (imageView != VK_NULL_HANDLE)
                vkImageToRgbShaders[i]->bindImageDescriptor(1, imageView);
        }
    }

    {
        for (uint32_t i = 0; i < 2; i++) {
            vkOutImageToRgbShaders[i] = new VulkanCompute(vkState);

            std::vector<VkDescriptorSetLayoutBinding>
                descriptorSetLayoutBindings;
            descriptorSetLayoutBindings.resize(2);

            descriptorSetLayoutBindings[0] = {};
            descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
            descriptorSetLayoutBindings[0].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[0].descriptorCount = 1;
            descriptorSetLayoutBindings[0].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            descriptorSetLayoutBindings[1] = {};
            descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
            descriptorSetLayoutBindings[1].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[1].descriptorCount = 1;
            descriptorSetLayoutBindings[1].stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT;

            vkOutImageToRgbShaders[i]->init(
                "shaders/img_to_rgb.spv",
                descriptorSetLayoutBindings,
                descriptorPool, commandPool
            );
            vkOutImageToRgbShaders[i]->bindImageDescriptor(
                0, outImages[i].imageView);
            vkOutImageToRgbShaders[i]->bindImageDescriptor(
                1, gradedImageState.imageView);
        }
    }

#ifdef ENABLE_CLAHE
    {
        vkClaheHistogramShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(3);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkClaheHistogramShader->init("shaders/clahe_histogram.spv",
                                     descriptorSetLayoutBindings,
                                     descriptorPool, commandPool);
        vkClaheHistogramShader->bindImageDescriptor(
            0, claheHistogramsImageState.imageView);
        vkClaheHistogramShader->bindImageDescriptor(
            1, expandImageStates[pyramidLevels - 1].imageView);
        vkClaheHistogramShader->bindImageDescriptor(
            2, relevantImageState.imageView);
    }

    {
        vkClaheGradCurveShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(2);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkClaheGradCurveShader->init("shaders/clahe_grad_curve.spv",
                                     descriptorSetLayoutBindings,
                                     descriptorPool, commandPool);
        vkClaheGradCurveShader->bindStorageBufferDescriptor(
            0, claheGradCurveBuffer.buffer, claheGradCurveBuffer.size);
        vkClaheGradCurveShader->bindImageDescriptor(
            1, claheHistogramsImageState.imageView);
    }

    {
        vkClaheGradCurveApplyShader = new VulkanCompute(vkState);

        std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
        descriptorSetLayoutBindings.resize(3);

        descriptorSetLayoutBindings[0] = {};
        descriptorSetLayoutBindings[0].binding = 0;  // binding = 0
        descriptorSetLayoutBindings[0].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[0].descriptorCount = 1;
        descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[1] = {};
        descriptorSetLayoutBindings[1].binding = 1;  // binding = 1
        descriptorSetLayoutBindings[1].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorSetLayoutBindings[1].descriptorCount = 1;
        descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2;  // binding = 2
        descriptorSetLayoutBindings[2].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        vkClaheGradCurveApplyShader->init("shaders/clahe_grad_curve_apply.spv",
                                          descriptorSetLayoutBindings,
                                          descriptorPool, commandPool);
        vkClaheGradCurveApplyShader->bindImageDescriptor(
            0, claheGradedImageState.imageView);
        vkClaheGradCurveApplyShader->bindImageDescriptor(
            1, expandImageStates[pyramidLevels - 1].imageView);
        vkClaheGradCurveApplyShader->bindStorageBufferDescriptor(
            2, claheGradCurveBuffer.buffer, claheGradCurveBuffer.size);
    }
#endif
}

VulkanProcessing::VulkanProcessing(VulkanState* vkState) {
    VulkanProcessing::vkState = vkState;
    VulkanProcessing::commandPool = VK_NULL_HANDLE;
    VulkanProcessing::descriptorPool = VK_NULL_HANDLE;
}

bool VulkanProcessing::init(
        uint32_t imageSize,
        std::vector<VkImageView>* outImageViews
    ) {
    VulkanProcessing::imageSize = imageSize;
    VulkanProcessing::pyramidLevels = std::ceil(std::log2(imageSize));

    ASSERT_MSG(createCommandBuffer(), "failed to init command buffer");
    ASSERT_MSG(createDescriptorPool(), "failed to init descriptor pool");
    ASSERT_MSG(initMemory(), "failed to init memory");

    createShaders();

    // fence
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;

    VK_CHECK_RESULT(
        vkCreateFence(vkState->getDevice(), &fenceCreateInfo, nullptr, &fence));

#ifdef MEASURE_PROCESS
    VK_CHECK_RESULT(vkCreateFence(vkState->getDevice(), &fenceCreateInfo, nullptr, &normalizeFence));
    VK_CHECK_RESULT(vkCreateFence(vkState->getDevice(), &fenceCreateInfo, nullptr, &pyramidExpandFence));
    VK_CHECK_RESULT(vkCreateFence(vkState->getDevice(), &fenceCreateInfo, nullptr, &imageAnalysisFence));
    VK_CHECK_RESULT(vkCreateFence(vkState->getDevice(), &fenceCreateInfo, nullptr, &pyramidApplyFence));
    VK_CHECK_RESULT(vkCreateFence(vkState->getDevice(), &fenceCreateInfo, nullptr, &pyramidReduceFence));
    VK_CHECK_RESULT(vkCreateFence(vkState->getDevice(), &fenceCreateInfo, nullptr, &gradationFence));
#endif

    (*outImageViews).resize(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; i++) {
        (*outImageViews)[i] = outImages[i].imageView;
    }

    return true;
}

void VulkanProcessing::clearImage(
    VkImage image,
    VkCommandBuffer commandBuffer
    ) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    //VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Step 1: Transition the image layout to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    // Step 2: Clear the image
    VkClearColorValue clearColor = {};
    clearColor.uint32[0] = 0;

    VkImageSubresourceRange imageSubresourceRange = {};
    imageSubresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageSubresourceRange.baseMipLevel = 0;
    imageSubresourceRange.levelCount = 1;
    imageSubresourceRange.baseArrayLayer = 0;
    imageSubresourceRange.layerCount = 1;

    vkCmdClearColorImage(
        commandBuffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        &clearColor,
        1,
        &imageSubresourceRange
    );

    // Step 3: Transition the image layout back to the required layout
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL; // or any other required layout
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    //VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;  // submit a single command buffer
    submitInfo.pCommandBuffers = &commandBuffer;  // the command buffer to submit

    //VK_CHECK_RESULT(vkQueueSubmit(vkState->getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueSubmit(vkState->getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE);
}

bool VulkanProcessing::execute(const uint16_t* imageData) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    running = true;

    uint8_t backIndex = (currentIndex + 1) % 2;
    
    vkResetFences(vkState->getDevice(), 1, &fence);

#ifdef MEASURE_PROCESS
    vkResetFences(vkState->getDevice(), 1, &normalizeFence);
    vkResetFences(vkState->getDevice(), 1, &pyramidExpandFence);
    vkResetFences(vkState->getDevice(), 1, &imageAnalysisFence);
    vkResetFences(vkState->getDevice(), 1, &pyramidApplyFence);
    vkResetFences(vkState->getDevice(), 1, &pyramidReduceFence);
    vkResetFences(vkState->getDevice(), 1, &gradationFence);
#endif

    memcpy(lastRawImage.data(), imageData, lastRawImage.size());

    //vkQueueWaitIdle(vkState->getComputeQueue());

    vkState->transitionImageLayout(
        vkState->getComputeQueue(),
        outImages[backIndex].image,
        outImages[backIndex].format,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL
    );

#ifdef RENDER_HISTS
    vkState->transitionImageLayout(
        vkState->getComputeQueue(),
        gradHistImages[backIndex].image,
        gradHistImages[backIndex].format,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL
    );

    vkState->transitionImageLayout(
        vkState->getComputeQueue(),
        noiseHistRenderImages[backIndex].image,
        noiseHistRenderImages[backIndex].format,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL
    );
#endif

    // clear things
    for (uint32_t i = 0; i < pyramidLevels; i++) {
        clearImage(
            noiseHistImageStates[i].image,
            clearCommandBuffers[i]
        );
    }
    clearImage(
        gradHistState.image,
        clearCommandBuffers[imageToClear - 1]
    );

    // load image data
    VkDeviceSize pixelsSize = imageSize * imageSize * sizeof(uint16_t);

    vkState->loadDataToImage(
        vkState->getComputeQueue(),
        (void*)imageData, pixelsSize, (uint32_t)imageSize, (uint32_t)imageSize,
        inputImageState.format, &inputImageState.image,
        &inputImageState.imageMemory, VK_IMAGE_LAYOUT_GENERAL
    );

#ifdef MEASURE_PROCESS
    auto initTime = std::chrono::high_resolution_clock::now();
    float initMillis = std::chrono::duration<float, std::chrono::milliseconds::period>(
        initTime - startTime
    ).count();
#endif

    // Sqrt
    vkImageSqrtShader->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    );

    // Max
    for (int i = 0; i < maxReduceImageStates.size(); i++) {
        vkImageMaxReduceShaders[i]->execute(
            (uint32_t)ceil(maxReduceImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(maxReduceImageStates[i].height / float(workgroupSize)),
            i == 0
                ? vkImageSqrtShader->getComputeFinishedSemaphore()
                : vkImageMaxReduceShaders[i - 1]->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE
        );
    }

    // Min
    for (int i = 0; i < minReduceImageStates.size(); i++) {
        vkImageMinReduceShaders[i]->execute(
            (uint32_t)ceil(minReduceImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(minReduceImageStates[i].height / float(workgroupSize)),
            i == 0
                ? vkImageSqrtShader->getComputeFinishedSemaphore()
                : vkImageMinReduceShaders[i - 1]->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE
        );
    }

    vkImageNormalizeShader->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        vkImageMaxReduceShaders[maxReduceImageStates.size() - 1]->getComputeFinishedSemaphore(),
#ifdef MEASURE_PROCESS
        normalizeFence
#else
        VK_NULL_HANDLE
#endif
    );

#ifdef MEASURE_PROCESS
    VK_CHECK_RESULT(vkWaitForFences(vkState->getDevice(), 1, &normalizeFence, VK_TRUE, UINT64_MAX));
    auto normalizedTime = std::chrono::high_resolution_clock::now();
    float normalizeMillis = std::chrono::duration<float, std::chrono::milliseconds::period>(
        normalizedTime - initTime
    ).count();
#endif

    // Pyramid reduce
    for (int i = 0; i < pyramidLevels; i++) {
        vkImageSmoothShaders[i]->execute(
            (uint32_t)ceil(smoothImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(smoothImageStates[i].height / float(workgroupSize)),
            i == 0 ? vkImageNormalizeShader->getComputeFinishedSemaphore()
                   : vkImageDownsampleShaders[i - 1]->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE
        );

        vkImageDownsampleShaders[i]->execute(
            (uint32_t)ceil(downsampledImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(downsampledImageStates[i].height / float(workgroupSize)),
            vkImageSmoothShaders[i]->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE
        );

        vkImageUpsampleShaders[i]->execute(
            (uint32_t)ceil(downsampledImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(downsampledImageStates[i].height / float(workgroupSize)),
            vkImageDownsampleShaders[i]->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE
        );

        vkImageSmoothUpsampledShaders[i]->execute(
            (uint32_t)ceil(lowpassImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(lowpassImageStates[i].height / float(workgroupSize)),
            vkImageUpsampleShaders[i]->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE
        );

        vkImageDifferenceShaders[i]->execute(
            (uint32_t)ceil(bandpassImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(bandpassImageStates[i].height / float(workgroupSize)),
            vkImageSmoothUpsampledShaders[i]->getComputeFinishedSemaphore(),
#ifdef MEASURE_PROCESS
        i == pyramidLevels - 1 ? pyramidReduceFence : VK_NULL_HANDLE
#else
        VK_NULL_HANDLE
#endif
        );
    }

#ifdef MEASURE_PROCESS
    VK_CHECK_RESULT(vkWaitForFences(vkState->getDevice(), 1, &pyramidReduceFence, VK_TRUE, UINT64_MAX));
    auto pyradmiReduceTime = std::chrono::high_resolution_clock::now();
    float pyramidReduceMillis = std::chrono::duration<float, std::chrono::milliseconds::period>(
        pyradmiReduceTime - normalizedTime
    ).count();
#endif

    // Image analysis
    for (int i = 0; i < pyramidLevels; i++) {
        if (i < coarserLevelsStart || i <= cnrLevel) {
            vkImageSDevShaders[i]->execute(                                             // PERF: 1.40ms
                (uint32_t)ceil(sdevImageStates[i].width / float(workgroupSize)),
                (uint32_t)ceil(sdevImageStates[i].height / float(workgroupSize)),
                vkImageDifferenceShaders[i]->getComputeFinishedSemaphore(),
                VK_NULL_HANDLE
            );

            vkNoiseHistShaders[i]->execute(                                             // PERF: 0.60ms
                imageSize / histWorkgroupCoverage,
                imageSize / histWorkgroupCoverage,
                vkImageSDevShaders[i]->getComputeFinishedSemaphore(),
                VK_NULL_HANDLE
            );

            vkNoiseHistMaxShaders[i]->execute(                                          // PERF: 0.30ms
                1, 1, vkNoiseHistShaders[i]->getComputeFinishedSemaphore(),
                VK_NULL_HANDLE
            );
        }

        // vkImageHistRenderShaders[i]->execute(
        //     1, 1, vkNoiseHistMaxShaders[i]->getComputeFinishedSemaphore(),
        //     VK_NULL_HANDLE);
        
        vkImgGenerateConstrastCurveShaders[i]->execute(                             // PERF: 0.10ms
            1, 1,
            i < coarserLevelsStart ?
                vkNoiseHistMaxShaders[i]->getComputeFinishedSemaphore() :
                vkImageDifferenceShaders[i]->getComputeFinishedSemaphore(),
#ifdef MEASURE_PROCESS
        i == pyramidLevels - 1 ? imageAnalysisFence : VK_NULL_HANDLE
#else
        VK_NULL_HANDLE
#endif
        );

//         VkRenderConstrastCurveShaders[i]->execute(                                  // PERF: 0.40ms
//             1, 1,
//             vkImgGenerateConstrastCurveShaders[i]->getComputeFinishedSemaphore(),
// #ifdef MEASURE_PROCESS
//         i == pyramidLevels - 1 ? imageAnalysisFence : VK_NULL_HANDLE
// #else
//         VK_NULL_HANDLE
// #endif
//         );
    }

#ifdef MEASURE_PROCESS
    VK_CHECK_RESULT(vkWaitForFences(vkState->getDevice(), 1, &imageAnalysisFence, VK_TRUE, UINT64_MAX));
    auto imageAnalysisTime = std::chrono::high_resolution_clock::now();
    float imageAnalysisMillis = std::chrono::duration<float, std::chrono::milliseconds::period>(
        imageAnalysisTime - pyradmiReduceTime
    ).count();
#endif


    // vkNoiseHistRenderShader->execute(
    //     1, 1, vkNoiseHistMaxShaders[cnrLevel]->getComputeFinishedSemaphore(),
    //     VK_NULL_HANDLE);

#ifdef RENDER_HISTS
    vkNoiseHistRenderShaders[backIndex]->execute(
        1, 1, vkNoiseHistMaxShaders[cnrLevel]->getComputeFinishedSemaphore(),
        VK_NULL_HANDLE);
#endif

    // cnr
    vkCnrImageShader->execute(
        (uint32_t)ceil(cnrImageState.width / float(workgroupSize)),
        (uint32_t)ceil(cnrImageState.height / float(workgroupSize)),
        vkNoiseHistMaxShaders[cnrLevel]->getComputeFinishedSemaphore(),
        VK_NULL_HANDLE);


    // apply contrast curve
    for (int i = 0; i < pyramidLevels; i++) {
        vkImageApplyContrastCurveShaders[i]->execute(
            (uint32_t)ceil(expandBandpassImageStates[i].width /
                           float(workgroupSize)),
            (uint32_t)ceil(expandBandpassImageStates[i].height /
                           float(workgroupSize)),
            vkImgGenerateConstrastCurveShaders[pyramidLevels - i - 1]
                ->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE);
    }

    // noise reduction
    for (uint32_t i = 0; i < cnrLevel; i++) {
        uint32_t l = pyramidLevels - cnrLevel + i;
        vkBandpassNoiseRedShaders[i]->execute(
            (uint32_t)ceil(expandBandpassNoiseRedImages[i].width / float(workgroupSize)),
            (uint32_t)ceil(expandBandpassNoiseRedImages[i].height / float(workgroupSize)),
            vkImageApplyContrastCurveShaders[l]->getComputeFinishedSemaphore(),
#ifdef MEASURE_PROCESS
        i == cnrLevel - 1 ? pyramidApplyFence : VK_NULL_HANDLE
#else
        VK_NULL_HANDLE
#endif
        );
    }

#ifdef MEASURE_PROCESS
    VK_CHECK_RESULT(vkWaitForFences(vkState->getDevice(), 1, &imageAnalysisFence, VK_TRUE, UINT64_MAX));
    auto pyramidApplyTime = std::chrono::high_resolution_clock::now();
    float pyramidApplyMillis = std::chrono::duration<float, std::chrono::milliseconds::period>(
        pyramidApplyTime - imageAnalysisTime
    ).count();
#endif

    // pyramid expand
    for (int i = 0; i < pyramidLevels; i++) {
        vkImageExpandUpsampleShaders[i]->execute(
            (uint32_t)ceil(expandUpsampledImageStates[i].width /
                           float(workgroupSize)),
            (uint32_t)ceil(expandUpsampledImageStates[i].height /
                           float(workgroupSize)),
            i == 0 ? vkImageDownsampleShaders[pyramidLevels - 1]
                         ->getComputeFinishedSemaphore()
                   : vkImageExpandAdditionShaders[i - 1]
                         ->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE);

        vkImageExpandLowpassShaders[i]->execute(
            (uint32_t)ceil(expandLowpassImageStates[i].width /
                           float(workgroupSize)),
            (uint32_t)ceil(expandLowpassImageStates[i].height /
                           float(workgroupSize)),
            vkImageExpandUpsampleShaders[i]->getComputeFinishedSemaphore(),
            VK_NULL_HANDLE);

        uint32_t currentLevel = pyramidLevels - i - 1;
        uint32_t noiseRedPos = i - (pyramidLevels - cnrLevel);

        vkImageExpandAdditionShaders[i]->execute(
            (uint32_t)ceil(expandImageStates[i].width / float(workgroupSize)),
            (uint32_t)ceil(expandImageStates[i].height / float(workgroupSize)),
            currentLevel < cnrLevel - 1 ?
                vkBandpassNoiseRedShaders[currentLevel]->getComputeFinishedSemaphore():
                vkImageApplyContrastCurveShaders[currentLevel]->getComputeFinishedSemaphore(),
#ifdef MEASURE_PROCESS
            i == 0 ? pyramidExpandFence : VK_NULL_HANDLE
#else
            VK_NULL_HANDLE
#endif
        );
    }

#ifdef MEASURE_PROCESS
    VK_CHECK_RESULT(vkWaitForFences(vkState->getDevice(), 1, &pyramidExpandFence, VK_TRUE, UINT64_MAX));
    auto pyramidExpandTime = std::chrono::high_resolution_clock::now();
    float pyramidExpandMillis = std::chrono::duration<float, std::chrono::milliseconds::period>(
        pyramidExpandTime - pyramidApplyTime
    ).count();
#endif

    // linear
#ifdef GRAD_WITH_LINEAR_IMAGE
    vkImageLinearShader->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        vkImageExpandAdditionShaders[0]->getComputeFinishedSemaphore(),
#ifdef MEASURE_PROCESS
        gradationFence
#else
        VK_NULL_HANDLE
#endif
    );
#endif

    // relevant image
    vkRelevantImageShader->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
#ifdef GRAD_WITH_LINEAR_IMAGE
        vkImageLinearShader->getComputeFinishedSemaphore(),
#else
        vkImageExpandAdditionShaders[0]->getComputeFinishedSemaphore(),
#endif
#ifdef MEASURE_PROCESS
        gradationFence
#else
        VK_NULL_HANDLE
#endif
    );

#ifdef ENABLE_CLAHE
    // clahe histogram
    uint32_t tilesCount = (uint32_t)ceil(imageSize / float(workgroupSize));
    vkClaheHistogramShader->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        vkRelevantImageShader->getComputeFinishedSemaphore(), VK_NULL_HANDLE);

    // clahe grad curve
    vkClaheGradCurveShader->execute(
        1, 1, vkClaheHistogramShader->getComputeFinishedSemaphore(),
        VK_NULL_HANDLE);

    // clahe grad curve apply
    vkClaheGradCurveApplyShader->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        vkClaheGradCurveShader->getComputeFinishedSemaphore(), VK_NULL_HANDLE);
#endif

    // old gradation
    vkGradHistShader->execute(
        (uint32_t)ceil(imageSize / float(histWorkgroupCoverage)),
        (uint32_t)ceil(imageSize / float(histWorkgroupCoverage)),
        vkRelevantImageShader->getComputeFinishedSemaphore(),
        VK_NULL_HANDLE
    );

    vkGradHistMaxShader->execute(
        1, 1, vkGradHistShader->getComputeFinishedSemaphore(), VK_NULL_HANDLE
    );

    vkGradCurveGenShader->execute(
        1, 1, vkGradHistMaxShader->getComputeFinishedSemaphore(), VK_NULL_HANDLE
    );

#ifdef RENDER_HISTS
    vkGradHistToRgbShaders[backIndex]->execute(
        1, 1, vkGradCurveGenShader->getComputeFinishedSemaphore(), VK_NULL_HANDLE
    );
#endif

    vkGradCurveApplyShader->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        vkGradCurveGenShader->getComputeFinishedSemaphore(),
        VK_NULL_HANDLE
    );

#ifdef MEASURE_PROCESS
    VK_CHECK_RESULT(vkWaitForFences(vkState->getDevice(), 1, &gradationFence, VK_TRUE, UINT64_MAX));
    auto gradationTime = std::chrono::high_resolution_clock::now();
    float gradationMillis = std::chrono::duration<float, std::chrono::milliseconds::period>(
        gradationTime - pyramidExpandTime
    ).count();
#endif

    vkOutImageToRgbShaders[backIndex]->execute(
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        (uint32_t)ceil(imageSize / float(workgroupSize)),
        vkGradCurveApplyShader->getComputeFinishedSemaphore(),
        fence
    );

    VK_CHECK_RESULT(
        vkWaitForFences(vkState->getDevice(), 1, &fence, VK_TRUE, UINT64_MAX));

    vkState->transitionImageLayout(
        vkState->getComputeQueue(),
        outImages[backIndex].image,
        outImages[backIndex].format,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

#ifdef RENDER_HISTS
    vkState->transitionImageLayout(
        vkState->getComputeQueue(),
        gradHistImages[backIndex].image,
        gradHistImages[backIndex].format,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    vkState->transitionImageLayout(
        vkState->getComputeQueue(),
        noiseHistRenderImages[backIndex].image,
        noiseHistRenderImages[backIndex].format,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );
#endif

    currentIndex = backIndex;

#ifdef ENABLE_CLAHE
    vkState->transitionImageLayout(
        claheGradedImageState.image, claheGradedImageState.format,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
#endif

    // auto currentTime = std::chrono::high_resolution_clock::now();
    // float time =
    //     std::chrono::duration<float, std::chrono::milliseconds::period>(
    //         currentTime - startTime)
    //         .count();

    // printf("processing time: %fms\n", time);

#ifdef MEASURE_PROCESS
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::milliseconds::period>(
            currentTime - startTime
        ).count();
    printf(
        "init: %.2f \t norm: %.2f \t red: %.2f \t anly: %.2f \t aply: %.2f \t exp: %.2f \t grad: %.2f \t tot: %.2f \n",
        initMillis,
        normalizeMillis,
        pyramidReduceMillis,
        imageAnalysisMillis,
        pyramidApplyMillis,
        pyramidExpandMillis,
        gradationMillis,
        time
    );
#endif

    running = false;

    return true;
}

bool VulkanProcessing::saveOutImage(std::string filePath) {
    VkImageState image = gradedImageState;
    float maxValue = 1.0f;
    float minValue = 0.0f;
    uint32_t margin = 10;       // TODO: this should be a parameter of the process, the out image in vulkan should be cropped

    std::vector<float> dstBuffer;
    dstBuffer.resize(image.width * image.height * sizeof(float));

    vkState->loadDataFromImage(
        vkState->getComputeQueue(),
        dstBuffer.data(),
        dstBuffer.size(),
        &(image.image),
        image.width,
        image.height,
        image.format,
        VK_IMAGE_LAYOUT_GENERAL
    );

    std::vector<uint8_t> outBuffer;
    uint32_t newWidth = image.width - 2 * margin;
    uint32_t newHeight = image.height - 2 * margin;
    outBuffer.resize(newWidth * newHeight * sizeof(uint8_t));

    for (uint32_t y = 0; y < newHeight; y++) {
        for (uint32_t x = 0; x < newWidth; x++) {
            uint32_t bufferIndex = (y + margin) * image.height + x + margin;
            outBuffer[y * newHeight + x] = 
                (uint8_t)(255.0f * (dstBuffer[bufferIndex] - minValue) / (maxValue - minValue));
        }    
    }

    ASSERT_MSG(stbi_write_bmp(
        filePath.c_str(),
        newWidth,
        newHeight,
        1,
        outBuffer.data()
    ) != 0, "failed to write out file");

    return true;
}

bool VulkanProcessing::cleanup() {
    // sync
    vkDestroyFence(vkState->getDevice(), fence, NULL);
    return true;
}

ImTextureID VulkanProcessing::getGradationHistogram() {
    return gradHistDescriptorSet[currentIndex];
}

ImTextureID VulkanProcessing::getNoiseHistogram() {
    return noiseHistDescriptorSet[currentIndex];
}

bool VulkanProcessing::debugProcess() {
    if (running) return false;

    vkState->downloadAndSaveImage(
        vkState->getComputeQueue(),
        "norm.bmp",
        &normalizedImageState,
        VK_IMAGE_LAYOUT_GENERAL,
        1.0f,
        0.0f
    );

    for (uint32_t i = 0; i < pyramidLevels; i++) {
        vkState->downloadAndSaveImage(
            vkState->getComputeQueue(),
            std::string("red_bandpass_" + std::to_string(i) + ".bmp"),
            &bandpassImageStates[i],
            VK_IMAGE_LAYOUT_GENERAL,
            1.0f,
            -1.0f
        );
        vkState->downloadAndSaveImage(
            vkState->getComputeQueue(),
            std::string("red_lowpass_" + std::to_string(i) + ".bmp"),
            &lowpassImageStates[i],
            VK_IMAGE_LAYOUT_GENERAL,
            1.0f,
            0.0f
        );
    }

    vkState->downloadAndSaveImage(
        vkState->getComputeQueue(),
        "sdev.bmp",
        &sdevImageStates[cnrLevel],
        VK_IMAGE_LAYOUT_GENERAL,
        1.0f,
        -1.0f
    );

    vkState->downloadAndSaveImage(
        vkState->getComputeQueue(),
        "cnr.bmp",
        &cnrImageState,
        VK_IMAGE_LAYOUT_GENERAL,
        1.0f,
        0.0f
    );

    for (uint32_t i = 0; i < pyramidLevels; i++) {
        vkState->downloadAndSaveImage(
            vkState->getComputeQueue(),
            std::string("exp_bandpass_" + std::to_string(i) + ".bmp"),
            &expandBandpassImageStates[i],
            VK_IMAGE_LAYOUT_GENERAL,
            1.0f,
            -1.0f
        );
        vkState->downloadAndSaveImage(
            vkState->getComputeQueue(),
            std::string("exp_lowpass_" + std::to_string(i) + ".bmp"),
            &expandLowpassImageStates[i],
            VK_IMAGE_LAYOUT_GENERAL,
            1.0f,
            0.0f
        );
    }

    vkState->downloadAndSaveImage(
        vkState->getComputeQueue(),
        "relevant.bmp",
        &relevantImageState,
        VK_IMAGE_LAYOUT_GENERAL,
        1.0f,
        0.0f
    );

#ifdef GRAD_WITH_LINEAR_IMAGE
    vkState->downloadAndSaveImage(
        vkState->getComputeQueue(),
        "linear.bmp",
        &linearImageState,
        VK_IMAGE_LAYOUT_GENERAL,
        1.0f,
        0.0f
    );
#endif

    vkState->downloadAndSaveImage(
        vkState->getComputeQueue(),
        "graded.bmp",
        &gradedImageState,
        VK_IMAGE_LAYOUT_GENERAL,
        1.0f,
        0.0f
    );

    {
        VkImageState image = noiseHistRenderImages[currentIndex];
        std::vector<float> dstBuffer;
        dstBuffer.resize(image.width * image.height * sizeof(uint8_t) * 4);

        vkState->loadDataFromImage(
            vkState->getComputeQueue(),
            dstBuffer.data(),
            dstBuffer.size(),
            &(image.image),
            image.width,
            image.height,
            image.format,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        ASSERT_MSG(stbi_write_bmp(
            "noise_hist.bmp",
            image.width,
            image.height,
            4,
            dstBuffer.data()
        ) != 0, "failed to write out file");
    }

    {
        VkImageState image = gradHistImages[currentIndex];
        std::vector<float> dstBuffer;
        dstBuffer.resize(image.width * image.height * sizeof(uint8_t) * 4);

        vkState->loadDataFromImage(
            vkState->getComputeQueue(),
            dstBuffer.data(),
            dstBuffer.size(),
            &(image.image),
            image.width,
            image.height,
            image.format,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        ASSERT_MSG(stbi_write_bmp(
            "grad_hist.bmp",
            image.width,
            image.height,
            4,
            dstBuffer.data()
        ) != 0, "failed to write out file");
    }

    return true;
}

bool VulkanProcessing::saveLastRawImage() {
    if (running) return false;
    writeFile("in.raw", (uint8_t*)lastRawImage.data(), lastRawImage.size());
    return true;
}