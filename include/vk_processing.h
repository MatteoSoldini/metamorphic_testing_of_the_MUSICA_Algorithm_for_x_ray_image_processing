#pragma once

#include <vulkan/vulkan.h>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <array>

#include "vk_compute.h"
#include "vk_state.h"

// #define ENABLE_CLAHE

// #define CNR_DEBUG

// #define LINEAR_LOW_CONTRAST_LEVELS_REDUCTION
// #define LINEAR_HIGH_CONTRAST_LEVELS_REDUCTION

#define MEASURE_PROCESS

#define RENDER_HISTS

//#define GRAD_WITH_LINEAR_IMAGE

class VulkanProcessing {
public:
    static const uint32_t coarserLevelsStart = 3;       // first coarder level (inclusive)
    static const uint32_t cnrLevel = 3;                 // 3 -> most reliable for grading less with noise,
                                                        // 2 -> most reliable in general
    static const uint32_t histRenderWidth = 512;        // also curves
    static const uint32_t histRenderHeight = 128;       // also curves
    static const uint32_t framesInFlight = 2;

private:
    static const uint32_t workgroupSize = 32;
    static const uint32_t noiseHistogramBins = 2048;
    static const uint32_t gradHistogramBins = 1024;
    static const uint32_t histogramAreaSize = 16;
    static const uint32_t histWorkgroupCoverage = workgroupSize * histogramAreaSize;
    static const uint32_t reduceAreaSize = 8;           // ONCHANGE: change in the shader
    
    const float nrHighCnr = 9.0f;                       // noise reduction
    const float nrMaxHighFactor = 1.2f;                 // ...
    const float nrLowCnr = 3.0f;                        // ...
    const float nrMinLowFactor = 0.6f;                  //
    
    const float highContrastMaxReduction = 0.2f;        // contrast enhancement
    const float lowContrastMaxEnhancment = 3.0f;        // ...


    bool running = false;
    uint32_t pyramidLevels;

    uint32_t imageSize;
    std::vector<uint16_t> lastRawImage;                 // keep last raw around

    VulkanState* vkState;

    VkDescriptorPool descriptorPool;
    bool createDescriptorPool();

    // commands
    VkCommandPool commandPool;
    
    // TODO: fix
    static const uint32_t imageToClear = 12 + 1;                    // noise hist + grad hist
    std::array<VkCommandBuffer, imageToClear> clearCommandBuffers;  // for clearing hist images
    bool createCommandBuffer();

    // input
    VkImageState inputImageState;

    // sqrt
    VkImageState sqrtImageState;                        // sqrt image (noise normalized)
    VulkanCompute* vkImageSqrtShader;

    // image max
    std::vector<VkImageState> maxReduceImageStates;     // max reduce images
    std::vector<VulkanCompute*> vkImageMaxReduceShaders;
    std::vector<VkImageState> minReduceImageStates;     // min reduce images
    std::vector<VulkanCompute*> vkImageMinReduceShaders;

    // normalize
    VkImageState normalizedImageState;                  // normalized image
    VulkanCompute* vkImageNormalizeShader;

    // pyramid images (reduce)
    std::vector<VkImageState> bandpassImageStates;      // bandpass images
    std::vector<VkImageState> downsampledImageStates;   // downsampled images
    std::vector<VkImageState> smoothImageStates;        // smoothed images
    std::vector<VkImageState> upsampledImageStates;     // upsampled images
    std::vector<VkImageState> lowpassImageStates;       // upsampled + smoothed = lowpass images

    // noise histogram
    uint32_t histCount = pow(imageSize / histWorkgroupCoverage, 2);
    std::vector<VkImageState> noiseHistImageStates;
    std::vector<VulkanCompute*> vkNoiseHistShaders;

    // LIVE: (cnr level) noise hist render
    VkImageState noiseHistRenderImage;

#ifdef RENDER_HISTS
    VulkanCompute* vkNoiseHistRenderShader;
#endif

    std::array<VkImageState, 2> noiseHistRenderImages;
    std::array<VkDescriptorSet, 2> noiseHistDescriptorSet;
    std::array<VulkanCompute*, 2> vkNoiseHistRenderShaders;

    // noise reduction
    struct NoiseReductionParams {
        float lowCnr;
        float lowFactor;
        float highCnr;
        float highFactor;
    };
    std::array<VkImageState, cnrLevel> expandBandpassNoiseRedImages;        // [cnrLevel, 0]
    std::array<VulkanCompute*, cnrLevel> vkBandpassNoiseRedShaders;
    std::array<VkBufferState, cnrLevel> noiseReductionParamsBuffers;

    // histogram max value
    struct HistogramMaxPoint {
        uint32_t maxValue;
        uint32_t maxBin;
    };
    std::vector<VkBufferState> noiseHistMaxBufferStates;
    std::vector<VulkanCompute*> vkNoiseHistMaxShaders;

    // noise histogram render
    // std::vector<VkImageState> histogramRenderImageStates;
    // std::vector<VulkanCompute*> vkImageHistRenderShaders;

    // contrast curve
    struct ContrastParameters {
        float lowContrastFactor;
        float highContrastFactor;
    };
    std::vector<VkBufferState> contrastParametersBufferStates;
    
    static struct ContrastPoint {
        float x;
        float y;
    };
    static const uint32_t maxContrastPoints = 256;
    static struct ContrastCurveObj {
        ContrastPoint points[maxContrastPoints];
        uint32_t pointsCount;
    };
    std::vector<VkBufferState> contrastCurveBufferStates;
    std::vector<VulkanCompute*>
        vkImgGenerateConstrastCurveShaders;

    // constrast curve render
    std::vector<VkImageState> constrastCurveImageStates;
    std::vector<VulkanCompute*> VkRenderConstrastCurveShaders;

    // apply contrast curve
    std::vector<VkImageState> expandBandpassImageStates;
    std::vector<VulkanCompute*> vkImageApplyContrastCurveShaders;

    // pyramid images (expand)
    std::vector<VkImageState> expandImageStates;              // reconstructed images
    std::vector<VkImageState> expandUpsampledImageStates;     // upsampled image
    std::vector<VkImageState> expandLowpassImageStates;       // lowpass images

    std::vector<VkImageState> sdevImageStates;                // standard deviation images
    std::vector<VulkanCompute*> vkImageSDevShaders;

#ifdef ENABLE_CLAHE
    // clahe histograms
    static const uint32_t claheHistogramBins = 256;
    static const uint32_t claheTiles = 4;  // ONCHANGE: change in the shader
    VkImageState claheHistogramsImageState;
    VulkanCompute* vkClaheHistogramShader;

    // clahe gradation curve
    VkBufferState claheGradCurveBuffer;
    VulkanCompute* vkClaheGradCurveShader;

    // clahe gradation curve apply
    VkImageState claheGradedImageState;
    VulkanCompute* vkClaheGradCurveApplyShader;
#endif

    static struct GradPoint {
        float x;
        float y;
    };

    // cnr
    VkImageState cnrImageState;
    VulkanCompute* vkCnrImageShader;

#ifdef CNR_DEBUG
    // cnr debug
    VkImageState cnrDebugImageState;
    VulkanCompute* vkCnrDebugImageShader;
#endif

    // relevant image
    VkImageState relevantImageState;
    VulkanCompute* vkRelevantImageShader;

    // gradation histogram
    VulkanCompute* vkGradHistShader;
    VkImageState gradHistState;

    // gradation histogram max
    VkBufferState gradHistMaxBufferState;
    VulkanCompute* vkGradHistMaxShader;

    // generate gradation curve
    static const uint32_t maxGradPoints = 256;
    static struct GradCurveObj {
        GradPoint points[maxGradPoints];
        uint32_t pointsCount;

        float t0;
        float ta;
        float t1;
    };
    VkBufferState gradCurveBufferState;
    VulkanCompute* vkGradCurveGenShader;

    // apply gradation curve
    VkImageState gradedImageState;
    VulkanCompute* vkGradCurveApplyShader;

    // linear
#ifdef GRAD_WITH_LINEAR_IMAGE
    VulkanCompute* vkImageLinearShader;
    VkImageState linearImageState;
#endif

    // r to rgb
    static const uint32_t rgbImagesCount = 6;
    std::array<VulkanCompute*, rgbImagesCount> vkImageToRgbShaders;
    std::array<VkImageState, rgbImagesCount> rgbImageStates;

    bool initMemory();

    // shaders (pyramid)
    std::vector<VulkanCompute*> vkImageSmoothShaders;
    std::vector<VulkanCompute*> vkImageDifferenceShaders;
    std::vector<VulkanCompute*> vkImageSmoothUpsampledShaders;
    std::vector<VulkanCompute*> vkImageDownsampleShaders;
    std::vector<VulkanCompute*> vkImageUpsampleShaders;
    std::vector<VulkanCompute*> vkImageExpandUpsampleShaders;
    std::vector<VulkanCompute*> vkImageExpandLowpassShaders;
    std::vector<VulkanCompute*> vkImageExpandAdditionShaders;

    // LIVE: out image
    uint8_t currentIndex = 0;   // 0 or 1
    std::array<VkImageState, framesInFlight> outImages;
    std::array<VulkanCompute*, framesInFlight> vkOutImageToRgbShaders;

    // LIVE: gradation histogram and curver
    std::array<VkImageState, framesInFlight> gradHistImages;
    std::array<VkDescriptorSet, framesInFlight> gradHistDescriptorSet;
    std::array<VulkanCompute*, framesInFlight> vkGradHistToRgbShaders;

    void createShaders();

    // sync
    VkFence fence;

#ifdef MEASURE_PROCESS
    VkFence normalizeFence;
    VkFence pyramidExpandFence;
    VkFence imageAnalysisFence;
    VkFence pyramidApplyFence;
    VkFence pyramidReduceFence;
    VkFence gradationFence;

#endif

    void clearImage(VkImage image, VkCommandBuffer commandBuffer);

public:
    VulkanProcessing(VulkanState* vkState);

    bool init(
        uint32_t imageSize,
        std::vector<VkImageView>* outImageViews
    );

    bool execute(const uint16_t* imageData);

    bool cleanup();

    uint32_t getCurrentIndex() {
        return currentIndex;
    }

    ImTextureID getGradationHistogram();

    ImTextureID getNoiseHistogram();

//     VkImageState* getInImage() { return &rgbImageStates[0]; };
//     VkImageState* getContrastEnhancedImage() { return &rgbImageStates[1]; };
//     VkImageState* getOutImage() {
//         return &rgbImageStates[2];
//     };  // old gradation
// #ifdef ENABLE_CLAHE
//     VkImageState* getClaheGradedImage() { return &rgbImageStates[3]; };
// #endif
//     VkImageState* getRelevantImage() { return &rgbImageStates[4]; };

//     VkImageState* getGradationHistogram() { return &gradHistRenderImageState; };

//     VkImageState* getGradationCurve() { return &gradCurveRenderImageState; };
//     VkImageState* getGradedImage() { return &gradedImageState; };

//     VkImageState* getNoiseHistogram() { return &noiseHistRenderImage; };

//     std::array<VkImageState, pyramidLevels> getConstrastCurveImageStates() {
//         return constrastCurveImageStates;
//     };
//     std::array<VkImageState, pyramidLevels> getReduceBandpassImageStates() {
//         return bandpassImageStates;
//     };
//     std::array<VkImageState, pyramidLevels> getReduceDownsampledImageStates() {
//         return downsampledImageStates;
//     };
//     std::array<VkImageState, pyramidLevels> getReduceLowpassImageStates() {
//         return lowpassImageStates;
//     };
//     std::array<VkImageState, pyramidLevels> getExpandBandpassImageStates() {
//         return expandBandpassImageStates;
//     };
//     std::array<VkImageState, pyramidLevels> getExpandImageStates() {
//         return expandImageStates;
//     };
//     std::array<VkImageState, pyramidLevels> getExpandLowpassImageStates() {
//         return expandLowpassImageStates;
//     };
//     std::array<VkImageState, pyramidLevels> getSdevImageStates() {
//         return sdevImageStates;
//     };
//     std::array<VkImageState, cnrLevel> getExpandBandpassNoiseRedImages() {
//         return expandBandpassNoiseRedImages;
//     };

//     VkImageState* getCnrImage() { return &cnrImageState; };

//     VkImageState* getLinearImage() { return &rgbImageStates[5]; };

    bool debugProcess();

    bool saveLastRawImage();

    uint32_t getImageSize() { return imageSize; };

    bool saveOutImage(std::string filePath);
};