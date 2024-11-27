#pragma once

#include <assert.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <mutex>

#define VK_CHECK_RESULT(f)                                                     \
    {                                                                          \
        VkResult res = (f);                                                    \
        if (res != VK_SUCCESS) {                                               \
            printf("Fatal : VkResult is %d in %s at line %d\n", res, __FILE__, \
                   __LINE__);                                                  \
            assert(res == VK_SUCCESS);                                         \
        }                                                                      \
    }

struct VkImageState {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMemory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageView imageView = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
};

struct VkBufferState {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
};

class VulkanState {
private:
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation",
    };

    std::vector<const char*> instanceExtensions = {
        // VK_EXT_DEBUG_UTILS_EXTENSION_NAME,         // shader debug
        // VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME  // shader debug
    };

    const std::vector<VkValidationFeatureEnableEXT> validationFeatures = {
        // VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT
    };

    std::vector<const char*> deviceExtensions = {
        //VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,  // shader debug
    };

    // device
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    // queues
    uint32_t graphicsFamily;
    VkQueue graphicsQueue;  // should be unique
    uint32_t computeFamily;
    VkQueue computeQueue;   // should be unique

    // commands
    VkCommandPool commandPool;

    // descriptor pool
    VkDescriptorPool descriptorPool;

    // samplers
    VkSampler textureSampler;
    VkSampler debugSampler;  // no interpolation

    bool checkValidationLayerSupport();
    VkPhysicalDevice pickPhysicalDevice();
    bool createSamplers();

    // images

    // single time command
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkQueue queue, VkCommandBuffer commandBuffer);

public:
    bool init(std::vector<const char*> instanceExtensions);

    VkPhysicalDevice getPhysicalDevice() { return physicalDevice; };
    VkDevice getDevice() { return device; };
    VkInstance getInstance() { return instance; };
    uint32_t getGraphicsFamily() { return graphicsFamily; };
    VkQueue getGraphicsQueue() { return graphicsQueue; };
    VkQueue getComputeQueue() { return computeQueue; };
    VkSampler getTextureSampler() { return textureSampler; };
    VkSampler getDebugSampler() { return debugSampler; };

    // buffers
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory);
    bool createBuffer2(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags properties,
                       VkBufferState* bufferState);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void loadDataToBuffer(void* data, VkDeviceSize dataSize,
                          VkBuffer destBuffer);
    void copyBufferToImage(VkQueue queue, VkBuffer buffer, VkImage image, uint32_t width,
                           uint32_t height);

    // images
    void transitionImageLayout(VkQueue queue, VkImage image, VkFormat format,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout);
    void loadDataToImage(VkQueue queue, void* pixels, VkDeviceSize pixelsSize, uint32_t width,
                         uint32_t height, VkFormat imageFormat,
                         VkImage* destImage, VkDeviceMemory* destImageMemory,
                         VkImageLayout destImageLayout);
    bool createImage(uint32_t width, uint32_t height, VkFormat format,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImageType imageType,
                     VkImage* image, VkDeviceMemory* imageMemory);

    bool createImage2(VkQueue queue, uint32_t width, uint32_t height, VkFormat format,
                      VkImageTiling tiling, VkImageUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      VkImageLayout finalImageLayout, VkImageState* imageState);

    bool create1DImage2(VkQueue queue, uint32_t width, VkFormat format, VkImageTiling tiling,
                        VkImageUsageFlags usage,
                        VkMemoryPropertyFlags properties,
                        VkImageLayout finalImageLayout,
                        VkImageState* imageState);

    bool create3DImage2(VkQueue queue, uint32_t width, uint32_t height, uint32_t depth,
                        VkFormat format, VkImageTiling tiling,
                        VkImageUsageFlags usage,
                        VkMemoryPropertyFlags properties,
                        VkImageLayout finalImageLayout,
                        VkImageState* imageState);

    bool createImageView(VkImage image, VkFormat format, VkImageView* imageView,
                         VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D);

    void loadDataFromImage(VkQueue queue, void* pixels, VkDeviceSize pixelsSize,
                           VkImage* srcImage, uint32_t width, uint32_t height,
                           VkFormat imageFormat, VkImageLayout imageLayout);
    void copyImageToBuffer(VkQueue queue, VkBuffer buffer, VkImage image, uint32_t width,
                           uint32_t height);

    bool downloadAndSaveImage(
        VkQueue queue,
        std::string filePath,
        VkImageState* image,
        VkImageLayout imageLayout,
        float maxValue,
        float minValue
    );

    // cleanup
    void cleanup();
};