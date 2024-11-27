#include "../include/image.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "../include/vk_utils.h"

bool Image::read(
    std::string path,
    VulkanState* vkState,
    VkCommandBuffer commandBuffer,
    VkImage textureImage,
    VkDeviceMemory textureImageMemory
) {
    int width, height, channels;
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);

    if (!pixels) {
        return false;
    }
    
    // TODO: load

    // free data
    stbi_image_free(pixels);

    return true;
}
