#pragma once
#include <string>
#include "vk_state.h"

namespace Image{
	bool read(
        std::string path,
        VulkanState* vkState,
        VkCommandBuffer commandBuffer,
        VkImage textureImage,
        VkDeviceMemory textureImageMemory
    );
}