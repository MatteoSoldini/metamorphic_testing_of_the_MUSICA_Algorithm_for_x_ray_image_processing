#pragma once

#include <vector>
#include <GLFW/glfw3.h>
#include <optional>
#include <vulkan/vulkan_core.h>

#include <assert.h>


#define VK_CHECK_RESULT(f) 																				\
{                                                                                                       \
    VkResult res = (f);																					\
    if (res != VK_SUCCESS) {																            \
        printf("Fatal: VkResult is %d\n", res);                                                         \
        return false;                                                                                   \
    }																									\
}

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface);

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window);

std::optional<uint32_t> queryGraphicsQueueFamily(VkPhysicalDevice physicalDevice);
std::optional<uint32_t> queryPresentQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);
std::optional<uint32_t> queryVideoQueueFamily(VkPhysicalDevice physicalDevice);
std::optional<uint32_t> queryComputeQueueFamily(VkPhysicalDevice physicalDevice);

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
uint32_t findGenericMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter);