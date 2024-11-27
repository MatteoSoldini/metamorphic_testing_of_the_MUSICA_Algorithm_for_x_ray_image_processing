#pragma once
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "vk_state.h"
#include "vk_window.h"

class VulkanImGui {
private:
	VulkanState* vkState;
	VkRenderPass renderPass;

	int framesInFlight;
	GLFWwindow* window;

	VkDescriptorPool descriptorPool;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
public:
	VulkanImGui(VulkanState* vkState, GLFWwindow* window, const int framesInFlight, VkRenderPass renderPass);
	
	bool init();

	VkCommandBuffer recordFrame(uint32_t currentFrame, VkFramebuffer frameBuffer, VkExtent2D frameBufferExtent);
	
	void cleanup();
};