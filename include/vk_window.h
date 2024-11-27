#pragma once
#include <vector>
#include "vk_state.h"
#include <GLFW/glfw3.h>

class VulkanWindow {
private:
	VulkanState* vkState;
	VkSurfaceKHR surface;
	GLFWwindow* window;

	// swapchain
	VkRenderPass renderPass;
	std::vector<VkFramebuffer> frameBuffers;
	std::vector<VkImage> swapChainImages;
	std::vector<VkImageView> swapChainImageViews;
	VkSwapchainKHR swapChain;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;

	VkQueue presentQueue;

	uint32_t framesInFlight;
	uint32_t currentFrame;

	// sync
	uint32_t imageIndex;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;

	bool createSwapChain();
	void cleanupSwapChain();

	void windowResize();
	
	bool createFrameBuffers();
	void cleanupFrameBuffers();

public:
	VulkanWindow(VulkanState* vkState, GLFWwindow* window, uint32_t framesInFlight);

	bool init();

	struct RenderingFrame {
		VkFramebuffer framebuffer;
		VkExtent2D extent;
	};

	RenderingFrame startFrameRecord(uint32_t currentFrame);

	void submitFrame(uint32_t currentFrame, std::vector<VkCommandBuffer> submitCommandBuffers);

	VkRenderPass getRenderPass() { return renderPass; };
};