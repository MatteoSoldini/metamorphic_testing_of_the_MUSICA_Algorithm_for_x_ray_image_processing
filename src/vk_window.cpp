#include "../include/vk_window.h"

#include <stdexcept>

#include "../include/vk_utils.h"

VulkanWindow::VulkanWindow(VulkanState* vkState, GLFWwindow* window,
                           uint32_t framesInFlight) {
    VulkanWindow::vkState = vkState;
    VulkanWindow::window = window;
    VulkanWindow::framesInFlight = framesInFlight;

    surface = VK_NULL_HANDLE;
    swapChain = VK_NULL_HANDLE;
    swapChainExtent = {0, 0};
    swapChainImageFormat = VK_FORMAT_UNDEFINED;
    currentFrame = 0;
    presentQueue = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
    imageIndex = 0;
}

bool VulkanWindow::init() {
    if (glfwCreateWindowSurface(vkState->getInstance(), window, nullptr,
                                &surface) != VK_SUCCESS) {
        return false;
    }

    // create swapchain
    createSwapChain();

    // create render pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp =
        VK_ATTACHMENT_LOAD_OP_CLEAR;  // clear before drawing
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout =
        VK_IMAGE_LAYOUT_UNDEFINED;  // necessary for the first render pass
    colorAttachment.finalLayout =
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  // translate to layout for
                                          // presentation

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(vkState->getDevice(), &renderPassInfo, nullptr,
                           &renderPass) != VK_SUCCESS) {
        return false;
    }

    // create frame buffers
    createFrameBuffers();

    // sync objects
    imageAvailableSemaphores.resize(framesInFlight);
    renderFinishedSemaphores.resize(framesInFlight);
    inFlightFences.resize(framesInFlight);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < framesInFlight; i++) {
        if (vkCreateSemaphore(vkState->getDevice(), &semaphoreInfo, nullptr,
                              &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(vkState->getDevice(), &semaphoreInfo, nullptr,
                              &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(vkState->getDevice(), &fenceInfo, nullptr,
                          &inFlightFences[i]) != VK_SUCCESS) {
            return false;
        }
    }
}

void VulkanWindow::windowResize() {
    // wait if window is minimized
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(vkState->getDevice());

    // destroy
    cleanupSwapChain();
    cleanupFrameBuffers();

    // create
    createSwapChain();
    createFrameBuffers();
}

bool VulkanWindow::createFrameBuffers() {
    frameBuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {swapChainImageViews[i]};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(vkState->getDevice(), &framebufferInfo, nullptr,
                                &frameBuffers[i]) != VK_SUCCESS) {
            return false;
        }
    }

    return true;
}

void VulkanWindow::cleanupFrameBuffers() {
    for (size_t i = 0; i < frameBuffers.size(); i++) {
        vkDestroyFramebuffer(vkState->getDevice(), frameBuffers[i], nullptr);
    }
}

bool VulkanWindow::createSwapChain() {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(vkState->getPhysicalDevice(), surface);

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window);

    // Simply sticking to this minimum means that we may sometimes have to wait
    // on the driver to complete internal operations before we can acquire
    // another image to render to. Therefore it is recommended to request at
    // least one more image than the minimum
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    // This is always 1 unless you are developing a stereoscopic 3D
    // application
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t presentFamily =
        queryPresentQueueFamily(vkState->getPhysicalDevice(), surface).value();

    vkGetDeviceQueue(vkState->getDevice(), presentFamily, 0, &presentQueue);

    uint32_t graphicsFamily = vkState->getGraphicsFamily();
    // Specify how to handle swap chain images that will be used across multiple
    // queue families
    uint32_t queueFamilyIndices[] = {graphicsFamily, presentFamily};

    if (graphicsFamily != presentFamily) {
        createInfo.imageSharingMode =
            VK_SHARING_MODE_CONCURRENT;  // Images can be used across multiple
        // queue families without explicit
        // ownership transfers.
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode =
            VK_SHARING_MODE_EXCLUSIVE;  // An image is owned by one queue family
        // at a time and ownership must be
        // explicitly transferred before using
        // it in another queue family. This
        // option offers the best performance.
        createInfo.queueFamilyIndexCount = 0;      // Optional
        createInfo.pQueueFamilyIndices = nullptr;  // Optional
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(vkState->getDevice(), &createInfo, nullptr,
                             &swapChain) != VK_SUCCESS) {
        return false;
    }
    // We only specified a minimum number of images in the swap chain, so the
    // implementation is allowed to create a swap chain with more. That's why
    // we'll first query the final number of images with
    // vkGetSwapchainImagesKHR, then resize the container and finally call it
    // again to retrieve the handles
    vkGetSwapchainImagesKHR(vkState->getDevice(), swapChain, &imageCount,
                            nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(vkState->getDevice(), swapChain, &imageCount,
                            swapChainImages.data());
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;

    // creates a basic image view for every image in the swap chain
    swapChainImageViews.resize(swapChainImages.size());

    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
        vkState->createImageView(swapChainImages[i], swapChainImageFormat,
                                 &swapChainImageViews[i]);
    }
}

void VulkanWindow::cleanupSwapChain() {
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vkDestroyImageView(vkState->getDevice(), swapChainImageViews[i],
                           nullptr);
    }

    vkDestroySwapchainKHR(vkState->getDevice(), swapChain, nullptr);
}

VulkanWindow::RenderingFrame VulkanWindow::startFrameRecord(
    uint32_t currentFrame) {
    VkResult result = VK_RESULT_MAX_ENUM;

    while (result != VK_SUCCESS) {
        result =
            vkAcquireNextImageKHR(vkState->getDevice(), swapChain, UINT64_MAX,
                                  imageAvailableSemaphores[currentFrame],
                                  VK_NULL_HANDLE, &imageIndex);

        // recreate swapchain on error
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            windowResize();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }
    }

    vkResetFences(vkState->getDevice(), 1, &inFlightFences[currentFrame]);

    return {frameBuffers[imageIndex], swapChainExtent};
}

void VulkanWindow::submitFrame(
    uint32_t currentFrame, std::vector<VkCommandBuffer> submitCommandBuffers) {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount =
        static_cast<uint32_t>(submitCommandBuffers.size());
    submitInfo.pCommandBuffers = submitCommandBuffers.data();

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(vkState->getGraphicsQueue(), 1, &submitInfo,
                      inFlightFences[currentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    // presentation
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;  // Optional

    vkQueuePresentKHR(presentQueue, &presentInfo);

    vkQueueWaitIdle(presentQueue);
}