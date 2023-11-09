/* gtk_vulkan_demo.cc -- GTK Vulkan drawing demo */

/*
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <gtkmm.h>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static auto
findMemoryType(
    vk::PhysicalDevice dev, uint32_t typeFilter, vk::MemoryPropertyFlags flags)
{
    auto props = dev.getMemoryProperties();

    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) &&
            (props.memoryTypes[i].propertyFlags & flags)) {
            return i;
        }
    }

    /* No suitable type found. */
    std::abort();
}

/* Vertex input data */
struct VertexInput {
    glm::vec3 pos;
    glm::vec2 uv;

    consteval static auto
    bindingDescription()
    {
        vk::VertexInputBindingDescription ret{
            .binding = 0,
            .stride = sizeof(VertexInput),
            .inputRate = vk::VertexInputRate::eVertex,
        };
        return ret;
    }

    consteval static auto
    attributeDescriptions()
    {
        vk::VertexInputAttributeDescription pos{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(VertexInput, pos),
        };

        vk::VertexInputAttributeDescription uv{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(VertexInput, uv),
        };

        return std::to_array({pos, uv});
    };
};

constinit static auto vertices = std::to_array<VertexInput>({
    /* Front */
    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f}},

    /* Back */
    {{-1.0f, -1.0f, 1.0f}, {0.0f, 0.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f}},
    {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

    /* Left */
    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
    {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f}},
    {{-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 1.0f}},

    /* Right */
    {{1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f}},
    {{1.0f, 1.0f, -1.0f}, {0.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 0.0f}},
    {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}, {0.0f, 1.0f}},

    /* Top */
    {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f}},
    {{-1.0f, -1.0f, 1.0f}, {0.0f, 1.0f}},
    {{1.0f, -1.0f, -1.0f}, {1.0f, 0.0f}},
    {{1.0f, -1.0f, 1.0f}, {1.0f, 1.0f}},
    {{-1.0f, -1.0f, 1.0f}, {0.0f, 1.0f}},

    /* Bottom */
    {{-1.0f, 1.0f, -1.0f}, {0.0f, 0.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 0.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
    {{1.0f, 1.0f, -1.0f}, {1.0f, 0.0f}},
    {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
    {{-1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
});

/* Uniform buffer. */
struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};

/* Render format. */
constinit static auto renderFormat = vk::Format::eR8G8B8A8Srgb;

/* shaders are statically linked in. */
extern "C" char _binary_shader_vert_spv_start[];
extern "C" char _binary_shader_vert_spv_end[];
extern "C" char _binary_shader_frag_spv_start[];
extern "C" char _binary_shader_frag_spv_end[];

class VulkanArea : public Gtk::DrawingArea {
  private:
    vk::Instance m_instance = nullptr;
    vk::PhysicalDevice m_physDev = nullptr;
    vk::Device m_dev = nullptr;
    vk::Queue m_graphicsQueue = nullptr;

    /* Framebuffer */
    vk::Image m_renderImage = nullptr;
    vk::DeviceMemory m_renderImageMemory = nullptr;
    vk::ImageView m_renderImageView = nullptr;
    vk::Buffer m_stagingBuffer = nullptr;
    vk::DeviceMemory m_stagingBufferMemory = nullptr;
    vk::Framebuffer m_framebuffer = nullptr;

    /* Depth buffer */
    vk::Image m_depthImage = nullptr;
    vk::DeviceMemory m_depthImageMemory = nullptr;
    vk::ImageView m_depthImageView = nullptr;

    /* Pipeline */
    vk::RenderPass m_renderPass = nullptr;
    vk::ShaderModule m_vertexShader = nullptr;
    vk::ShaderModule m_fragmentShader = nullptr;
    vk::DescriptorSetLayout m_descriptorSetLayout = nullptr;
    vk::PipelineLayout m_pipelineLayout = nullptr;
    vk::Pipeline m_pipeline = nullptr;

    /* Vertex and Uniform buffers */
    vk::Buffer m_vertexBuffer = nullptr;
    vk::DeviceMemory m_vertexBufferMemory = nullptr;
    vk::Buffer m_uniformBuffer = nullptr;
    vk::DeviceMemory m_uniformBufferMemory = nullptr;

    /* Texture */
    vk::Image m_textureImage = nullptr;
    vk::DeviceMemory m_textureImageMemory = nullptr;
    vk::ImageView m_textureImageView = nullptr;
    vk::Sampler m_textureSampler = nullptr;

    /* Rendering */
    vk::DescriptorPool m_descriptorPool;
    vk::DescriptorSet m_descriptorSet;
    vk::CommandPool m_commandPool = nullptr;
    vk::CommandBuffer m_commandBuffer = nullptr;

    void* m_stagingBufferMmap = nullptr;
    void* m_vertexBufferMmap = nullptr;
    void* m_uniformBufferMmap = nullptr;
    uint32_t m_graphicsQueueFamilyIndex = -1;

    uint32_t m_width = 0, m_height = 0;
    bool m_needResize = true;

    struct Rotations {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    } m_rotations;

    float m_scale = 1.0f;

    void
    m_createInstance()
    {
        vk::ApplicationInfo appInfo{
            .pApplicationName = "GTK Vulkan Demo",
            .applicationVersion = 1,
            .pEngineName = "No Engine",
            .engineVersion = 1,
            .apiVersion = vk::ApiVersion13,
        };

        const static auto enabledLayers = std::to_array({
            "VK_LAYER_KHRONOS_validation",
        });

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = enabledLayers.size(),
            .ppEnabledLayerNames = enabledLayers.data(),
            .enabledExtensionCount = 0,
        };

        m_instance = vk::createInstance(createInfo);
    }

    void
    m_selectPhysDev()
    {
        /* Picking the first one available for now. */
        m_physDev = m_instance.enumeratePhysicalDevices()[0];

        /* Get graphics queue family index. */
        uint32_t idx = 0;
        auto families = m_physDev.getQueueFamilyProperties();
        for (const auto& f : families) {
            if (f.queueFlags & vk::QueueFlagBits::eGraphics) {
                m_graphicsQueueFamilyIndex = idx;
                break;
            }

            ++idx;
        }

        if (idx >= families.size()) {
            /* Not found. */
            throw std::runtime_error("Graphics queue family not found.");
        }
    }

    void
    m_createDevice()
    {
        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = m_graphicsQueueFamilyIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
        };

        auto enabledExtensions =
            std::to_array({VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME});

        vk::PhysicalDeviceFeatures features{
            .samplerAnisotropy = true,
        };

        vk::DeviceCreateInfo createInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queueCreateInfo,
            .enabledExtensionCount = enabledExtensions.size(),
            .ppEnabledExtensionNames = enabledExtensions.data(),
            .pEnabledFeatures = &features,
        };

        m_dev = m_physDev.createDevice(createInfo);
        m_graphicsQueue = m_dev.getQueue(m_graphicsQueueFamilyIndex, 0);
    }

    void
    m_createRenderPass()
    {
        vk::AttachmentDescription colorAttachment{
            .format = renderFormat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eTransferSrcOptimal,
        };

        vk::AttachmentDescription depthAttachment{
            .format = vk::Format::eD32Sfloat,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        vk::AttachmentReference colorAttachmentRef{
            .attachment = 0,
            .layout = vk::ImageLayout::eColorAttachmentOptimal,
        };

        vk::AttachmentReference depthAttachmentRef{
            .attachment = 1,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
        };

        vk::SubpassDependency dependency{
            .srcSubpass = vk::SubpassExternal,
            .dstSubpass = 0,
            .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits::eEarlyFragmentTests,
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                             vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        };

        vk::SubpassDescription subpass{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAttachmentRef,
            .pDepthStencilAttachment = &depthAttachmentRef,
        };

        auto attachments = std::to_array({colorAttachment, depthAttachment});

        vk::RenderPassCreateInfo createInfo{
            .attachmentCount = attachments.size(),
            .pAttachments = attachments.data(),
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &dependency,
        };
        m_renderPass = m_dev.createRenderPass(createInfo);
    }

    void
    m_createShaders()
    {
        std::string_view vertCode(
            _binary_shader_vert_spv_start, _binary_shader_vert_spv_end);
        std::string_view fragCode(
            _binary_shader_frag_spv_start, _binary_shader_frag_spv_end);

        vk::ShaderModuleCreateInfo vertInfo{
            .codeSize = vertCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(vertCode.data()),
        };
        vk::ShaderModuleCreateInfo fragInfo{
            .codeSize = fragCode.size(),
            .pCode = reinterpret_cast<const uint32_t*>(fragCode.data()),
        };

        m_vertexShader = m_dev.createShaderModule(vertInfo);
        m_fragmentShader = m_dev.createShaderModule(fragInfo);
    }

    void
    m_createPipeline()
    {
        vk::PipelineShaderStageCreateInfo vertInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = m_vertexShader,
            .pName = "main",
        };

        vk::PipelineShaderStageCreateInfo fragInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = m_fragmentShader,
            .pName = "main",
        };

        auto shaderStages = std::to_array({vertInfo, fragInfo});

        auto dynamicStates = std::to_array({
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
        });

        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data(),
        };

        auto bindingDescription = VertexInput::bindingDescription();
        auto attributeDescriptions = VertexInput::attributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = attributeDescriptions.size(),
            .pVertexAttributeDescriptions = attributeDescriptions.data(),
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
            .topology = vk::PrimitiveTopology::eTriangleList,
            .primitiveRestartEnable = false,
        };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1,
        };

        vk::PipelineRasterizationStateCreateInfo rasterizerInfo{
            .depthClampEnable = false,
            .rasterizerDiscardEnable = false,
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eNone,
            .frontFace = vk::FrontFace::eClockwise,
            .depthBiasEnable = false,
            .lineWidth = 1.0f,
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = false,
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = false,
            .colorWriteMask = vk::ColorComponentFlagBits::eR |
                              vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB |
                              vk::ColorComponentFlagBits::eA,
        };

        vk::PipelineColorBlendStateCreateInfo colorBlendInfo{
            .logicOpEnable = false,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = true,
            .depthWriteEnable = true,
            .depthCompareOp = vk::CompareOp::eLess,
            .depthBoundsTestEnable = false,
            .stencilTestEnable = false,
        };

        vk::DescriptorSetLayoutBinding uboBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        };

        vk::DescriptorSetLayoutBinding samplerBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
            .pImmutableSamplers = nullptr,
        };

        auto bindings = std::to_array({uboBinding, samplerBinding});

        vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo{
            .bindingCount = bindings.size(),
            .pBindings = bindings.data(),
        };
        m_descriptorSetLayout =
            m_dev.createDescriptorSetLayout(descSetLayoutInfo);

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &m_descriptorSetLayout,
        };
        m_pipelineLayout = m_dev.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .stageCount = shaderStages.size(),
            .pStages = shaderStages.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssemblyInfo,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizerInfo,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlendInfo,
            .pDynamicState = &dynamicStateInfo,
            .layout = m_pipelineLayout,
            .renderPass = m_renderPass,
            .subpass = 0,
        };
        auto [result, value] =
            m_dev.createGraphicsPipeline(nullptr, pipelineInfo);
        if (result != vk::Result::eSuccess) {
            std::abort();
        }
        m_pipeline = value;
    }

    void
    m_createDescriptorSet()
    {
        vk::DescriptorPoolSize uboPoolSize{
            .type = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
        };

        vk::DescriptorPoolSize samplerPoolSize{
            .type = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
        };

        auto poolSizes = std::to_array({uboPoolSize, samplerPoolSize});

        vk::DescriptorPoolCreateInfo poolInfo{
            .maxSets = 1,
            .poolSizeCount = poolSizes.size(),
            .pPoolSizes = poolSizes.data(),
        };

        m_descriptorPool = m_dev.createDescriptorPool(poolInfo);

        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = m_descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &m_descriptorSetLayout,
        };
        m_descriptorSet = m_dev.allocateDescriptorSets(allocInfo)[0];
    }

    void
    m_createBuffers()
    {
        /* Create vertex buffer. */
        auto vertBufSize = vertices.size() * sizeof(VertexInput);
        vk::BufferCreateInfo vertBufInfo{
            .size = vertBufSize,
            .usage = vk::BufferUsageFlagBits::eVertexBuffer,
            .sharingMode = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &m_graphicsQueueFamilyIndex,
        };
        m_vertexBuffer = m_dev.createBuffer(vertBufInfo);

        auto vertMemReq = m_dev.getBufferMemoryRequirements(m_vertexBuffer);
        vk::MemoryAllocateInfo vertAllocInfo{
            .allocationSize = vertMemReq.size,
            .memoryTypeIndex =
                findMemoryType(m_physDev, vertMemReq.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent),
        };
        m_vertexBufferMemory = m_dev.allocateMemory(vertAllocInfo);
        m_dev.bindBufferMemory(m_vertexBuffer, m_vertexBufferMemory, 0);
        m_vertexBufferMmap =
            m_dev.mapMemory(m_vertexBufferMemory, 0, vertBufSize);
        /* Setup vertex buffer right now since it never changes. */
        std::memcpy(m_vertexBufferMmap, vertices.data(), vertBufSize);

        /* Create uniform buffer. */
        auto uniformBufSize = sizeof(UniformBufferObject);
        vk::BufferCreateInfo uniformBufInfo{
            .size = uniformBufSize,
            .usage = vk::BufferUsageFlagBits::eUniformBuffer,
            .sharingMode = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &m_graphicsQueueFamilyIndex,
        };
        m_uniformBuffer = m_dev.createBuffer(uniformBufInfo);

        auto uniformMemReq = m_dev.getBufferMemoryRequirements(m_uniformBuffer);
        vk::MemoryAllocateInfo uniformAllocInfo{
            .allocationSize = uniformMemReq.size,
            .memoryTypeIndex =
                findMemoryType(m_physDev, uniformMemReq.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent),
        };
        m_uniformBufferMemory = m_dev.allocateMemory(uniformAllocInfo);
        m_dev.bindBufferMemory(m_uniformBuffer, m_uniformBufferMemory, 0);
        m_uniformBufferMmap = m_dev.mapMemory(
            m_uniformBufferMemory, 0, sizeof(UniformBufferObject));
        /* Uniform buffer are updated every time we draw. */

        /* Update descriptor set. */
        vk::DescriptorBufferInfo descBufInfo{
            .buffer = m_uniformBuffer,
            .offset = 0,
            .range = vk::WholeSize,
        };
        vk::WriteDescriptorSet descWrite{
            .dstSet = m_descriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &descBufInfo,
        };
        m_dev.updateDescriptorSets({descWrite}, {});
    }

    void
    m_createTexture()
    {
        int w, h, n;
        auto pixels = stbi_load("texture.png", &w, &h, &n, STBI_rgb_alpha);
        if (!pixels) {
            std::abort();
        }

        auto texWidth = static_cast<uint32_t>(w);
        auto texHeight = static_cast<uint32_t>(h);

        vk::DeviceSize pixelSize = w * h * 4;

        /* Allocate staging buffer. */
        vk::BufferCreateInfo bufInfo{
            .size = pixelSize,
            .usage = vk::BufferUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &m_graphicsQueueFamilyIndex,
        };
        auto staging = m_dev.createBufferUnique(bufInfo);

        auto bufMemReq = m_dev.getBufferMemoryRequirements(*staging);
        vk::MemoryAllocateInfo bufAllocInfo{
            .allocationSize = bufMemReq.size,
            .memoryTypeIndex =
                findMemoryType(m_physDev, bufMemReq.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent),
        };
        auto stagingMemory = m_dev.allocateMemoryUnique(bufAllocInfo);
        m_dev.bindBufferMemory(*staging, *stagingMemory, 0);
        auto stagingMmap = m_dev.mapMemory(*stagingMemory, 0, pixelSize);

        /* Copy pixel data into buffer. */
        std::memcpy(stagingMmap, pixels, pixelSize);

        stbi_image_free(pixels);

        /* Create texture image. */
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Srgb,
            .extent =
                {
                    .width = texWidth,
                    .height = texHeight,
                    .depth = 1,
                },
            .mipLevels = 1,
            .arrayLayers = 1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eSampled,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        m_textureImage = m_dev.createImage(imageInfo);

        /* Allocate memory for texture. */
        auto imgMemReq = m_dev.getImageMemoryRequirements(m_textureImage);
        vk::MemoryAllocateInfo imgAllocInfo{
            .allocationSize = imgMemReq.size,
            .memoryTypeIndex =
                findMemoryType(m_physDev, imgMemReq.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eDeviceLocal),
        };
        m_textureImageMemory = m_dev.allocateMemory(imgAllocInfo);
        m_dev.bindImageMemory(m_textureImage, m_textureImageMemory, 0);

        /* Transfer pixel data into image. */
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = m_commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        auto cmd = std::move(m_dev.allocateCommandBuffersUnique(allocInfo)[0]);

        vk::CommandBufferBeginInfo beginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };
        cmd->begin(beginInfo);

        /* Undefined -> TransferSrcOptimal */
        vk::ImageMemoryBarrier barrier1{
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = m_textureImage,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlagBits::eByRegion, {}, {}, {barrier1});

        /* Buffer -> Image */
        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0},
            .imageExtent =
                {
                    .width = texWidth,
                    .height = texHeight,
                    .depth = 1,
                },
        };
        cmd->copyBufferToImage(*staging, m_textureImage,
            vk::ImageLayout::eTransferDstOptimal, {region});

        /* TransferDstOptimal -> ShaderReadOnlyOptimal */
        vk::ImageMemoryBarrier barrier2{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = m_textureImage,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            }};
        cmd->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eFragmentShader,
            vk::DependencyFlagBits::eByRegion, {}, {}, {barrier2});

        cmd->end();
        vk::SubmitInfo submitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &(*cmd),
        };
        m_graphicsQueue.submit(submitInfo);
        m_graphicsQueue.waitIdle();

        /* Create image view */
        vk::ImageViewCreateInfo imageViewInfo{
            .image = m_textureImage,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Srgb,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        m_textureImageView = m_dev.createImageView(imageViewInfo);

	/* Update descriptor set. */
	vk::DescriptorImageInfo descImageInfo {
	    .sampler = m_textureSampler,
	    .imageView = m_textureImageView,
	    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
	};

	vk::WriteDescriptorSet descWrite {
	    .dstSet = m_descriptorSet,
	    .dstBinding = 1,
	    .dstArrayElement = 0,
	    .descriptorCount = 1,
	    .descriptorType = vk::DescriptorType::eCombinedImageSampler,
	    .pImageInfo = &descImageInfo,
	};

        m_dev.updateDescriptorSets({descWrite}, {});
    }

    void
    m_createSampler()
    {
        auto props = m_physDev.getProperties();

        vk::SamplerCreateInfo createInfo{
            .magFilter = vk::Filter::eLinear,
            .minFilter = vk::Filter::eLinear,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = true,
            .maxAnisotropy = props.limits.maxSamplerAnisotropy,
            .compareEnable = false,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = false,
        };
        m_textureSampler = m_dev.createSampler(createInfo);
    }

    void
    m_createCommandBuffer()
    {
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = m_graphicsQueueFamilyIndex,
        };
        m_commandPool = m_dev.createCommandPool(poolInfo);

        vk::CommandBufferAllocateInfo bufInfo{
            .commandPool = m_commandPool,
            .commandBufferCount = 1,
        };
        m_commandBuffer = m_dev.allocateCommandBuffers(bufInfo)[0];
    }

    void
    m_recreateImage()
    {
        if (m_width == 0 || m_height == 0) {
            /* Skip creation. */
            return;
        }

        m_dev.waitIdle();

        /* Destroy previously created image. */
        m_dev.destroyFramebuffer(m_framebuffer);
        m_dev.destroyImageView(m_depthImageView);
        m_depthImageView = nullptr;
        m_dev.destroyImage(m_depthImage);
        m_depthImage = nullptr;
        m_dev.freeMemory(m_depthImageMemory);
        m_depthImageMemory = nullptr;
        m_dev.destroyImageView(m_renderImageView);
        m_renderImageView = nullptr;
        m_dev.destroyImage(m_renderImage);
        m_renderImage = nullptr;
        m_dev.freeMemory(m_renderImageMemory);
        m_renderImageMemory = nullptr;
        m_dev.destroyBuffer(m_stagingBuffer);
        m_stagingBuffer = nullptr;
        m_stagingBufferMmap = nullptr;
        m_dev.freeMemory(m_stagingBufferMemory);
        m_stagingBufferMemory = nullptr;

        /* Create render target image. */
        vk::ImageCreateInfo imageInfo{
            .imageType = vk::ImageType::e2D,
            .format = renderFormat,
            .extent =
                {
                    .width = m_width,
                    .height = m_height,
                    .depth = 1,
                },
            .mipLevels = 1,
            .arrayLayers = 1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferSrc |
                     vk::ImageUsageFlagBits::eColorAttachment,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        m_renderImage = m_dev.createImage(imageInfo);

        /* Allocate memory for the image. */
        auto imgMemReq = m_dev.getImageMemoryRequirements(m_renderImage);
        vk::MemoryAllocateInfo imgAllocInfo{
            .allocationSize = imgMemReq.size,
            .memoryTypeIndex =
                findMemoryType(m_physDev, imgMemReq.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eDeviceLocal),
        };
        m_renderImageMemory = m_dev.allocateMemory(imgAllocInfo);
        m_dev.bindImageMemory(m_renderImage, m_renderImageMemory, 0);

        /* Create image view for render target. */
        vk::ImageViewCreateInfo viewInfo{
            .image = m_renderImage,
            .viewType = vk::ImageViewType::e2D,
            .format = renderFormat,
            .components =
                {
                    .r = vk::ComponentSwizzle::eIdentity,
                    .g = vk::ComponentSwizzle::eIdentity,
                    .b = vk::ComponentSwizzle::eIdentity,
                    .a = vk::ComponentSwizzle::eIdentity,
                },
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        m_renderImageView = m_dev.createImageView(viewInfo);

        /* Create staging buffer for receiving pixel data. */
        std::size_t pixelSize =
            m_width * m_height * 4; /* RGBA, 8bits/sample. */
        vk::BufferCreateInfo bufferInfo{
            .size = pixelSize,
            .usage = vk::BufferUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &m_graphicsQueueFamilyIndex,
        };
        m_stagingBuffer = m_dev.createBuffer(bufferInfo);

        /* Allocate memory for staging buffer. */
        auto bufMemReq = m_dev.getBufferMemoryRequirements(m_stagingBuffer);
        vk::MemoryAllocateInfo bufAllocInfo{
            .allocationSize = bufMemReq.size,
            .memoryTypeIndex =
                findMemoryType(m_physDev, bufMemReq.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent),
        };
        m_stagingBufferMemory = m_dev.allocateMemory(bufAllocInfo);
        m_dev.bindBufferMemory(m_stagingBuffer, m_stagingBufferMemory, 0);
        m_stagingBufferMmap =
            m_dev.mapMemory(m_stagingBufferMemory, 0, pixelSize);

        /* Create depth image. */
        vk::ImageCreateInfo depthImageInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eD32Sfloat,
            .extent =
                {
                    .width = m_width,
                    .height = m_height,
                    .depth = 1,
                },
            .mipLevels = 1,
            .arrayLayers = 1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
            .initialLayout = vk::ImageLayout::eUndefined,
        };
        m_depthImage = m_dev.createImage(depthImageInfo);

        /* Allocate memory for image.. */
        auto memReq = m_dev.getImageMemoryRequirements(m_depthImage);
        vk::MemoryAllocateInfo depthAllocInfo{
            .allocationSize = memReq.size,
            .memoryTypeIndex = findMemoryType(m_physDev, memReq.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal),
        };
        m_depthImageMemory = m_dev.allocateMemory(depthAllocInfo);
        m_dev.bindImageMemory(m_depthImage, m_depthImageMemory, 0);

        /* Create dpeth image view. */
        vk::ImageViewCreateInfo depthViewInfo{
            .image = m_depthImage,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eD32Sfloat,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eDepth,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };
        m_depthImageView = m_dev.createImageView(depthViewInfo);

        auto attachments = std::to_array({
            m_renderImageView,
            m_depthImageView,
        });

        /* Create framebuffer. */
        vk::FramebufferCreateInfo fbInfo{
            .renderPass = m_renderPass,
            .attachmentCount = attachments.size(),
            .pAttachments = attachments.data(),
            .width = m_width,
            .height = m_height,
            .layers = 1,
        };
        m_framebuffer = m_dev.createFramebuffer(fbInfo);
    }

  public:
    VulkanArea()
    {
        m_createInstance();
        m_selectPhysDev();
        m_createDevice();
        m_createRenderPass();
        m_createShaders();
        m_createPipeline();
        m_createDescriptorSet();
	m_createSampler();
        m_createBuffers();
        m_createCommandBuffer();
        m_createTexture();

        signal_resize().connect(sigc::mem_fun(*this, &VulkanArea::on_resize));
        set_draw_func(sigc::mem_fun(*this, &VulkanArea::on_draw));
    }

    ~VulkanArea()
    {
        m_dev.waitIdle();

        m_dev.destroyCommandPool(m_commandPool);
        m_dev.destroyBuffer(m_vertexBuffer);
        m_dev.freeMemory(m_vertexBufferMemory);
        m_dev.destroyBuffer(m_uniformBuffer);
        m_dev.freeMemory(m_uniformBufferMemory);

        m_dev.destroyDescriptorPool(m_descriptorPool);
        m_dev.destroyPipeline(m_pipeline);
        m_dev.destroyPipelineLayout(m_pipelineLayout);
        m_dev.destroyRenderPass(m_renderPass);
        m_dev.destroyShaderModule(m_vertexShader);
        m_dev.destroyShaderModule(m_fragmentShader);
        m_dev.destroyDescriptorSetLayout(m_descriptorSetLayout);

        m_dev.destroyImageView(m_textureImageView);
        m_dev.destroyImage(m_textureImage);
        m_dev.freeMemory(m_textureImageMemory);
        m_dev.destroySampler(m_textureSampler);

        m_dev.destroyFramebuffer(m_framebuffer);
        m_dev.destroyImageView(m_depthImageView);
        m_dev.destroyImage(m_depthImage);
        m_dev.freeMemory(m_depthImageMemory);
        m_dev.destroyBuffer(m_stagingBuffer);
        m_dev.freeMemory(m_stagingBufferMemory);
        m_dev.destroyImageView(m_renderImageView);
        m_dev.destroyImage(m_renderImage);
        m_dev.freeMemory(m_renderImageMemory);

        m_dev.destroy();
        m_instance.destroy();
    }

    void
    on_resize(int, int)
    {
        m_needResize = true;
    }

    void
    on_draw(const Cairo::RefPtr<Cairo::Context>& cr, int, int)
    {
        if (m_needResize) {
            /* Resize Framebuffer. */
            auto scale = get_scale_factor();
            auto width = get_width() * scale;
            auto height = get_height() * scale;
            m_width = width;
            m_height = height;
            m_recreateImage();
        }

        /* Update uniform buffer. */
        auto scale =
            glm::scale(glm::mat4(1.0f), glm::vec3(m_scale, m_scale, m_scale));
        auto rotateZ =
            glm::rotate(scale, glm::radians(m_rotations.z), glm::vec3(0, 0, 1));
        auto rotateY = glm::rotate(
            rotateZ, glm::radians(m_rotations.y), glm::vec3(0, 1, 0));
        auto rotateX = glm::rotate(
            rotateY, glm::radians(m_rotations.x), glm::vec3(1, 0, 0));
        auto lookAt = glm::lookAt(glm::vec3(0.0f, 3.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        auto perspective = glm::perspective(glm::radians(45.0f),
            m_width / static_cast<float>(m_height), 0.1f, 10.0f);

        UniformBufferObject ubo = {
            .model = rotateX,
            .view = lookAt,
            .projection = perspective,
        };

        std::memcpy(m_uniformBufferMmap, &ubo, sizeof(ubo));

        /* Recording command buffer */
        vk::CommandBufferBeginInfo beginInfo{};
        m_commandBuffer.begin(beginInfo);

        vk::ClearValue clearColor = {
            .color = {std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}}};
        vk::ClearValue clearDepth = {.depthStencil = {1.0f, 0}};

        auto clearValues = std::to_array({clearColor, clearDepth});

        vk::RenderPassBeginInfo renderPassInfo{
            .renderPass = m_renderPass,
            .framebuffer = m_framebuffer,
            .renderArea =
                {
                    .offset = {0, 0},
                    .extent = {m_width, m_height},
                },
            .clearValueCount = clearValues.size(),
            .pClearValues = clearValues.data(),
        };
        m_commandBuffer.beginRenderPass(
            renderPassInfo, vk::SubpassContents::eInline);

        m_commandBuffer.bindPipeline(
            vk::PipelineBindPoint::eGraphics, m_pipeline);

        vk::Viewport viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(m_width),
            .height = static_cast<float>(m_height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        m_commandBuffer.setViewport(0, {viewport});

        vk::Rect2D scissor{
            .offset = {0, 0},
            .extent = {m_width, m_height},
        };
        m_commandBuffer.setScissor(0, {scissor});

        m_commandBuffer.bindVertexBuffers(0, {m_vertexBuffer}, {0});
        m_commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            m_pipelineLayout, 0, m_descriptorSet, {});
        m_commandBuffer.draw(vertices.size(), 1, 0, 0);

        m_commandBuffer.endRenderPass();

        /* Synchronize image access. */
        vk::ImageMemoryBarrier imageBarrier{
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image = m_renderImage,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        };

        m_commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eTransfer,
            vk::DependencyFlagBits::eByRegion, {}, {}, {imageBarrier});

        /* Copy framebuffer content to staging buffer. */
        vk::BufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,

            .imageSubresource =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = {m_width, m_height, 1},
        };
        m_commandBuffer.copyImageToBuffer(m_renderImage,
            vk::ImageLayout::eTransferSrcOptimal, m_stagingBuffer, {region});

        m_commandBuffer.end();

        vk::SubmitInfo submitInfo{
            .waitSemaphoreCount = 0,
            .commandBufferCount = 1,
            .pCommandBuffers = &m_commandBuffer,
            .signalSemaphoreCount = 0,
        };

        /* Submit draw calls to GPU. */
        m_graphicsQueue.submit(submitInfo);

        /* Wait until finish. */
        m_graphicsQueue.waitIdle();

        /* Retrive data into Gdk::PixBuf */
        auto pixbuf = Gdk::Pixbuf::create(
            Gdk::Colorspace::RGB, true, 8, m_width, m_height);
        if (!pixbuf) {
            throw std::bad_alloc();
        }
        std::memcpy(
            pixbuf->get_pixels(), m_stagingBufferMmap, m_width * m_height * 4);

        /* Draw area. */
        Gdk::Cairo::set_source_pixbuf(cr, pixbuf);
        cr->rectangle(0, 0, m_width, m_height);
        cr->fill();
    }

    void
    setRotationX(float x)
    {
        m_rotations.x = x;
        queue_draw();
    }

    void
    setRotationY(float y)
    {
        m_rotations.y = y;
        queue_draw();
    }

    void
    setRotationZ(float z)
    {
        m_rotations.z = z;
        queue_draw();
    }

    void
    setScale(float s)
    {
        m_scale = s;
        queue_draw();
    }
};

class VulkanDemoWindow : public Gtk::Window {
  private:
    VulkanArea m_vulkanArea;

    Gtk::Box m_vulkanAreaBox;

    Gtk::Label m_rotationXLabel;
    Gtk::Scale m_rotationXScale;
    Gtk::Box m_rotationXBox;
    Gtk::Label m_rotationYLabel;
    Gtk::Scale m_rotationYScale;
    Gtk::Box m_rotationYBox;
    Gtk::Scale m_rotationZScale;
    Gtk::Label m_rotationZLabel;
    Gtk::Box m_rotationZBox;
    Gtk::Label m_zoomLabel;
    Gtk::Scale m_zoomScale;
    Gtk::Box m_zoomBox;

    Gtk::Box m_box;

    void
    on_rotationXValue()
    {
        m_vulkanArea.setRotationX(m_rotationXScale.get_value());
    }

    void
    on_rotationYValue()
    {
        m_vulkanArea.setRotationY(m_rotationYScale.get_value());
    }

    void
    on_rotationZValue()
    {
        m_vulkanArea.setRotationZ(m_rotationZScale.get_value());
    }

    void
    on_zoomValue()
    {
        m_vulkanArea.setScale(1 - m_zoomScale.get_value());
    }

  public:
    VulkanDemoWindow()
    {
        set_title("GTK Vulkan Demo");
        set_default_size(400, 500);

        m_vulkanArea.set_size_request(400, 400);

        m_vulkanAreaBox.set_halign(Gtk::Align::CENTER);
        m_vulkanAreaBox.append(m_vulkanArea);

        m_rotationXLabel.set_text("X");
        m_rotationXScale.set_range(0, 360);
        m_rotationXScale.set_expand(true);
        m_rotationXScale.signal_value_changed().connect(
            sigc::mem_fun(*this, &VulkanDemoWindow::on_rotationXValue));
        m_rotationXBox.append(m_rotationXLabel);
        m_rotationXBox.append(m_rotationXScale);

        m_rotationYLabel.set_text("Y");
        m_rotationYScale.set_range(0, 360);
        m_rotationYScale.set_expand(true);
        m_rotationYScale.signal_value_changed().connect(
            sigc::mem_fun(*this, &VulkanDemoWindow::on_rotationYValue));
        m_rotationYBox.append(m_rotationYLabel);
        m_rotationYBox.append(m_rotationYScale);

        m_rotationZLabel.set_text("Z");
        m_rotationZScale.set_range(0, 360);
        m_rotationZScale.set_expand(true);
        m_rotationZScale.signal_value_changed().connect(
            sigc::mem_fun(*this, &VulkanDemoWindow::on_rotationZValue));
        m_rotationZBox.append(m_rotationZLabel);
        m_rotationZBox.append(m_rotationZScale);

        m_zoomLabel.set_text("Scale");
        m_zoomScale.set_range(0, 1);
        m_zoomScale.set_expand(true);
        m_zoomScale.signal_value_changed().connect(
            sigc::mem_fun(*this, &VulkanDemoWindow::on_zoomValue));
        m_zoomBox.append(m_zoomLabel);
        m_zoomBox.append(m_zoomScale);

        m_box.set_orientation(Gtk::Orientation::VERTICAL);
        m_box.append(m_vulkanAreaBox);
        m_box.append(m_rotationXBox);
        m_box.append(m_rotationYBox);
        m_box.append(m_rotationZBox);
        m_box.append(m_zoomBox);

        set_child(m_box);
    }
};

int
main(int argc, char* argv[])
{
    auto app = Gtk::Application::create("dev.hitomi.gtkvulkandemo");
    return app->make_window_and_run<VulkanDemoWindow>(argc, argv);
}
