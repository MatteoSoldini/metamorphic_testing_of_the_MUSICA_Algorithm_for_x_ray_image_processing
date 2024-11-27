cmake_minimum_required (VERSION 3.13)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("maverick")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

if (MSVC)
	set_target_properties(${BUILD_TARGET} PROPERTIES LINK_FLAGS "/PROFILE")
endif()

add_library (
	maverick STATIC

	include/vk_state.h
	include/app.h
	include/vk_utils.h
	include/vk_imgui.h
	include/vk_window.h
	include/vk_processing.h
	include/file.h
	include/image.h
	include/vk_compute.h
	include/bezier_curve.h

	src/vk_state.cpp
	src/app.cpp
	src/vk_utils.cpp
	src/vk_imgui.cpp
	src/vk_window.cpp
	src/vk_processing.cpp
	src/file.cpp
	src/image.cpp
	src/vk_compute.cpp
	src/bezier_curve.cpp
)


if (CMAKE_VERSION VERSION_GREATER 3.12)
set_property(TARGET maverick PROPERTY CXX_STANDARD 20)
endif()

# vulkan
find_package(Vulkan REQUIRED)
target_link_libraries(maverick PUBLIC Vulkan::Vulkan)

# glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(dependencies/glfw)
target_link_libraries(maverick PUBLIC glfw)

# imgui
add_library(imgui STATIC)

target_sources(imgui
    PRIVATE dependencies/imgui/imgui.cpp
	PRIVATE dependencies/imgui/imgui_demo.cpp
    PRIVATE dependencies/imgui/imgui_draw.cpp
    PRIVATE dependencies/imgui/imgui_tables.cpp
    PRIVATE dependencies/imgui/imgui_widgets.cpp
	PRIVATE dependencies/imgui/backends/imgui_impl_vulkan.cpp
	PRIVATE dependencies/imgui/backends/imgui_impl_glfw.cpp
)

target_include_directories(imgui
	PUBLIC dependencies/imgui/
	PUBLIC dependencies/imgui/backends
)
target_link_libraries(imgui PRIVATE glfw)
target_link_libraries(imgui PRIVATE Vulkan::Vulkan)

target_link_libraries(maverick PUBLIC imgui)

# stb
add_library(stb INTERFACE)
target_include_directories(stb
	INTERFACE dependencies/stb
)
target_link_libraries(maverick PRIVATE stb)


# shaders
file(GLOB_RECURSE SPIRV_FILES
    "${PROJECT_SOURCE_DIR}/shaders/*.spv"
)

foreach(SPV ${SPIRV_FILES})
	message(STATUS "BUILDING SHADER")
	get_filename_component(SPV_NAME ${SPV} NAME)
	
	message(STATUS ${SPV_NAME})

	add_custom_target(${SPV_NAME} ALL
		COMMAND ${CMAKE_COMMAND} -E copy
		${SPV}
		${PROJECT_BINARY_DIR}/shaders/${SPV_NAME}
		COMMENT "Copying resources into binary directory"
	)

	add_dependencies(maverick ${SPV_NAME})
endforeach(SPV)

# executable
add_executable (
	maverick-app
	"main.cpp" 
)

target_link_libraries(maverick-app
    maverick
)

# # test
# add_executable(maverick-test
# 	test/processing_test/main.cpp
# )

# target_link_libraries(maverick-test
#     maverick
# )

# standalone
add_executable(maverick-standalone
	test/standalone/main.cpp
)

target_link_libraries(maverick-standalone
    maverick
)