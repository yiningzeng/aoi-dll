cmake_minimum_required(VERSION 2.8)
project(main)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
find_package(OpenCV REQUIRED)

add_library( aoi SHARED
	aoi.cpp
)

target_link_libraries( aoi
	${OpenCV_LIBRARIES}
)

if(OpenMP_FOUND)
	target_link_libraries( aoi OpenMP::OpenMP_CXX)
endif()

add_executable( ${PROJECT_NAME}
	main.cpp
)

target_link_libraries( ${PROJECT_NAME}
	${OpenCV_LIBRARIES}
	aoi
)
