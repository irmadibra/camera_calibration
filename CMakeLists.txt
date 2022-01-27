cmake_minimum_required(VERSION 2.8.12)

project(camera_calibration)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR "/home/irma/opencv4.5/install4.5.5/lib/cmake/opencv4")

FIND_PACKAGE( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})
