# CMakeLists.txt -- builds the sensor_loop binary.
# On a Jetson (JetPack/Ubuntu), plain g++/CMake is sufficient -- no special
# cross-compilation needed since we're building natively on the ARM target.
cmake_minimum_required(VERSION 3.10)
project(detector_sensor_interface CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -O2)

add_executable(sensor_loop
    main_inference_loop.cpp
    i2c_reader.cpp
    uart_reader.cpp
    can_reader.cpp
)

target_link_libraries(sensor_loop pthread)

# Usage on-device:
#   mkdir build && cd build && cmake .. && make
#   ./sensor_loop | python3 ../../jetson_inference.py --stdin-loop
