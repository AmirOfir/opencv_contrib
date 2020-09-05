
# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
#cmake_minimum_required (VERSION 3.8)

ocv_add_module(sepfm opencv_core opencv_imgproc opencv_calib3d WRAP python)
ocv_glob_module_sources()
ocv_module_include_directories()
ocv_create_module()


ocv_warnings_disable(CMAKE_CXX_FLAGS -Wunused-parameter)
ocv_target_link_libraries(${the_module})


#project ("Separable_Fundemental_matrix")

# Include sub-projects.
#add_subdirectory ("Separable_Fundemental_matrix")
