cmake_minimum_required(VERSION 3.19)
project(End2End)

include(CTest)
enable_testing()
# Тестовый модуль
file(GLOB testfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(file ${testfiles})
  get_filename_component(exec ${file} NAME_WE)
  add_executable(${exec} ${file})
  target_link_libraries(${exec} PRIVATE ETClib)
  target_include_directories(${exec} PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_compile_features(${exec} PUBLIC cxx_std_17)

  add_test(NAME TEST:${file}
          CONFIGURATIONS e2e
          COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test.py ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${exec})
          set_tests_properties(TEST:${file} PROPERTIES DEPENDS ${exec})
endforeach()
