file(GLOB SOURCE "*.cpp" )
list(FILTER SOURCE EXCLUDE REGEX "main*.cpp")

add_library(vti ${SOURCE})

add_executable(surflove main_love.cpp)
add_executable(surfrayl main_rayl.cpp)
target_link_libraries(surflove vti mesh shared LAPACK::LAPACK -llapacke)
target_link_libraries(surfrayl vti mesh shared LAPACK::LAPACK -llapacke)