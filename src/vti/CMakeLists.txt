file(GLOB SOURCE "*.cpp" )
list(FILTER SOURCE EXCLUDE REGEX "main*.cpp")

add_library(vti ${SOURCE})

add_executable(surflove main_love.cpp)
target_link_libraries(surflove vti mesh shared ${LAPACK_LIBRARIES})