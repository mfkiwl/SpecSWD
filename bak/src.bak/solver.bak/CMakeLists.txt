file(GLOB SOURCE "*.cpp" )
list(FILTER SOURCE EXCLUDE REGEX "main*.cpp")

add_library(libsolver ${SOURCE})


add_executable(surflove ${PROJECT_SOURCE_DIR}/src/solver/main_love.cpp)
target_link_libraries(surflove libsolver shared ${LAPACK_LIBRARIES})