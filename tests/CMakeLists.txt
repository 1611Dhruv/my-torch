# Pull GoogleTest at configure time so nobody has to install it system-wide.
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: prevent overriding the parent project's compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# One executable per unit under test. Add new files here as the team grows them.
add_executable(storage_test storage_test.cpp)
target_link_libraries(storage_test PRIVATE mytorch GTest::gtest_main)
gtest_discover_tests(storage_test)

add_executable(tensor_test tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE mytorch GTest::gtest_main)
gtest_discover_tests(tensor_test)

add_executable(cuda_add_test cuda_add_test.cpp)
target_link_libraries(cuda_add_test PRIVATE mytorch GTest::gtest_main)
gtest_discover_tests(cuda_add_test)
