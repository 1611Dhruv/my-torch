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

add_executable(elementwise_test elementwise_test.cpp)
target_link_libraries(elementwise_test PRIVATE mytorch GTest::gtest_main)
gtest_discover_tests(elementwise_test)

add_executable(autograd_test autograd_test.cpp)
target_link_libraries(autograd_test PRIVATE mytorch GTest::gtest_main)
gtest_discover_tests(autograd_test)
