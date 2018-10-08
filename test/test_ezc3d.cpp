#include <iostream>
#include "gtest/gtest.h"

#include "ezc3d.h"

// The fixture for testing class Project1. From google test primer.
class Ezc3dTest : public ::testing::Test {
protected:
    Ezc3dTest() : c3d("markers_analogs.c3d")
    {

    }

    // Objects declared here can be used by all tests in the test case for Ezc3dTest.
    ezc3d::c3d c3d_empty;
    ezc3d::c3d c3d;
};

// Test case must be called the class above
// Also note: use TEST_F instead of TEST to access the test fixture (from google test primer)
TEST_F(Ezc3dTest, readTest) {
    std::cout << c3d.data().frame(0).points().point(0).x() << std::endl;
    EXPECT_EQ(1, 1);
}

// }  // namespace - could surround Project1Test in a namespace
