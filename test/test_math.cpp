#include <iostream>
#include <gtest/gtest.h>

#include "Matrix.h"
double requiredPrecision(1e-10);

TEST(Matrix, create){
    ezc3d::Matrix m1(2, 3);
    EXPECT_EQ(m1.nbRows(), 2);
    EXPECT_EQ(m1.nbCols(), 3);

    m1(0,0) = 1.2;
    m1(0,1) = 2.4;
    m1(0,2) = 3.6;
    m1(1,0) = 4.8;
    m1(1,1) = 6.0;
    m1(1,2) = 7.2;

    EXPECT_EQ(m1(0,0), 1.2);
    EXPECT_EQ(m1(0,1), 2.4);
    EXPECT_EQ(m1(0,2), 3.6);
    EXPECT_EQ(m1(1,0), 4.8);
    EXPECT_EQ(m1(1,1), 6.0);
    EXPECT_EQ(m1(1,2), 7.2);
}

TEST(Matrix, unittest){
    ezc3d::Matrix m1(2, 3);
    m1(0,0) = 1.1;
    m1(0,1) = 2.2;
    m1(0,2) = 3.3;
    m1(1,0) = 4.4;
    m1(1,1) = 5.5;
    m1(1,2) = 6.6;

    ezc3d::Matrix m2(3, 2);
    m2(0,0) = 7.7;
    m2(0,1) = 8.8;
    m2(1,0) = 9.9;
    m2(1,1) = 10.0;
    m2(2,0) = 11.1;
    m2(2,1) = 12.2;

    // Transpose
    ezc3d::Matrix m_transp(m1.T());
    EXPECT_EQ(m_transp.nbRows(), 3);
    EXPECT_EQ(m_transp.nbCols(), 2);
    EXPECT_NEAR(m_transp(0, 0), 1.1, requiredPrecision);
    EXPECT_NEAR(m_transp(0, 1), 4.4, requiredPrecision);
    EXPECT_NEAR(m_transp(1, 0), 2.2, requiredPrecision);
    EXPECT_NEAR(m_transp(1, 1), 5.5, requiredPrecision);
    EXPECT_NEAR(m_transp(2, 0), 3.3, requiredPrecision);
    EXPECT_NEAR(m_transp(2, 1), 6.6, requiredPrecision);

    // Addition
    ezc3d::Matrix m_add(m1 * m2);
    EXPECT_NEAR(m_add(0,0), 66.88, requiredPrecision);
    EXPECT_NEAR(m_add(0,1), 71.94, requiredPrecision);
    EXPECT_NEAR(m_add(1,0), 161.59, requiredPrecision);
    EXPECT_NEAR(m_add(1,1), 174.24, requiredPrecision);

    // Multiplication
    ezc3d::Matrix m_mult(m1 * m2);
    EXPECT_NEAR(m_mult(0,0), 66.88, requiredPrecision);
    EXPECT_NEAR(m_mult(0,1), 71.94, requiredPrecision);
    EXPECT_NEAR(m_mult(1,0), 161.59, requiredPrecision);
    EXPECT_NEAR(m_mult(1,1), 174.24, requiredPrecision);
}
