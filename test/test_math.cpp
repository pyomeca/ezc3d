#include <iostream>
#include <gtest/gtest.h>

#include "Matrix.h"
#include "Vector3d.h"
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

    ezc3d::Matrix m1_copy(m1);
    EXPECT_EQ(m1_copy(0,0), 1.2);
    EXPECT_EQ(m1_copy(0,1), 2.4);
    EXPECT_EQ(m1_copy(0,2), 3.6);
    EXPECT_EQ(m1_copy(1,0), 4.8);
    EXPECT_EQ(m1_copy(1,1), 6.0);
    EXPECT_EQ(m1_copy(1,2), 7.2);

    ezc3d::Matrix m1_equal = m1;
    EXPECT_EQ(m1_copy(0,0), 1.2);
    EXPECT_EQ(m1_copy(0,1), 2.4);
    EXPECT_EQ(m1_copy(0,2), 3.6);
    EXPECT_EQ(m1_copy(1,0), 4.8);
    EXPECT_EQ(m1_copy(1,1), 6.0);
    EXPECT_EQ(m1_copy(1,2), 7.2);
}

TEST(Matrix, unittest){
    ezc3d::Matrix m1(2, 3);
    m1(0,0) = 1.1;
    m1(0,1) = 2.2;
    m1(0,2) = 3.3;
    m1(1,0) = 4.4;
    m1(1,1) = 5.5;
    m1(1,2) = 6.6;

    ezc3d::Matrix m2(2, 3);
    m2(0,0) = 7.7;
    m2(0,1) = 8.8;
    m2(0,2) = 9.9;
    m2(1,0) = 10.0;
    m2(1,1) = 11.1;
    m2(1,2) = 12.2;

    ezc3d::Matrix m3(3, 2);
    m3(0,0) = 7.7;
    m3(0,1) = 8.8;
    m3(1,0) = 9.9;
    m3(1,1) = 10.0;
    m3(2,0) = 11.1;
    m3(2,1) = 12.2;

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
    ezc3d::Matrix m_add1(2.5 + m1);
    EXPECT_EQ(m_add1.nbRows(), 2);
    EXPECT_EQ(m_add1.nbCols(), 3);
    EXPECT_NEAR(m_add1(0, 0), 3.6, requiredPrecision);
    EXPECT_NEAR(m_add1(0, 1), 4.7, requiredPrecision);
    EXPECT_NEAR(m_add1(0, 2), 5.8, requiredPrecision);
    EXPECT_NEAR(m_add1(1, 0), 6.9, requiredPrecision);
    EXPECT_NEAR(m_add1(1, 1), 8.0, requiredPrecision);
    EXPECT_NEAR(m_add1(1, 2), 9.1, requiredPrecision);

    ezc3d::Matrix m_add2(m1 + m2);
    EXPECT_EQ(m_add2.nbRows(), 2);
    EXPECT_EQ(m_add2.nbCols(), 3);
    EXPECT_NEAR(m_add2(0, 0), 8.8, requiredPrecision);
    EXPECT_NEAR(m_add2(0, 1), 11.0, requiredPrecision);
    EXPECT_NEAR(m_add2(0, 2), 13.2, requiredPrecision);
    EXPECT_NEAR(m_add2(1, 0), 14.4, requiredPrecision);
    EXPECT_NEAR(m_add2(1, 1), 16.6, requiredPrecision);
    EXPECT_NEAR(m_add2(1, 2), 18.8, requiredPrecision);

    // Subtraction
    ezc3d::Matrix m_sub1(m1 - 2.5);
    EXPECT_EQ(m_sub1.nbRows(), 2);
    EXPECT_EQ(m_sub1.nbCols(), 3);
    EXPECT_NEAR(m_sub1(0, 0), -1.4, requiredPrecision);
    EXPECT_NEAR(m_sub1(0, 1), -0.3, requiredPrecision);
    EXPECT_NEAR(m_sub1(0, 2), 0.8, requiredPrecision);
    EXPECT_NEAR(m_sub1(1, 0), 1.9, requiredPrecision);
    EXPECT_NEAR(m_sub1(1, 1), 3.0, requiredPrecision);
    EXPECT_NEAR(m_sub1(1, 2), 4.1, requiredPrecision);

    ezc3d::Matrix m_sub2(2.5 - m1);
    EXPECT_EQ(m_sub2.nbRows(), 2);
    EXPECT_EQ(m_sub2.nbCols(), 3);
    EXPECT_NEAR(m_sub2(0, 0), 1.4, requiredPrecision);
    EXPECT_NEAR(m_sub2(0, 1), 0.3, requiredPrecision);
    EXPECT_NEAR(m_sub2(0, 2), -0.8, requiredPrecision);
    EXPECT_NEAR(m_sub2(1, 0), -1.9, requiredPrecision);
    EXPECT_NEAR(m_sub2(1, 1), -3.0, requiredPrecision);
    EXPECT_NEAR(m_sub2(1, 2), -4.1, requiredPrecision);

    ezc3d::Matrix m_sub3(m1 - m2);
    EXPECT_EQ(m_sub3.nbRows(), 2);
    EXPECT_EQ(m_sub3.nbCols(), 3);
    EXPECT_NEAR(m_sub3(0, 0), -6.6, requiredPrecision);
    EXPECT_NEAR(m_sub3(0, 1), -6.6, requiredPrecision);
    EXPECT_NEAR(m_sub3(0, 2), -6.6, requiredPrecision);
    EXPECT_NEAR(m_sub3(1, 0), -5.6, requiredPrecision);
    EXPECT_NEAR(m_sub3(1, 1), -5.6, requiredPrecision);
    EXPECT_NEAR(m_sub3(1, 2), -5.6, requiredPrecision);

    // Multiplication
    ezc3d::Matrix m_mult1(2 * m1);
    EXPECT_EQ(m_mult1.nbRows(), 2);
    EXPECT_EQ(m_mult1.nbCols(), 3);
    EXPECT_NEAR(m_mult1(0, 0), 2.2, requiredPrecision);
    EXPECT_NEAR(m_mult1(0, 1), 4.4, requiredPrecision);
    EXPECT_NEAR(m_mult1(0, 2), 6.6, requiredPrecision);
    EXPECT_NEAR(m_mult1(1, 0), 8.8, requiredPrecision);
    EXPECT_NEAR(m_mult1(1, 1), 11.0, requiredPrecision);
    EXPECT_NEAR(m_mult1(1, 2), 13.2, requiredPrecision);

    ezc3d::Matrix m_mult2(m1 * m3);
    EXPECT_EQ(m_mult2.nbRows(), 2);
    EXPECT_EQ(m_mult2.nbCols(), 2);
    EXPECT_NEAR(m_mult2(0,0), 66.88, requiredPrecision);
    EXPECT_NEAR(m_mult2(0,1), 71.94, requiredPrecision);
    EXPECT_NEAR(m_mult2(1,0), 161.59, requiredPrecision);
    EXPECT_NEAR(m_mult2(1,1), 174.24, requiredPrecision);

    // Division
    ezc3d::Matrix m_div(m1 / 2.);
    EXPECT_EQ(m_div.nbRows(), 2);
    EXPECT_EQ(m_div.nbCols(), 3);
    EXPECT_NEAR(m_div(0, 0), 0.55, requiredPrecision);
    EXPECT_NEAR(m_div(0, 1), 1.1, requiredPrecision);
    EXPECT_NEAR(m_div(0, 2), 1.65, requiredPrecision);
    EXPECT_NEAR(m_div(1, 0), 2.2, requiredPrecision);
    EXPECT_NEAR(m_div(1, 1), 2.75, requiredPrecision);
    EXPECT_NEAR(m_div(1, 2), 3.3, requiredPrecision);
}

TEST(Vector3d, create){
    ezc3d::Vector3d zeros;
    EXPECT_NEAR(zeros(0), 0.0, requiredPrecision);
    EXPECT_NEAR(zeros(1), 0.0, requiredPrecision);
    EXPECT_NEAR(zeros(2), 0.0, requiredPrecision);

    ezc3d::Vector3d random(1.1, 2.2, 3.3);
    EXPECT_NEAR(random(0), 1.1, requiredPrecision);
    EXPECT_NEAR(random(1), 2.2, requiredPrecision);
    EXPECT_NEAR(random(2), 3.3, requiredPrecision);

    ezc3d::Vector3d random_copy(random);
    EXPECT_NEAR(random_copy(0), 1.1, requiredPrecision);
    EXPECT_NEAR(random_copy(1), 2.2, requiredPrecision);
    EXPECT_NEAR(random_copy(2), 3.3, requiredPrecision);

    ezc3d::Vector3d random_equal = random;
    EXPECT_NEAR(random_equal(0), 1.1, requiredPrecision);
    EXPECT_NEAR(random_equal(1), 2.2, requiredPrecision);
    EXPECT_NEAR(random_equal(2), 3.3, requiredPrecision);

}

TEST(Vector3d, unittest){
    ezc3d::Vector3d random(1.1, 2.2, 3.3);
    EXPECT_NEAR(random.norm(), 4.1158231254513353, requiredPrecision);

//    ezc3d::Vector3d random_normalized(random.normalize());


    ezc3d::Vector3d random2(2.2, 3.3, 4.4);
    EXPECT_NEAR(random.norm(), 4.1158231254513353, requiredPrecision);
}
