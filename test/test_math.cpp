#include <iostream>
#include <gtest/gtest.h>

#include "ezc3d_all.h"

TEST(Matrix, create){
    ezc3d::Matrix m1(2, 3);
    EXPECT_EQ(m1.nbRows(), 2);
    EXPECT_EQ(m1.nbCols(), 3);
    EXPECT_EQ(m1.size(), 6);

    m1(0,0) = 1.2;
    m1(0,1) = 2.4;
    m1(0,2) = 3.6;
    m1(1,0) = 4.8;
    m1(1,1) = 6.0;
    m1(1,2) = 7.2;
    m1.print();

    EXPECT_EQ(m1(0,0), 1.2);
    EXPECT_EQ(m1(0,1), 2.4);
    EXPECT_EQ(m1(0,2), 3.6);
    EXPECT_EQ(m1(1,0), 4.8);
    EXPECT_EQ(m1(1,1), 6.0);
    EXPECT_EQ(m1(1,2), 7.2);
#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(m1(2, 0), std::runtime_error);
    EXPECT_THROW(m1(0, 3), std::runtime_error);
#endif

    ezc3d::Matrix m1_insider(m1);
    EXPECT_EQ(m1_insider(0,0), 1.2);
    EXPECT_EQ(m1_insider(0,1), 2.4);
    EXPECT_EQ(m1_insider(0,2), 3.6);
    EXPECT_EQ(m1_insider(1,0), 4.8);
    EXPECT_EQ(m1_insider(1,1), 6.0);
    EXPECT_EQ(m1_insider(1,2), 7.2);

    ezc3d::Matrix m1_copy(m1);
    EXPECT_EQ(m1_copy.nbRows(), 2);
    EXPECT_EQ(m1_copy.nbCols(), 3);
    EXPECT_EQ(m1_copy.size(), 6);
    EXPECT_EQ(m1_copy(0,0), 1.2);
    EXPECT_EQ(m1_copy(0,1), 2.4);
    EXPECT_EQ(m1_copy(0,2), 3.6);
    EXPECT_EQ(m1_copy(1,0), 4.8);
    EXPECT_EQ(m1_copy(1,1), 6.0);
    EXPECT_EQ(m1_copy(1,2), 7.2);

    ezc3d::Matrix m1_equal = m1;
    EXPECT_EQ(m1_equal.nbRows(), 2);
    EXPECT_EQ(m1_equal.nbCols(), 3);
    EXPECT_EQ(m1_equal.size(), 6);
    EXPECT_EQ(m1_copy(0,0), 1.2);
    EXPECT_EQ(m1_copy(0,1), 2.4);
    EXPECT_EQ(m1_copy(0,2), 3.6);
    EXPECT_EQ(m1_copy(1,0), 4.8);
    EXPECT_EQ(m1_copy(1,1), 6.0);
    EXPECT_EQ(m1_copy(1,2), 7.2);

    ezc3d::Matrix m_toResize;
    EXPECT_EQ(m_toResize.nbRows(), 0);
    EXPECT_EQ(m_toResize.nbCols(), 0);
    EXPECT_EQ(m_toResize.size(), 0);
    m_toResize.resize(2, 3);
    EXPECT_EQ(m_toResize.nbRows(), 2);
    EXPECT_EQ(m_toResize.nbCols(), 3);
    EXPECT_EQ(m_toResize.size(), 6);
    m_toResize(0,0) = 1.2;
    m_toResize(0,1) = 2.4;
    m_toResize(0,2) = 3.6;
    m_toResize(1,0) = 4.8;
    m_toResize(1,1) = 6.0;
    m_toResize(1,2) = 7.2;

    m_toResize.resize(3, 4);
    EXPECT_EQ(m_toResize.nbRows(), 3);
    EXPECT_EQ(m_toResize.nbCols(), 4);
    EXPECT_EQ(m_toResize.size(), 12);

    ezc3d::Matrix m_zeros(2, 3);
    m_zeros.setZeros();
    EXPECT_EQ(m_zeros(0,0), 0.0);
    EXPECT_EQ(m_zeros(0,1), 0.0);
    EXPECT_EQ(m_zeros(0,2), 0.0);
    EXPECT_EQ(m_zeros(1,0), 0.0);
    EXPECT_EQ(m_zeros(1,1), 0.0);
    EXPECT_EQ(m_zeros(1,2), 0.0);

    ezc3d::Matrix m_ones(2, 3);
    m_ones.setOnes();
    EXPECT_EQ(m_ones(0,0), 1.0);
    EXPECT_EQ(m_ones(0,1), 1.0);
    EXPECT_EQ(m_ones(0,2), 1.0);
    EXPECT_EQ(m_ones(1,0), 1.0);
    EXPECT_EQ(m_ones(1,1), 1.0);
    EXPECT_EQ(m_ones(1,2), 1.0);

    ezc3d::Matrix m_identity(2, 3);
    m_identity.setIdentity();
    EXPECT_EQ(m_identity(0,0), 1.0);
    EXPECT_EQ(m_identity(0,1), 0.0);
    EXPECT_EQ(m_identity(0,2), 0.0);
    EXPECT_EQ(m_identity(1,0), 0.0);
    EXPECT_EQ(m_identity(1,1), 1.0);
    EXPECT_EQ(m_identity(1,2), 0.0);

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
    EXPECT_EQ(m_transp.size(), 6);
    EXPECT_DOUBLE_EQ(m_transp(0, 0), 1.1);
    EXPECT_DOUBLE_EQ(m_transp(0, 1), 4.4);
    EXPECT_DOUBLE_EQ(m_transp(1, 0), 2.2);
    EXPECT_DOUBLE_EQ(m_transp(1, 1), 5.5);
    EXPECT_DOUBLE_EQ(m_transp(2, 0), 3.3);
    EXPECT_DOUBLE_EQ(m_transp(2, 1), 6.6);

    // Addition
    ezc3d::Matrix m_add1(2.5 + m1);
    EXPECT_EQ(m_add1.nbRows(), 2);
    EXPECT_EQ(m_add1.nbCols(), 3);
    EXPECT_EQ(m_add1.size(), 6);
    EXPECT_FLOAT_EQ(m_add1(0, 0), 3.6);
    EXPECT_DOUBLE_EQ(m_add1(0, 1), 4.7);
    EXPECT_DOUBLE_EQ(m_add1(0, 2), 5.8);
    EXPECT_DOUBLE_EQ(m_add1(1, 0), 6.9);
    EXPECT_DOUBLE_EQ(m_add1(1, 1), 8.0);
    EXPECT_DOUBLE_EQ(m_add1(1, 2), 9.1);

    ezc3d::Matrix m_add2(m1 + m2);
    EXPECT_EQ(m_add2.nbRows(), 2);
    EXPECT_EQ(m_add2.nbCols(), 3);
    EXPECT_EQ(m_add2.size(), 6);
    EXPECT_DOUBLE_EQ(m_add2(0, 0), 8.8);
    EXPECT_DOUBLE_EQ(m_add2(0, 1), 11.0);
    EXPECT_DOUBLE_EQ(m_add2(0, 2), 13.2);
    EXPECT_DOUBLE_EQ(m_add2(1, 0), 14.4);
    EXPECT_DOUBLE_EQ(m_add2(1, 1), 16.6);
    EXPECT_DOUBLE_EQ(m_add2(1, 2), 18.8);

#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(m1 + m3, std::runtime_error);
#endif

    // Subtraction
    ezc3d::Matrix m_sub1(m1 - 2.5);
    EXPECT_EQ(m_sub1.nbRows(), 2);
    EXPECT_EQ(m_sub1.nbCols(), 3);
    EXPECT_EQ(m_sub1.size(), 6);
    EXPECT_DOUBLE_EQ(m_sub1(0, 0), -1.4);
    EXPECT_DOUBLE_EQ(m_sub1(0, 1), -0.3);
    EXPECT_DOUBLE_EQ(m_sub1(0, 2), 0.8);
    EXPECT_DOUBLE_EQ(m_sub1(1, 0), 1.9);
    EXPECT_DOUBLE_EQ(m_sub1(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(m_sub1(1, 2), 4.1);

    ezc3d::Matrix m_sub2(2.5 - m1);
    EXPECT_EQ(m_sub2.nbRows(), 2);
    EXPECT_EQ(m_sub2.nbCols(), 3);
    EXPECT_EQ(m_sub2.size(), 6);
    EXPECT_DOUBLE_EQ(m_sub2(0, 0), 1.4);
    EXPECT_DOUBLE_EQ(m_sub2(0, 1), 0.3);
    EXPECT_DOUBLE_EQ(m_sub2(0, 2), -0.8);
    EXPECT_DOUBLE_EQ(m_sub2(1, 0), -1.9);
    EXPECT_DOUBLE_EQ(m_sub2(1, 1), -3.0);
    EXPECT_DOUBLE_EQ(m_sub2(1, 2), -4.1);

    ezc3d::Matrix m_sub3(m1 - m2);
    EXPECT_EQ(m_sub3.nbRows(), 2);
    EXPECT_EQ(m_sub3.nbCols(), 3);
    EXPECT_EQ(m_sub3.size(), 6);
    EXPECT_DOUBLE_EQ(m_sub3(0, 0), -6.6);
    EXPECT_DOUBLE_EQ(m_sub3(0, 1), -6.6);
    EXPECT_DOUBLE_EQ(m_sub3(0, 2), -6.6);
    EXPECT_DOUBLE_EQ(m_sub3(1, 0), -5.6);
    EXPECT_DOUBLE_EQ(m_sub3(1, 1), -5.6);
    EXPECT_DOUBLE_EQ(m_sub3(1, 2), -5.6);

#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(m1 - m3, std::runtime_error);
#endif

    // Multiplication
    ezc3d::Matrix m_mult1(2 * m1);
    EXPECT_EQ(m_mult1.nbRows(), 2);
    EXPECT_EQ(m_mult1.nbCols(), 3);
    EXPECT_EQ(m_mult1.size(), 6);
    EXPECT_DOUBLE_EQ(m_mult1(0, 0), 2.2);
    EXPECT_DOUBLE_EQ(m_mult1(0, 1), 4.4);
    EXPECT_DOUBLE_EQ(m_mult1(0, 2), 6.6);
    EXPECT_DOUBLE_EQ(m_mult1(1, 0), 8.8);
    EXPECT_DOUBLE_EQ(m_mult1(1, 1), 11.0);
    EXPECT_DOUBLE_EQ(m_mult1(1, 2), 13.2);

#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(m1 * m2, std::runtime_error);
#endif

    ezc3d::Matrix m_mult2(m1 * m3);
    EXPECT_EQ(m_mult2.nbRows(), 2);
    EXPECT_EQ(m_mult2.nbCols(), 2);
    EXPECT_EQ(m_mult2.size(), 4);
    EXPECT_DOUBLE_EQ(m_mult2(0,0), 66.88);
    EXPECT_DOUBLE_EQ(m_mult2(0,1), 71.94);
    EXPECT_DOUBLE_EQ(m_mult2(1,0), 161.59);
    EXPECT_DOUBLE_EQ(m_mult2(1,1), 174.24);

    // Division
    ezc3d::Matrix m_div(m1 / 2.);
    EXPECT_EQ(m_div.nbRows(), 2);
    EXPECT_EQ(m_div.nbCols(), 3);
    EXPECT_EQ(m_div.size(), 6);
    EXPECT_DOUBLE_EQ(m_div(0, 0), 0.55);
    EXPECT_DOUBLE_EQ(m_div(0, 1), 1.1);
    EXPECT_DOUBLE_EQ(m_div(0, 2), 1.65);
    EXPECT_DOUBLE_EQ(m_div(1, 0), 2.2);
    EXPECT_DOUBLE_EQ(m_div(1, 1), 2.75);
    EXPECT_DOUBLE_EQ(m_div(1, 2), 3.3);
}

TEST(Matrix33, unittest){
    ezc3d::Matrix33 m_toFill;
    EXPECT_EQ(m_toFill.nbRows(), 3);
    EXPECT_EQ(m_toFill.nbCols(), 3);
    EXPECT_EQ(m_toFill.size(), 9);
    EXPECT_THROW(m_toFill.resize(0, 0), std::runtime_error);
    m_toFill(0, 0) = 2.;
    m_toFill(0, 1) = 3.;
    m_toFill(0, 2) = 4.;
    m_toFill(1, 0) = 5.;
    m_toFill(1, 1) = 6.;
    m_toFill(1, 2) = 7.;
    m_toFill(2, 0) = 8.;
    m_toFill(2, 1) = 9.;
    m_toFill(2, 2) = 10.;
    EXPECT_DOUBLE_EQ(m_toFill(0, 0), 2.);
    EXPECT_DOUBLE_EQ(m_toFill(0, 1), 3.);
    EXPECT_DOUBLE_EQ(m_toFill(0, 2), 4.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 0), 5.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 1), 6.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 2), 7.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 0), 8.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 1), 9.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 2), 10.);

    ezc3d::Matrix33 m1(
                1., 2., 3.,
                4., 5., 6.,
                7., 8., 9.);
    EXPECT_EQ(m1.nbRows(), 3);
    EXPECT_EQ(m1.nbCols(), 3);
    EXPECT_EQ(m1.size(), 9);

    EXPECT_DOUBLE_EQ(m1(0, 0), 1.);
    EXPECT_DOUBLE_EQ(m1(0, 1), 2.);
    EXPECT_DOUBLE_EQ(m1(0, 2), 3.);
    EXPECT_DOUBLE_EQ(m1(1, 0), 4.);
    EXPECT_DOUBLE_EQ(m1(1, 1), 5.);
    EXPECT_DOUBLE_EQ(m1(1, 2), 6.);
    EXPECT_DOUBLE_EQ(m1(2, 0), 7.);
    EXPECT_DOUBLE_EQ(m1(2, 1), 8.);
    EXPECT_DOUBLE_EQ(m1(2, 2), 9.);

    ezc3d::Matrix33 m1_copy(m1);
    EXPECT_EQ(m1_copy.nbRows(), 3);
    EXPECT_EQ(m1_copy.nbCols(), 3);
    EXPECT_EQ(m1_copy.size(), 9);
    EXPECT_DOUBLE_EQ(m1_copy(0, 0), 1.);
    EXPECT_DOUBLE_EQ(m1_copy(0, 1), 2.);
    EXPECT_DOUBLE_EQ(m1_copy(0, 2), 3.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 0), 4.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 1), 5.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 2), 6.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 0), 7.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 1), 8.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 2), 9.);

    ezc3d::Matrix33 m_fromMult(m1 * m_toFill);
    EXPECT_DOUBLE_EQ(m_fromMult(0, 0), 36.);
    EXPECT_DOUBLE_EQ(m_fromMult(0, 1), 42.);
    EXPECT_DOUBLE_EQ(m_fromMult(0, 2), 48.);
    EXPECT_DOUBLE_EQ(m_fromMult(1, 0), 81.);
    EXPECT_DOUBLE_EQ(m_fromMult(1, 1), 96.);
    EXPECT_DOUBLE_EQ(m_fromMult(1, 2), 111.);
    EXPECT_DOUBLE_EQ(m_fromMult(2, 0), 126.);
    EXPECT_DOUBLE_EQ(m_fromMult(2, 1), 150.);
    EXPECT_DOUBLE_EQ(m_fromMult(2, 2), 174.);

#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(ezc3d::Matrix33(ezc3d::Matrix(2, 3)), std::runtime_error);
    EXPECT_THROW(ezc3d::Matrix33(ezc3d::Matrix(3, 2)), std::runtime_error);
#endif
}

TEST(Matrix66, unittest){
    ezc3d::Matrix66 m_toFill;
    EXPECT_EQ(m_toFill.nbRows(), 6);
    EXPECT_EQ(m_toFill.nbCols(), 6);
    EXPECT_EQ(m_toFill.size(), 36);
    EXPECT_THROW(m_toFill.resize(0, 0), std::runtime_error);
    m_toFill(0, 0) = 2.;
    m_toFill(0, 1) = 3.;
    m_toFill(0, 2) = 4.;
    m_toFill(0, 3) = 2.;
    m_toFill(0, 4) = 3.;
    m_toFill(0, 5) = 4.;
    m_toFill(1, 0) = 5.;
    m_toFill(1, 1) = 6.;
    m_toFill(1, 2) = 7.;
    m_toFill(1, 3) = 5.;
    m_toFill(1, 4) = 6.;
    m_toFill(1, 5) = 7.;
    m_toFill(2, 0) = 8.;
    m_toFill(2, 1) = 9.;
    m_toFill(2, 2) = 10.;
    m_toFill(2, 3) = 8.;
    m_toFill(2, 4) = 9.;
    m_toFill(2, 5) = 10.;
    m_toFill(3, 0) = 2.;
    m_toFill(3, 1) = 3.;
    m_toFill(3, 2) = 4.;
    m_toFill(3, 3) = 2.;
    m_toFill(3, 4) = 3.;
    m_toFill(3, 5) = 4.;
    m_toFill(4, 0) = 5.;
    m_toFill(4, 1) = 6.;
    m_toFill(4, 2) = 7.;
    m_toFill(4, 3) = 5.;
    m_toFill(4, 4) = 6.;
    m_toFill(4, 5) = 7.;
    m_toFill(5, 0) = 8.;
    m_toFill(5, 1) = 9.;
    m_toFill(5, 2) = 10.;
    m_toFill(5, 3) = 8.;
    m_toFill(5, 4) = 9.;
    m_toFill(5, 5) = 10.;
    EXPECT_DOUBLE_EQ(m_toFill(0, 0), 2.);
    EXPECT_DOUBLE_EQ(m_toFill(0, 1), 3.);
    EXPECT_DOUBLE_EQ(m_toFill(0, 2), 4.);
    EXPECT_DOUBLE_EQ(m_toFill(0, 3), 2.);
    EXPECT_DOUBLE_EQ(m_toFill(0, 4), 3.);
    EXPECT_DOUBLE_EQ(m_toFill(0, 5), 4.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 0), 5.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 1), 6.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 2), 7.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 3), 5.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 4), 6.);
    EXPECT_DOUBLE_EQ(m_toFill(1, 5), 7.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 0), 8.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 1), 9.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 2), 10.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 3), 8.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 4), 9.);
    EXPECT_DOUBLE_EQ(m_toFill(2, 5), 10.);
    EXPECT_DOUBLE_EQ(m_toFill(3, 0), 2.);
    EXPECT_DOUBLE_EQ(m_toFill(3, 1), 3.);
    EXPECT_DOUBLE_EQ(m_toFill(3, 2), 4.);
    EXPECT_DOUBLE_EQ(m_toFill(3, 3), 2.);
    EXPECT_DOUBLE_EQ(m_toFill(3, 4), 3.);
    EXPECT_DOUBLE_EQ(m_toFill(3, 5), 4.);
    EXPECT_DOUBLE_EQ(m_toFill(4, 0), 5.);
    EXPECT_DOUBLE_EQ(m_toFill(4, 1), 6.);
    EXPECT_DOUBLE_EQ(m_toFill(4, 2), 7.);
    EXPECT_DOUBLE_EQ(m_toFill(4, 3), 5.);
    EXPECT_DOUBLE_EQ(m_toFill(4, 4), 6.);
    EXPECT_DOUBLE_EQ(m_toFill(4, 5), 7.);
    EXPECT_DOUBLE_EQ(m_toFill(5, 0), 8.);
    EXPECT_DOUBLE_EQ(m_toFill(5, 1), 9.);
    EXPECT_DOUBLE_EQ(m_toFill(5, 2), 10.);
    EXPECT_DOUBLE_EQ(m_toFill(5, 3), 8.);
    EXPECT_DOUBLE_EQ(m_toFill(5, 4), 9.);
    EXPECT_DOUBLE_EQ(m_toFill(5, 5), 10.);

    ezc3d::Matrix66 m1_copy(m_toFill);
    EXPECT_EQ(m1_copy.nbRows(), 6);
    EXPECT_EQ(m1_copy.nbCols(), 6);
    EXPECT_EQ(m1_copy.size(), 36);
    EXPECT_DOUBLE_EQ(m1_copy(0, 0), 2.);
    EXPECT_DOUBLE_EQ(m1_copy(0, 1), 3.);
    EXPECT_DOUBLE_EQ(m1_copy(0, 2), 4.);
    EXPECT_DOUBLE_EQ(m1_copy(0, 3), 2.);
    EXPECT_DOUBLE_EQ(m1_copy(0, 4), 3.);
    EXPECT_DOUBLE_EQ(m1_copy(0, 5), 4.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 0), 5.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 1), 6.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 2), 7.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 3), 5.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 4), 6.);
    EXPECT_DOUBLE_EQ(m1_copy(1, 5), 7.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 0), 8.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 1), 9.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 2), 10.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 3), 8.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 4), 9.);
    EXPECT_DOUBLE_EQ(m1_copy(2, 5), 10.);
    EXPECT_DOUBLE_EQ(m1_copy(3, 0), 2.);
    EXPECT_DOUBLE_EQ(m1_copy(3, 1), 3.);
    EXPECT_DOUBLE_EQ(m1_copy(3, 2), 4.);
    EXPECT_DOUBLE_EQ(m1_copy(3, 3), 2.);
    EXPECT_DOUBLE_EQ(m1_copy(3, 4), 3.);
    EXPECT_DOUBLE_EQ(m1_copy(3, 5), 4.);
    EXPECT_DOUBLE_EQ(m1_copy(4, 0), 5.);
    EXPECT_DOUBLE_EQ(m1_copy(4, 1), 6.);
    EXPECT_DOUBLE_EQ(m1_copy(4, 2), 7.);
    EXPECT_DOUBLE_EQ(m1_copy(4, 3), 5.);
    EXPECT_DOUBLE_EQ(m1_copy(4, 4), 6.);
    EXPECT_DOUBLE_EQ(m1_copy(4, 5), 7.);
    EXPECT_DOUBLE_EQ(m1_copy(5, 0), 8.);
    EXPECT_DOUBLE_EQ(m1_copy(5, 1), 9.);
    EXPECT_DOUBLE_EQ(m1_copy(5, 2), 10.);
    EXPECT_DOUBLE_EQ(m1_copy(5, 3), 8.);
    EXPECT_DOUBLE_EQ(m1_copy(5, 4), 9.);
    EXPECT_DOUBLE_EQ(m1_copy(5, 5), 10.);

#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(ezc3d::Matrix66(ezc3d::Matrix(5, 6)), std::runtime_error);
    EXPECT_THROW(ezc3d::Matrix66(ezc3d::Matrix(6, 5)), std::runtime_error);
#endif

    ezc3d::Vector6d v_fromMul(m_toFill * ezc3d::Vector6d(2, 3, 4, 5, 6, 7));
    EXPECT_DOUBLE_EQ(v_fromMul(0), 85.);
    EXPECT_DOUBLE_EQ(v_fromMul(1), 166.);
    EXPECT_DOUBLE_EQ(v_fromMul(2), 247.);
    EXPECT_DOUBLE_EQ(v_fromMul(3), 85.);
    EXPECT_DOUBLE_EQ(v_fromMul(4), 166.);
    EXPECT_DOUBLE_EQ(v_fromMul(5), 247.);

}

TEST(Vector3d, create){
    ezc3d::Vector3d zeros;
    zeros.setZeros();
    EXPECT_DOUBLE_EQ(zeros(0), 0.0);
    EXPECT_DOUBLE_EQ(zeros(1), 0.0);
    EXPECT_DOUBLE_EQ(zeros(2), 0.0);
    EXPECT_THROW(zeros.resize(0, 0), std::runtime_error);

    ezc3d::Vector3d random(1.1, 2.2, 3.3);
    EXPECT_DOUBLE_EQ(random(0), 1.1);
    EXPECT_DOUBLE_EQ(random(1), 2.2);
    EXPECT_DOUBLE_EQ(random(2), 3.3);

    ezc3d::Vector3d random_copy(random);
    EXPECT_DOUBLE_EQ(random_copy(0), 1.1);
    EXPECT_DOUBLE_EQ(random_copy(1), 2.2);
    EXPECT_DOUBLE_EQ(random_copy(2), 3.3);

    ezc3d::Vector3d random_equal;
    random_equal = random;
    EXPECT_DOUBLE_EQ(random_equal(0), 1.1);
    EXPECT_DOUBLE_EQ(random_equal(1), 2.2);
    EXPECT_DOUBLE_EQ(random_equal(2), 3.3);
#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(random_equal = ezc3d::Matrix(4, 1), std::runtime_error);
    EXPECT_THROW(random_equal = ezc3d::Matrix(3, 2), std::runtime_error);
#endif

    EXPECT_NO_THROW(ezc3d::Vector3d(ezc3d::Matrix(3, 1)));
#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(ezc3d::Vector3d(ezc3d::Matrix(3, 2)), std::runtime_error);
#endif
}

TEST(Vector3d, unittest){
    ezc3d::Vector3d random(1.1, 2.2, 3.3);
    EXPECT_DOUBLE_EQ(random.norm(), 4.1158231254513353);
#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(random(3) = 0, std::runtime_error);
#endif

    ezc3d::Vector3d random_normalized(random);
    random_normalized.normalize();
    EXPECT_DOUBLE_EQ(random_normalized(0), 0.26726124191242445);
    EXPECT_DOUBLE_EQ(random_normalized(1), 0.5345224838248489);
    EXPECT_DOUBLE_EQ(random_normalized(2), 0.80178372573727319);
    EXPECT_DOUBLE_EQ(random_normalized.norm(), 1.0);

    ezc3d::Vector3d random2(2.2, 3.3, 4.4);
    ezc3d::Vector3d random_cross(random.cross(random2));
    EXPECT_NEAR(random_cross(0), -1.21, 1e-10);
    EXPECT_NEAR(random_cross(1), 2.42, 1e-10);
    EXPECT_NEAR(random_cross(2), -1.21, 1e-10);

    ezc3d::Vector3d random_add1(random + random2);
    EXPECT_DOUBLE_EQ(random_add1(0), 3.3);
    EXPECT_DOUBLE_EQ(random_add1(1), 5.5);
    EXPECT_DOUBLE_EQ(random_add1(2), 7.7);

    ezc3d::Vector3d random_add2 = random + random2;
    EXPECT_DOUBLE_EQ(random_add2(0), 3.3);
    EXPECT_DOUBLE_EQ(random_add2(1), 5.5);
    EXPECT_DOUBLE_EQ(random_add2(2), 7.7);

    ezc3d::Vector3d random_add3(random);
    random_add3 += random2;
    EXPECT_DOUBLE_EQ(random_add3(0), 3.3);
    EXPECT_DOUBLE_EQ(random_add3(1), 5.5);
    EXPECT_DOUBLE_EQ(random_add3(2), 7.7);
}

TEST(Vector6d, unittest){
    ezc3d::Vector6d random(1.1, 2.2, 3.3, 4.4, 5.5, 6.6);
    EXPECT_EQ(random.nbRows(), 6);
    EXPECT_EQ(random.nbCols(), 1);
    EXPECT_EQ(random.size(), 6);
    EXPECT_THROW(random.resize(0, 0), std::runtime_error);

    random.print();
#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(random(6), std::runtime_error);
    EXPECT_THROW(random(6) = 0, std::runtime_error);
#endif

#ifndef USE_MATRIX_FAST_ACCESSOR
    EXPECT_THROW(ezc3d::Vector6d dummy(ezc3d::Matrix(6, 2)), std::runtime_error);
    EXPECT_THROW(ezc3d::Vector6d dummy(ezc3d::Matrix(5, 1)), std::runtime_error);
    {
    ezc3d::Vector6d dummy2;
    EXPECT_THROW(dummy2 = ezc3d::Matrix(5, 1), std::runtime_error);
    EXPECT_THROW(dummy2 = ezc3d::Matrix(6, 2), std::runtime_error);
    }
#endif


    EXPECT_DOUBLE_EQ(random(0), 1.1);
    EXPECT_DOUBLE_EQ(random(1), 2.2);
    EXPECT_DOUBLE_EQ(random(2), 3.3);
    EXPECT_DOUBLE_EQ(random(3), 4.4);
    EXPECT_DOUBLE_EQ(random(4), 5.5);
    EXPECT_DOUBLE_EQ(random(5), 6.6);

    ezc3d::Vector6d random_copy(random);
    EXPECT_EQ(random_copy.nbRows(), 6);
    EXPECT_EQ(random_copy.nbCols(), 1);
    EXPECT_EQ(random_copy.size(), 6);
    EXPECT_DOUBLE_EQ(random_copy(0), 1.1);
    EXPECT_DOUBLE_EQ(random_copy(1), 2.2);
    EXPECT_DOUBLE_EQ(random_copy(2), 3.3);
    EXPECT_DOUBLE_EQ(random_copy(3), 4.4);
    EXPECT_DOUBLE_EQ(random_copy(4), 5.5);
    EXPECT_DOUBLE_EQ(random_copy(5), 6.6);

    random_copy(5) =  7.7;
    EXPECT_DOUBLE_EQ(random_copy(5), 7.7);

    ezc3d::Matrix random_inMat(random);

    ezc3d::Vector6d random_equal;
    random_equal = random_inMat;
    EXPECT_EQ(random_equal.nbRows(), 6);
    EXPECT_EQ(random_equal.nbCols(), 1);
    EXPECT_EQ(random_equal.size(), 6);
    EXPECT_DOUBLE_EQ(random_equal(0), 1.1);
    EXPECT_DOUBLE_EQ(random_equal(1), 2.2);
    EXPECT_DOUBLE_EQ(random_equal(2), 3.3);
    EXPECT_DOUBLE_EQ(random_equal(3), 4.4);
    EXPECT_DOUBLE_EQ(random_equal(4), 5.5);
    EXPECT_DOUBLE_EQ(random_equal(5), 6.6);
}
