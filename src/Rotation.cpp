#define EZC3D_API_EXPORTS
///
/// \file Rotation.cpp
/// \brief Implementation of Rotation class
/// \author Pariterre
/// \version 1.0
/// \date April 30th, 2022
///

#include "Rotation.h"

#include <bitset>

ezc3d::DataNS::RotationNS::Rotation::Rotation() :
    ezc3d::Matrix44(),
    _residual(-1)
{

}

ezc3d::DataNS::RotationNS::Rotation::Rotation(
        const ezc3d::DataNS::RotationNS::Rotation &r) :
    ezc3d::Matrix44(r)
{
    residual(r.residual());
}

void ezc3d::DataNS::RotationNS::Rotation::print() const {
    ezc3d::Matrix44::print();
    std::cout << "Residual = " << residual() << std::endl;
}

void ezc3d::DataNS::RotationNS::Rotation::write(
        std::fstream &f,
        float scaleFactor) const {
//    if (residual() >= 0){
//        for (size_t i = 0; i<size(); ++i) {
//            float data(static_cast<float>(_data[i]));
//            f.write(reinterpret_cast<const char*>(&data), ezc3d::DATA_TYPE::FLOAT);
//        }
//        std::bitset<8> cameraMasksBits;
//        for (size_t i = 0; i < _cameraMasks.size(); ++i){
//            if (_cameraMasks[i]){
//                cameraMasksBits[i] = 1;
//            }
//            else {
//                cameraMasksBits[i] = 0;
//            }
//        }
//        cameraMasksBits[7] = 0;
//        size_t cameraMasks(cameraMasksBits.to_ulong());
//        f.write(reinterpret_cast<const char*>(&cameraMasks), ezc3d::DATA_TYPE::WORD);
//        int residual(static_cast<int>(_residual / fabsf(scaleFactor)));
//        f.write(reinterpret_cast<const char*>(&residual), ezc3d::DATA_TYPE::WORD);
//    }
//    else {
//        float zero(0);
//        int minusOne(-16512); // 0xbf80 - 0xFFFF - 1   This is the Qualisys and Vicon value for missing marker);
//        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
//        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
//        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::FLOAT);
//        f.write(reinterpret_cast<const char*>(&zero), ezc3d::DATA_TYPE::WORD);
//        f.write(reinterpret_cast<const char*>(&minusOne), ezc3d::DATA_TYPE::WORD);
//    }
}

void ezc3d::DataNS::RotationNS::Rotation::set(
        double elem00, double elem01, double elem02, double elem03,
        double elem10, double elem11, double elem12, double elem13,
        double elem20, double elem21, double elem22, double elem23,
        double elem30, double elem31, double elem32, double elem33,
        double residual)
{
    ezc3d::Matrix44::set(
        elem00, elem01, elem02, elem03,
        elem10, elem11, elem12, elem13,
        elem20, elem21, elem22, elem23,
        elem30, elem31, elem32, elem33);
    _residual = residual;
}

void ezc3d::DataNS::RotationNS::Rotation::set(
        double elem00, double elem01, double elem02, double elem03,
        double elem10, double elem11, double elem12, double elem13,
        double elem20, double elem21, double elem22, double elem23,
        double elem30, double elem31, double elem32, double elem33)
{
    ezc3d::Matrix44::set(
        elem00, elem01, elem02, elem03,
        elem10, elem11, elem12, elem13,
        elem20, elem21, elem22, elem23,
        elem30, elem31, elem32, elem33);
    residual(0);
}

double ezc3d::DataNS::RotationNS::Rotation::residual() const {
    return _residual;
}

void ezc3d::DataNS::RotationNS::Rotation::residual(
        double residual) {
    _residual = residual;
}
