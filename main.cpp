
#include <vector>
#include "ezC3D.h"

int main()
{
    ezC3D file("../markers_analogs.c3d");
    if (file.is_open())
    {

        // Read the Parameters Header
        int parametersStart     (file.readInt(1*ezC3D::READ_SIZE::BYTE, 256*ezC3D::READ_SIZE::WORD*(file.header()->parametersAddress()-1), std::ios::beg));   // Byte 1 ==> if 1 then it starts at byte 3 otherwise at byte 512*parametersStart
        int iChecksum2          (file.readInt(1*ezC3D::READ_SIZE::BYTE));            // Byte 2 ==> should be 80 if it is a c3d
        int nbParamBlock        (file.readInt(1*ezC3D::READ_SIZE::BYTE));            // Byte 3 ==> Number of parameter blocks to follow
        int processorType       (file.readInt(1*ezC3D::READ_SIZE::BYTE));            // Byte 4 ==> Processor type (83 + [1 Inter, 2 DEC, 3 MIPS])

        std::cout << "Parameters header" << std::endl;
        std::cout << "reservedParam1 = " << parametersStart << std::endl;
        std::cout << "reservedParam2 = " << iChecksum2 << std::endl;
        std::cout << "nbParamBlock = " << nbParamBlock << std::endl;
        std::cout << "processorType = " << processorType << std::endl;
        std::cout << std::endl;

        // Parameters or group
        bool finishedReading(false);
        int nextParamByteInFile((int)file.tellg() + parametersStart - ezC3D::READ_SIZE::BYTE);
        while (!finishedReading)
        {
            // Check if we spontaneously got to the next parameter. Otherwise c3d is messed up
            if (file.tellg() != nextParamByteInFile)
                throw std::ios_base::failure("Bad c3d formatting");

            // Nb of char in the group name, locked if negative, 0 if we finished the section
            int nbCharInName       (file.readInt(1*ezC3D::READ_SIZE::BYTE));
            if (nbCharInName == 0)
                break;
            bool isLocked(false);
            if (nbCharInName < 0){
                nbCharInName = -nbCharInName;
                isLocked = true;
            }

            // Group ID always negative for groups and positive parameter of group ID
            int id(file.readInt(1*ezC3D::READ_SIZE::BYTE));
            std::string name(file.readString(nbCharInName*ezC3D::READ_SIZE::BYTE));

            // number of byte to the next group from here
            int offsetNext((int)file.readUint(2*ezC3D::READ_SIZE::BYTE));
            if (offsetNext == 0)
                finishedReading = true;
            nextParamByteInFile = (int)file.tellg() + offsetNext - ezC3D::READ_SIZE::WORD;

            std::cout << "nbCharInName = " << nbCharInName << std::endl;
            std::cout << "isLocked = " << isLocked << std::endl;
            std::cout << "nbGroupID = " << id << std::endl;
            std::cout << "groupName = " << name << std::endl;
            std::cout << "offSetNextGroup = " << offsetNext << std::endl;

            if (id < 0){
                std::cout << "Group " << id << std::endl;
            } else {
                // -1 sizeof(char), 1 byte, 2 int, 4 float
                int lengthInByte        (file.readInt(1*ezC3D::READ_SIZE::BYTE));

                // number of dimension of parameter (0 for scalar)
                int nDimensions         (file.readInt(1*ezC3D::READ_SIZE::BYTE));
                std::vector<int> dimension;
                if (nDimensions == 0) // In the special case of a scalar
                    dimension.push_back(1);
                else // otherwise it's a matrix
                    for (int i=0; i<nDimensions; ++i)
                        dimension.push_back (file.readInt(1*ezC3D::READ_SIZE::BYTE));    // Read the dimension size of the matrix

                // Read the data for the parameters
                std::vector<int> param_data_int;
                std::vector<std::string> param_data_string;
                if (lengthInByte > 0)
                    file.readMatrix(lengthInByte, dimension, param_data_int);
                else {
                    std::vector<std::string> param_data_string_tp;
                    file.readMatrix(dimension, param_data_string_tp);
                    // Vicon c3d organize its text in column-wise format, I am not sure if
                    // this is a standard or a custom made stuff
                    if (dimension.size() == 1){
                        std::string tp;
                        for (int i = 0; i < dimension[0]; ++i)
                            tp += param_data_string_tp[i];
                        param_data_string.push_back(tp);
                    }
                    else if (dimension.size() == 2){
                        int idx(0);
                        for (int i = 0; i < dimension[1]; ++i){
                            std::string tp;
                            for (int j = 0; j < dimension[0]; ++j){
                                tp += param_data_string_tp[idx];
                                ++idx;
                            }
                            param_data_string.push_back(tp);
                        }
                    }
                    else
                        throw std::ios_base::failure("Parsing char on matrix other than 2d or 1d matrix is not implemented yet");
                }

                std::cout << "Parameter " << id << std::endl;
                std::cout << "lengthInByte = " << lengthInByte << std::endl;
                std::cout << "nDimensions = " << nDimensions << std::endl;
                for (int i = 0; i< dimension.size(); ++i)
                    std::cout << "dimension[" << i << "] = " << dimension[i] << std::endl;
                for (int i = 0; i< param_data_int.size(); ++i)
                    std::cout << "param_data_int[" << i << "] = " << param_data_int[i] << std::endl;
                for (int i = 0; i< param_data_string.size(); ++i)
                    std::cout << "param_data_string[" << i << "] = " << param_data_string[i] << std::endl;
            }

            // Byte 5+nbCharInName ==> Number of characters in group description
            int nbCharInDesc(file.readInt(1*ezC3D::READ_SIZE::BYTE));
            // Byte 6+nbCharInName ==> Group description
            std::string desc(file.readString(nbCharInDesc));

            std::cout << "nbCharInDesc = " << nbCharInDesc << std::endl;
            std::cout << "desc = " << desc << std::endl;
            std::cout << std::endl;
        }

        // Read the 3dPoints data
        std::cout << "Points reading" << std::endl;
        // Firstly read a dummy value just prior to the data so it moves the pointer to the right place
        file.readInt(ezC3D::READ_SIZE::BYTE, 256*ezC3D::READ_SIZE::WORD*(file.header()->parametersAddress()-1) + 256*ezC3D::READ_SIZE::WORD*nbParamBlock - ezC3D::READ_SIZE::BYTE, std::ios::beg); // "- BYTE" so it is just prior
        std::vector<ezC3D::Frame> allFrames;
        for (int j = 0; j < file.header()->nbFrames(); ++j){
            ezC3D::Frame f;
            std::cout << "Frame " << j << ":" << std::endl;
            if (file.header()->scaleFactor() < 0){
                ezC3D::Point3d p;
                for (int i = 0; i < file.header()->nb3dPoints(); ++i){
                    std::cout << "Point " << i << " = [";
                    p.x(file.readFloat());
                    p.y(file.readFloat());
                    p.z(file.readFloat());
                    p.residual(file.readFloat());
                    f.add(p);
                    std::cout << p.x() << ", " << p.y() << ", " << p.z() << "]; ";
                    std::cout << "Residual = " << p.residual() << std::endl;
                }

                for (int k = 0; k < file.header()->nbAnalogByFrame(); ++k){
                    ezC3D::Analog a;
                    std::cout << "Frame " << j << "." << k << std::endl;
                    for (int i = 0; i < file.header()->nbAnalogs(); ++i){
                        a.addChannel(file.readFloat());
                    }
                    a.print();
                    f.add(a);
                }
            }
            else
                throw std::invalid_argument("Points were recorded using int number which is not implemented yet");
            allFrames.push_back(f);
            std::cout << std::endl;
        }

        // Terminate
        std::cout << "Successfully read the c3d file" << std::endl;
        file.close();

    }
    return 0;
}
