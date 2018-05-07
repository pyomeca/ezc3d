
#include <vector>
#include "ezC3D.h"

int main()
{
    ezC3D file("../markers_analogs.c3d");
    if (file.is_open())
    {
        // Read the 3dPoints data
        std::cout << "Points reading" << std::endl;
        // Firstly read a dummy value just prior to the data so it moves the pointer to the right place
        file.readInt(ezC3D::READ_SIZE::BYTE, 256*ezC3D::READ_SIZE::WORD*(file.header()->parametersAddress()-1) + 256*ezC3D::READ_SIZE::WORD*file.parameters()->nbParamBlock() - ezC3D::READ_SIZE::BYTE, std::ios::beg); // "- BYTE" so it is just prior
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
