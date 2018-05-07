
#include <vector>
#include "ezC3D.h"

int main()
{
    ezC3D file("../markers_analogs.c3d");
    if (file.is_open())
    {

        // Print the read C3D
        file.header()->print();
        file.parameters()->print();
        file.data()->print();

        // Terminate
        std::cout << "Successfully read the c3d file" << std::endl;
        file.close();

    }
    return 0;
}
