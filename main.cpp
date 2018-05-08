
#include <vector>
#include "ezC3D.h"

int main()
{
    ezC3D file("../markers_analogs.c3d");


    // Print the read C3D
    file.header()->print();
    file.parameters()->print();
    file.data()->print();

    // Test for specific frame Point
    std::string namePoint(file.data()->frame(10).points()->point(0).name());
    ezC3D::Data::Points3d::Point p(file.data()->frame(10).points()->point(namePoint));
    p.print();

    // Test for specific subframe Analog
    std::string nameAnalog(file.data()->frame(10).analogs()->subframe(2).channel(2).name());
    ezC3D::Data::Analogs::SubFrame::Channel c(file.data()->frame(10).analogs()->subframe(2).channel(nameAnalog));
    c.print();

    // Terminate
    std::cout << "Successfully read the c3d file" << std::endl;

    return 0;
}
