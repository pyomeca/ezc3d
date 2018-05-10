
#include <vector>
#include "ezC3D.h"

int main()
{
    ezC3D c3d("../markers_analogs.c3d");


    // Print the read C3D
    c3d.header().print();
    //c3d.parameters()->print();
    //c3d.data()->print();


    // Test for specific frame Point
    std::string namePoint(c3d.data().frame(10).points()->point(0).name());
    ezC3D_NAMESPACE::Data::Points3d::Point p(c3d.data().frame(10).points()->point(namePoint));
    p.print();

    // Test for specific subframe Analog
    std::string nameAnalog(c3d.data().frame(10).analogs()->subframe(2).channel(2).name());
    ezC3D_NAMESPACE::Data::Analogs::SubFrame::Channel c(c3d.data().frame(10).analogs()->subframe(2).channel(nameAnalog));
    c.print();
    c = c3d.data().frame(10).analogs()->subframe(2).channel(3);
    c.print();

    // Terminate
    std::cout << "Successfully read the c3d file" << std::endl;

    return 0;
}
