
#include <vector>
#include "ezc3d.h"
#include <chrono>

// SANDBOX
int main()
{
    //ezc3d::c3d c3d("markers_analogs.c3d");
    //c3d.write("tata.c3d");
    ezc3d::c3d c3d_2("/home/laboratoire/mnt/F/Data/Archet/Poids_et_deformation/RAW/2017_05_31_AMd/Musique/tata.c3d");
}


// EXAMPLE TO BUILD
//int main()
//{
//    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 3; ++i){
//        ezc3d::c3d c3d("markers_analogs.c3d");
//        std::cout << "coucou" << std::endl;
//    }
//    auto stop = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

//    std::cout << double(duration.count()) /1000/1000 << std::endl;


//    // Print the read c3d
//    ezc3d::c3d c3d("markers_analogs.c3d");
//    c3d.header().print();
//    c3d.parameters().print();
//    //c3d.data().print(); // Commented for time issue (it is long to plot all the data)


//    // Test for specific frame Point
//    std::string namePoint(c3d.data().frame(10).points().point(0).name());
//    ezc3d::DataNS::Points3dNS::Point p(c3d.data().frame(10).points().point(namePoint));
//    p.print();

//    // Test for specific subframe Analog
//    std::string nameAnalog(c3d.data().frame(10).analogs().subframe(2).channel(2).name());
//    ezc3d::DataNS::AnalogsNS::Channel c(c3d.data().frame(10).analogs().subframe(2).channel(nameAnalog));
//    c.print();
//    c = c3d.data().frame(10).analogs().subframe(2).channel(3);
//    c.print();

//    // Terminate
//    std::cout << "Successfully read and printed the c3d file" << std::endl;

//    return 0;
//}
