#include "Data.h"
#include "Header.h"
#include "Parameters.h"
#include "ezc3d.h"
#include <vector>

int main() {
  // Create an empty fresh c3d
  ezc3d::c3d c3d_empty;
  ezc3d::ParametersNS::GroupNS::Parameter t("SCALE");
  t.set(std::vector<double>(), {0});

  // Fill it with some values
  ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
  pointRate.set(std::vector<double>() = {100}, {1});
  c3d_empty.parameter("POINT", pointRate);

  ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
  analogRate.set(std::vector<double>() = {1000}, {1});
  c3d_empty.parameter("ANALOG", analogRate);

  c3d_empty.point("new_point1");   // Add empty
  c3d_empty.point("new_point2");   // Add empty
  c3d_empty.analog("new_analog1"); // add the empty
  c3d_empty.analog("new_analog2"); // add the empty
  // Add a new frame
  ezc3d::DataNS::Frame f;
  std::vector<std::string> labels(c3d_empty.parameters()
                                      .group("POINT")
                                      .parameter("LABELS")
                                      .valuesAsString());
  int nPoints(
      c3d_empty.parameters().group("POINT").parameter("USED").valuesAsInt()[0]);
  ezc3d::DataNS::Points3dNS::Points pts;
  for (size_t i = 0; i < static_cast<size_t>(nPoints); ++i) {
    ezc3d::DataNS::Points3dNS::Point pt;
    pt.x(1.0);
    pt.y(2.0);
    pt.z(3.0);
    pts.point(pt);
  }
  ezc3d::DataNS::AnalogsNS::Analogs analog;
  ezc3d::DataNS::AnalogsNS::SubFrame subframe;
  for (size_t i = 0; i < c3d_empty.header().nbAnalogs(); ++i) {
    ezc3d::DataNS::AnalogsNS::Channel c;
    c.data(static_cast<double>(i) + 1.0);
    subframe.channel(c);
  }
  for (size_t i = 0; i < c3d_empty.header().nbAnalogByFrame(); ++i)
    analog.subframe(subframe);
  f.add(pts, analog);
  c3d_empty.frame(f);
  c3d_empty.frame(f);

  // Write the brand new c3d
  c3d_empty.write("emptyC3d.c3d");

  // Read it back!
  ezc3d::c3d emptyC3d("emptyC3d.c3d");
  emptyC3d.parameters().print();

  return 0;
}
