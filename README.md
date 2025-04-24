# EZC3D

<img src="logo/logo.png" width="40%" height="40%">

EZC3D is an easy to use reader, modifier and writer for C3D format files. It is written in C++ with proper binders for Python and MATLAB/Octave scripting languages. 

C3D (http://c3d.org) is a format specifically designed to store biomechanics data. Hence many biomechanics softwares can produce C3D files in order to share data. However, there is a lack in the biomechanics community of an easy to use, free and open source library to read, modify and write them as needed when it gets to the data analysis. There was at some point the BTK project (https://github.com/Biomechanical-ToolKit/BTKCore) that was targeting this goal, but the project is now obsolete. 

EZC3D addresses these issues. It offers a comprehensive and light API to read and write C3D files. The source code is written in C++ allowing to be compiled and used by higher level languages thanks to SWIG (http://www.swig.org/). Still, proper interface are written on top of the SWIG binder in order to facilitate the experience of the coders in their respective languages. 

You can get the online version of the paper for EZC3D here: [![DOI](https://joss.theoj.org/papers/10.21105/joss.02911/status.svg)](https://doi.org/10.21105/joss.02911)

So, without further ado, let's begin C3Ding!

# Table of Contents  
- [EZC3D](#ezc3d)
- [Table of Contents](#table-of-contents)
  - [Headers](#headers)
- [How to install](#how-to-install)
  - [Pip (For Python users on Windows, Linux and Mac)](#pip-for-python-users-on-windows-linux-and-mac)
  - [Anaconda (For Python users on Windows, Linux and Mac)](#anaconda-for-python-users-on-windows-linux-and-mac)
  - [Download binaries (For MATLAB users on Windows, Linux and Mac)](#download-binaries-for-matlab-users-on-windows-linux-and-mac)
  - [Compiling (For Windows, Linux and Mac)](#compiling-for-windows-linux-and-mac)
  - [Compile via `setup.py`](#compile-via-setuppy)
- [How to use](#how-to-use)
  - [The C++ API](#the-c-api)
    - [Create an empty yet valid C3D structure](#create-an-empty-yet-valid-c3d-structure)
    - [Read a C3D](#read-a-c3d)
    - [Write a C3D](#write-a-c3d)
    - [Navigating through the C3D class](#navigating-through-the-c3d-class)
    - [Copying the C3D class](#copying-the-c3d-class)
    - [Force platform filter](#force-platform-filter)
  - [MATLAB](#matlab)
    - [Create an empty yet valid C3D structure](#create-an-empty-yet-valid-c3d-structure-1)
    - [Read a C3D](#read-a-c3d-1)
    - [Write a C3D](#write-a-c3d-1)
    - [Force platform filter](#force-platform-filter-1)
  - [Octave](#octave)
  - [Python 3](#python-3)
    - [Create an empty yet valid C3D structure](#create-an-empty-yet-valid-c3d-structure-2)
    - [Read a C3D](#read-a-c3d-2)
    - [Write a C3D](#write-a-c3d-2)
    - [Force platform filter](#force-platform-filter-2)
  - [R](#r)  
- [How to contribute](#how-to-contribute)
  - [Using the test suite](#using-the-test-suite)
  - [Running the tests](#running-the-tests)
  - [Tests for the binders](#tests-for-the-binders)
    - [Matlab/Octave](#matlaboctave)
    - [Python](#python)
- [Supported generated C3D](#supported-generated-c3d)
- [Documentation](#documentation)
  - [EZC3D](#ezc3d-1)
  - [C3D format](#c3d-format)
- [Support](#support)
  - [Report issues](#report-issues)
  - [Known issues](#known-issues)
    - [Slow C3D opening](#slow-c3d-opening)
    - [Non-working C3D](#non-working-c3d)
- [Cite](#cite)

# How to install
There are two main ways to install EZC3D on your computer: the easy method, installing the binaries from pip or Anaconda (Python users) or from the Release page (Matlab users); or the hard method, compiling the source code yourself (more versatile and up to date).

## Pip (For Python users on Windows, Linux and Mac)
For Python users, the easiest way to install EZC3D is to install it using the pip command. Simply type in a shell:
```bash
pip install ezc3d
```
assuming pip is installed and, voilà! 

## Anaconda (For Python users on Windows, Linux and Mac)
For Python users, the second easiest way to install EZC3D is to download the binaries from anaconda (https://anaconda.org/) repositories (while binaries are available for Python3 and Octave, there are not any for MATLAB, apart from using the mex file produced for Octave). The project is hosted on the conda-forge channel (https://anaconda.org/conda-forge/ezc3d).

After having installed properly an anaconda client [my suggestion would be Miniconda (https://conda.io/miniconda.html)] and loaded the desired environment to install EZC3D in, just type the following command for installing the Python3 binaries:
```bash
conda install -c conda-forge ezc3d
```
or this command for installing the Octave binary:
```bash
conda install -c conda-forge ezc3d=*=*octave*
```
The binaries and includes of the core of EZC3D will be installed in `bin` and `include` folders of the environment respectively. Moreover, the Python3 or Octave binder will also be installed in the environment.

DEPRECATED: As a workaround, that it is possible to use the Octave binaries in MATLAB. The `.mex` extension must however be changed according to your operating system, namely `mexw32` or `.mexw64` for Windows (32 or 64-bits), `.mexmaci64` or `.mexmaca64` for MacOSX and `.mexa64` for Linux. This seems to recently have stopped working since MATLAB changed their API (earlier than R2021b?). Please refer to [Download binaries](#download-binaries-for-matlab-users-on-windows-linux-and-mac).


The current building status for Anaconda release is as follow.

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-ezc3d-green.svg)](https://anaconda.org/conda-forge/ezc3d) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ezc3d.svg)](https://anaconda.org/conda-forge/ezc3d) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ezc3d.svg)](https://anaconda.org/conda-forge/ezc3d) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ezc3d.svg)](https://anaconda.org/conda-forge/ezc3d) |

## Download binaries (For MATLAB users on Windows, Linux and Mac)
The MATLAB users can download the binaries directly from the Release page of `ezc3d` at this URL: [https://github.com/pyomeca/ezc3d/releases/latest](https://github.com/pyomeca/ezc3d/releases/latest). 

Once the folder is downloaded, you simply unzip it, add it to the MATLAB's path, and enjoy ezc3d!

https://github.com/user-attachments/assets/9abf62db-f750-41de-80a9-b421daad2dfa

## Compiling (For Windows, Linux and Mac)
The main drawback with downloading the pre-compiled version from Anaconda is that it may be out-of-date. Moreover, since it is already compiled, it doesn't allow you to modify EZC3D if you need it. Therefore, a more versatile way to enjoy EZC3D is to compile it by yourself.

The building status for the current EZC3D branches is as follow

| Name | Status |
| --- | --- |
| Dev | [![Build status](https://ci.appveyor.com/api/projects/status/cyiaxoflypuk4eb4/branch/dev?svg=true)](https://ci.appveyor.com/project/pariterre/ezc3d/branch/dev) |
| Test coverage | [![codecov](https://codecov.io/gh/pyomeca/ezc3d/branch/dev/graph/badge.svg?token=fc2ZGOexD1)](https://codecov.io/gh/pyomeca/ezc3d) |
| DOI | [![DOI](https://zenodo.org/badge/131555942.svg)](https://zenodo.org/badge/latestdoi/131555942) |

### Dependencies <!-- omit from toc -->
EZC3D does not rely on any external dependency. However, it comes in the form of a CMake (https://cmake.org/) project. Consequently, CMake must be installed on your computer to compile EZC3D. It can be installed from the official website or by Anaconda using the following command:
```bash
conda install -c conda-forge cmake
```

Moreover, if ones is interested in developing EZC3D, the ```googletest``` suite is required to test your modifications. Fortunately, the CMake project should download and compile the test suite for you!

When compiling the binders, some additional dependencies are required. For the Python binder, Python3 is indeed required but also *numpy* (https://numpy.org/) and *SWIG* (http://www.swig.org/). They can be installed from their respective official websites or by Anaconda using the following command:
```bash
conda install -c conda-forge numpy swig
```
For the MATLAB binder, the only additional dependency is MATLAB (https://www.mathworks.com/) itself.

For the Octave binder, the only additional dependency is Octave (https://www.gnu.org/software/octave/index) itself.
On Linux and Mac, it can be easily installed using conda:
```bash
conda install -c conda-forge octave
```
On Windows, one is required to manually install it from the website


### CMake <!-- omit from toc -->
EZC3D comes in the form of a CMake (https://cmake.org/) project. If you don't know how to use CMake, you will find many examples on Internet. For the Windows user, a quick video was made to show how to compile for the MATLAB binder [here](https://youtu.be/gWno_NXrITA). Please note that the video is made from a french computer. This should not impair the workflow, but may be a bit confusing for some english folks!

The cmake variables to set are:

> `CMAKE_INSTALL_PREFIX` Which is the `path/to/install` EZC3D in. If you compile the Python3 binder, a valid installation of Python with Numpy should be installed relative to this path.
>
> `BUILD_SHARED_LIBS` If you wan to build EZC3D in a shared `TRUE` or static `FALSE` library manner. Default is `TRUE`.
>
> `CMAKE_BUILD_TYPE` Which type of build you want. Options are `Debug`, `RelWithDebInfo`, `MinSizeRel` or `Release`. This is relevant only for the build done using the `make` command. Please note that you may experience a slow EZC3D library if you compile it without any optimization (i.e. `Debug`) especially on Windows. 
>
> `USE_MATRIX_FAST_ACCESSOR` If fast accessor should be used (`ON`) or not (`OFF`). Fast accessor is, as its name suggests faster, but do not check for sanity of the elements and can lead to segmentation faults for ill-c3d. Default is `ON`.
>
> `BUILD_EXAMPLE` If you want (`TRUE`) or not (`FALSE`) to build the C++ example. Default is `TRUE`.
> 
> `BUILD_TESTS` If you want `ON` or not `OFF` to build the tests of the project. Please note that this will automatically download gtest (https://github.com/google/googletest). Default is `OFF`.
> 
> `BUILD_DOC` If you want (`ON`) or not (`OFF`) to build the documentation of the project. Default is `OFF`.
> 
> `BINDER_PYTHON3` If you want (`ON`) or not (`OFF`) to build the Python binder. Default is `OFF`.
>
> `Python3_EXECUTABLE`  If `BINDER_PYTHON3` is set to `ON` then this variable should point to the Python executable. This python should have *SWIG* and *Numpy* installed with it. This variable should be found automatically, but Anaconda finds the base prior to the actual environment, so one should gives attention to that particular variable.
> 
> `PYTHON_INSTALL_PREFIX` The folder to install the Python binder. The default value is the site-package folder of the current Python (which may require administrator privileges)
> 
> `SWIG_EXECUTABLE`  If `BINDER_PYTHON3` is set to `ON` then this variable should point to the SWIG executable. This variable will be found automatically if `Python3_EXECUTABLE` is properly set.
> 
> `BINDER_MATLAB` If you want (`ON`) or not (`OFF`) to build the MATLAB binder. Default is `OFF`.
> 
> `MATLAB_ROOT_DIR` If `BINDER_MATLAB` is set to `ON` then this variable should point to the root path of MATLAB directory (something like "C:\\Program Files\\MATLAB\\R2022b"). Please note that the MATLAB binder is based on MATLAB R2018a API and won't compile on earlier versions. This variable should be found automatically, except on Mac where the value should manually be set to the MATLAB in the App folder.
> 
> `MATLAB_ezc3d_INSTALL_DIR` If `BINDER_MATLAB` is set to `ON` then this variable should point to the path where you want to install EZC3D. Typically, you want this to be in `{MY DOCUMENTS}/MATLAB`, which is not the default location. The default value is the toolbox folder of MATLAB, i.e., `{MATLAB_ROOT}/toolbox`. Please note that if you leave the default value, you will probably need to grant administrator rights to the installer. 
> 
> `BINDER_OCTAVE` If you want (`ON`) or not (`OFF`) to build the Octave binder. Default is `OFF`.
> 
> `OCTAVE_ROOT_DIR` If `BINDER_OCTAVE` is set to `ON` then this variable should point to the root path of Octave directory. When Octave is installed using conda, you probably don't have to set this variable. On Windows, the folder that this variable points is the folder containing `octave.vbs`. 
> 
> `Octave_ezc3d_INSTALL_DIR` If `BINDER_OCTAVE` is set to `ON` then this variable should point to the path where you want to install EZC3D. Typically, you want this to be in `{MY DOCUMENTS}/Octave`, which is not the default location. The default value is the toolbox folder on root, i.e., `/toolbox`. Please note that if you leave the default value, you will probably need to grant administrator rights to the installer. 

*Fix for MATLAB*: There is a known issue with libstdc++.so.6 in MATLAB on Ubuntu. To fix it, you need to copy the libstdc++.so.6 from your system to the MATLAB bin folder. For example, on Ubuntu 18.04, you can do the following:
```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
```
Then, you can run MATLAB from the same terminal. Or if you want to run MATLAB from the desktop, you can remove the original libstdc++.so.6 file in the MATLAB/sys/os/glnxa64 folder and replace it with a symbolic link to the system libstdc++.so.6 file with the following:
```bash
cd /usr/local/MATLAB/R2022b/sys/os/glnxa64
ln -s /lib/x86_64-linux-gnu/libstdc++.so.6
```
### VCPKG (For Windows, Linux and Mac) <!-- omit from toc -->

An automated script for compilation is offered on vcpkg. Install vcpkg by making a local clone from its GitHub repo [https://github.com/Microsoft/vcpkg](https://github.com/Microsoft/vcpkg). Then run the vcpkg-bootstrapper script to set it up. For detailed installation instructions, see [Install vcpkg](https://docs.microsoft.com/en-us/cpp/build/install-vcpkg). To integrate vcpkg with your Visual Studio or Visual Studio Code development environment, see [Integrate vcpkg](https://docs.microsoft.com/en-us/cpp/build/integrate-vcpkg). Then, to use vcpkg to install or update a library, see [Manage libraries with vcpkg](https://docs.microsoft.com/en-us/cpp/build/manage-libraries-with-vcpkg). For more information about vcpkg commands, see [vcpkg command-line reference](https://docs.microsoft.com/en-us/cpp/build/vcpkg-command-line-reference).

👀 EZC3D is available in VCPKG since [2020-11 release](https://github.com/microsoft/vcpkg/releases/tag/2020.11)

## Compile via `setup.py`
This way of "installing" is mostly for convenience when developing on ezc3d and the python-wrapper. It is **not** recommended for normal usage. Refer to [Anaconda](#anaconda-for-windows-linux-and-mac) for that, simply call:
```bash
python install .
```


# How to use
The aim of EZC3D is to be, indeed, easy to use. Still, it is a C++ library and therefore requires some time to adapt. This section aims to help you level up as fast as possible, in order to enjoy EZC3D as fast as possible. 

There are example codes for C++, Python3 and MATLAB in the folder `example` that can be used as template to perform all the day-to-day tasks. Moreover, the test files in the tests folder can also be very useful.
Octave example are not specifically provided, but it is used in the exact same way as the MATLAB binder.

## The C++ API
The core code is written in C++, meaning that you can fully create from scratch, read and write C3D from C++. 
The information that follows is a basic guide that should allow you to perform everything you want to do.

### Create an empty yet valid C3D structure
To create a new valid yet empty C3D, just call the `c3d` class without parameter.
```C++
ezc3d::c3d c3d;
```

### Read a C3D
To read a C3D file you simply have to call the `c3d` class with a path
```C++
ezc3d::c3d c3d("path/to/c3d.c3d");
```
Additionnally, a default `ignoreBadFormatting` flag can be set to `true` so file with bad formatting are read even though they are not formatted properly. This must be used with caution as it can result in segmentation fault, depending on the reason the formatting is bad.
Please note that on Windows, the path must be `/` or `\\` separated (and not only`\`), for obvious reasons. 

### Write a C3D
A `c3d` class is able to write itself to a file using the method `write`
```C++
ezc3d::c3d c3d;
c3d.write("path_to_c3d.c3d")
```

Please note that there is a `parametrizedWrite` method as well which allows for non-standard `c3d` to be written. This must be used with care as the resulting `c3d` may or may not be readable by a third-party software. That said, some software expect non-standard `c3d`.

### Navigating through the C3D class
The C3D class mimics the C3D structures as defined by the standard, that is separated into a `header`, a `parameters` and a `data` class. You can get a const-reference to these classes by simply calling their names (see below for more specific examples)

### Copying the C3D class
Please not that a copy of a c3d class will results in a shallow copy

#### Get a value from the header <!-- omit from toc -->
To retrieve some information from the header, just call the `header` class and then the specific information you are interested in. If for example, you want to get the frame rate of the cameras, you should do as follow:
```C++
ezc3d::c3d c3d("path_to_c3d.c3d");
float pointRate(c3d.header().frameRate());
```
Please note that the names mimics those used by the C3D format as described by the c3d.org documentation. For more information on what you can get from the header, please refer to the documentation on [header](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1Header.html).

#### Set a value to the header <!-- omit from toc -->
It is not possible from outside to add, remove or even modify the header directly. The reason for that is that the header has a very specific formatting to be compliant to the standard. Therefore, the header will update itself if needed when the parameters class is modified. If it doesn't this is a bug that should be reported. 

#### Get a parameter <!-- omit from toc -->
Parameters in C3D are arranged in a GROUP:PARAMETER manner and the classes in EZC3D mimic this arrangement. Therefore a particular parameter always stands inside of a group. For example, if you are interested in the labels of the points, you can navigate up to the POINT group and then to the LABELS parameter. 
```C++
ezc3d::c3d c3d;
std::vector<std::string> point_labels(c3d.parameters().group("POINT").parameter("LABELS").valuesAsString());
for (size_t m = 0; m < point_labels.size(); ++m){
  std::cout << point_labels[m] << std::endl;
}
```
For more information on what you can get from the parameters, please refer to the documentation on [parameters](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1ParametersNS_1_1Parameters.html).

#### Set a parameter <!-- omit from toc -->
To set a parameter into a group, you must call the accessor method provided into the `c3d` class. The first parameter of the function is the name of the group to set the new parameter in, and the second parameter of the function is the actual parameter to set.
```C++
ezc3d::c3d c3d;
ezc3d::ParametersNS::GroupNS::Parameter param("name_of_my_new_parameter"); // Create a new parameter
param.set(2.0); // Give a value to the parameter
c3d.parameter("GroupName", param); // Add the parameter to the c3d structure
```
Please note that if this parameter already exist in the group named "GroupName", then this parameter is replaced by the new one. Otherwise, if it doesn't exist or the group doesn't exist, then it is added to the group or the group is created then the parameter is added. For more information on how to set a new parameter from `c3d` accessors methods, please refer to the documentation on [c3d](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1c3d.html).

#### Get data <!-- omit from toc -->
Point and analogous data are the core of the C3D file (please note that rotation data are also available, but are non-standard). To understand the structure though it is essential to understand that everything is based on points. For example, the base frame rate the point frame rate, while the analogous data is based on the number of data per point frame. Therefore to get a particular point in time, you must get the data at a certain frame and specify which point you are interested in, while to get a particular analogous data you must also specify the subframe.
```C++
ezc3d::c3d c3d("path_to_c3d.c3d");
ezc3d::DataNS::Points3dNS::Point pt(new_c3d.c3d.data().frame(f).points().point(0));
pt.print();
ezc3d::DataNS::AnalogsNS::Channel channel(new_c3d.c3d.data().frame(0).analogs().subframe(0).channel("channel1"));
channel.print();
```
For more information on what you can get from the points, please refer to the documentation on [points](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1DataNS_1_1Points3dNS_1_1Points.html) or [analogs](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1DataNS_1_1AnalogsNS_1_1Analogs.html).

#### Set data <!-- omit from toc -->
There are two ways to add data to the data set. 

##### Using the c3d accessor <!-- omit from toc -->
The first and preferred way is to add a frame via the accessors method of the class `c3d`. The parameter to send is the filled frame to add/replace to the data structure. 
Please note that the points and channel must have been declare to the parameters before adding them to the data set. This is so the whole c3d structure is properly harmonized. 
Please also note, for the same reason, that POINT:RATE and ANALOG:RATE must have been declared before adding points and analogs. 
Here is a full example that creates a new C3D, add points and analogs and print it to the console. 
```C++
// Create an empy c3d
ezc3d::c3d c3d_empty;

// Declare rates
ezc3d::ParametersNS::GroupNS::Parameter pointRate("RATE");
pointRate.set(std::vector<float>() = {100}, {1});
c3d_empty.parameter("POINT", pointRate);

ezc3d::ParametersNS::GroupNS::Parameter analogRate("RATE");
analogRate.set(std::vector<float>() = {1000}, {1});
c3d_empty.parameter("ANALOG", analogRate);

// Declare the points and channels to the c3d
c3d_empty.point("new_marker1"); // Add empty
c3d_empty.point("new_marker2"); // Add empty
c3d_empty.analog("new_analog1"); // add the empty
c3d_empty.analog("new_analog2"); // add the empty

// Fill them with some random values
ezc3d::DataNS::Frame f;
std::vector<std::string>labels(c3d_empty.parameters().group("POINT").parameter("LABELS").valuesAsString());
int nPoints(c3d_empty.parameters().group("POINT").parameter("USED").valuesAsInt()[0]);
ezc3d::DataNS::Points3dNS::Points pts;
for (size_t i=0; i<static_cast<size_t>(nPoints); ++i){
    ezc3d::DataNS::Points3dNS::Point pt;
    pt.name(labels[i]);
    pt.x(1.0);
    pt.y(2.0);
    pt.z(3.0);
    pts.point(pt);
}
ezc3d::DataNS::AnalogsNS::Analogs analog;
ezc3d::DataNS::AnalogsNS::SubFrame subframe;
for (size_t i=0; i < c3d_empty.header().nbAnalogs(); ++i){
    ezc3d::DataNS::AnalogsNS::Channel c;
    c.data(i+1);
    subframe.channel(c);
}
for (size_t i=0; i < c3d_empty.header().nbAnalogByFrame(); ++i)
    analog.subframe(subframe);
    
// add them to the data set
f.add(pts, analog);
c3d_empty.frame(f);
c3d_empty.frame(f); // Why not adding a second frame?

// Print them to the console
c3d_empty.print();
```
For more information on how to set data from `c3d` accessors methods, please refer to the documentation on [c3d](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1c3d.html).

##### Using the "non-const" reference <!-- omit from toc -->
The second method is more designed for internal purpose. However, you may find yourself in situation where the normal method is just to long or restrictive for what you want to do. Then you can access directly the data via a reference. For example, you can add channels that way:
```C++
// Add a new analog to the c3d (one filled with zeros, the other one with data)
ezc3d::c3d c3d;

// Add a analog rate
ezc3d::ParametersNS::GroupNS::Parameter analog_rate("RATE");
analog_rate.set(1000.0);
c3d.parameter("ANALOG", analog_rate);

c3d.analog("new_analog1"); // Declare an empty channel (Note the name will be overridden)
std::vector<ezc3d::DataNS::Frame> frames_analog;
ezc3d::DataNS::Frame frame;
// Fill the frame
for (size_t sf = 0; sf < c3d.header().nbAnalogByFrame(); ++sf){
    ezc3d::DataNS::AnalogsNS::Channel newChannel;
    newChannel.data(sf+100);
    ezc3d::DataNS::AnalogsNS::SubFrame subframes_analog;
    subframes_analog.channel(newChannel);
    frame.analogs().subframe(subframes_analog); // The non-const reference makes it easier to add the subframe
}
c3d.frame(frame);

// Print it
c3d.print();
```
Please note that this method bypasses some protections and may create invalid C3D if not used properly.

### Force platform filter
The standard for force platforms in C3D is pretty lax.
Consequently, analyzing force platforms may be tricky. 

To help the user, `ezc3d` include a force platform analyzer filter. 
So if one is interested by extracting some process data related, they may use the filter like so:
```C++
#include <vector>
#include "ezc3d/ezc3d_all.h"

int main()
{
    ezc3d::c3d c3d("my_c3d_with_force_plate_data.c3d");
    ezc3d::Modules::ForcePlatforms pf(c3d);

    // ...
    
    return 0;
}
```

From there, each platform can be separately extracted using the STL vector
```C++
  // ...

  const auto& pf_0 = pf.forcePlatform(0); // Select the first platform

  // ...
```

Metadata can be extracted and are pretty self-explanatory. 
The following list showcase what can be extracted:
```C++
    // ...
    
    pf_0.nbFrames();      // Number of frames
    pf_0.forceUnit();     // Units of forces
    pf_0.momentUnit();    // Units of moments
    pf_0.positionUnit();  // Units of center of pressure
    pf_0.calMatrix();     // Calibration matrix
    pf_0.corners();       // Position of the corners
    pf_0.origin();        // Position of the origin
    
    // ...
```

Finally, the data can be extracted by calling the method related the desired values
```
    // ...
    int desired_frame = 0;
    
    pf_0.forces()[desired_frame];   // Forces on the platform
    pf_0.moments()[desired_frame];  // Moments on the platform in global reference frame
    pf_0.CoP()[desired_frame];      // Center of pressure
    pf_0.Tz()[desired_frame];       // Moments expressed at the center of pressure

    // These STL vectors of Vector3d can easily converted to Matrix
    ezc3d::Matrix forces(pf_0.forces());

    // ...
}
```
Please note that the moments are transported to the ground plane using the ORIGIN parameter of the FORCE_PLATFORM group.
This may or may not be what is expected (i.e., the C3D standard specifies that this is ultimately the choice of the manufacturer) with your data. 
If you know that this should not be done, you will need to compute the forces, moments, and center of pressure separately from the analog data. 

Warning: Something important to remember is that there is no easy way to detect what is the upward vector.
Consequently, `ezc3d` has to assume one. 
The most common being Z-axis pointing upward, this is what is assume.
If one has a C3D with the Y-axis pointing upward, they must transform their data accordingly in order to use the force platform filter.

## MATLAB
MATLAB (https://www.mathworks.com/) is a prototyping language largely used in industry and fairly used by the biomechanical scientific community. Despite the existence of Octave as an open-source and very similar language or the growing popularity of Python as a free and open-source alternative, MATLAB remains an important player as a programming languages. Therefore EZC3D comes with a binder for MATLAB (that can theoretically used with Octave as well with some minor changes to the CMakeLists.txt file).

MATLAB stands for MATrix LABoratory. As the name suggest, it is mainly used to perform operation on matrix. With that in mind, the binder was written to organize the point so it is easy to perform matrix multiplication on them. Hence, EZC3D works on MATLAB structure that separate the `header`, the `parameter` and the `data`. Into the `header` structure, you will find information on the `points`, the `analogs` and the `events`. Into the `parameter`, you will find all the groups and parameters as they appear in the C3D file. Finally, in the `data`, there is the `points` values organized into a 3d hypermatrix (XYZ x N_POINTS x N_FRAMES) and the `analogs` values organized into a 2d matrix (N_FRAMES x N_CHANNELS).

### Create an empty yet valid C3D structure
To create a new valid yet empty C3D, just call the `ezc3dRead` without any argument. 
```MATLAB
c3d = ezc3dRead();
disp(c3d.parameters.POINT.USED.DATA); % Print the number of points used
```

### Read a C3D
To read a C3D file you simply to call the `ezc3dRead` with the path to c3d as the first argument.
```MATLAB
c3d = ezc3dRead('path_to_c3d.c3d');
disp(c3d.parameters.POINT.USED.DATA); % Print the number of points used
```
Additionally, a default `ignoreBadFormatting` flag can be set to `true` so files with bad formatting are read even though they are not formatted properly. This must be used with caution as it can result in segmentation fault, depending on the reason the formatting is bad.

### Write a C3D
To write a C3D to a file, you must call the `ezc3dWrite` function. This function waits for the path of the C3D to write and a valid structure. Please note that the header is actually ignored since it is fully constructed from required parameters. Hence, a valid structure may omit the header. Still, for simplicity, it is easier to send a structure created via the `ezc3dRead` function.
```MATLAB
% Create a valid structure to work on
c3d = ezc3dRead();

% Add a point to the structure. 
c3d.parameters.POINT.RATE.DATA = 100;
c3d.parameters.POINT.USED.DATA = c3d.parameters.POINT.USED.DATA + 1;
c3d.parameters.POINT.LABELS.DATA = [c3d.parameters.POINT.LABELS.DATA, 'NewMarkerName'];
c3d.data.points = rand(3,1,100);

% Write the C3D
ezc3dWrite('path_to_c3d.c3d', c3d);
```

### Force platform filter
One can access the force platform if their C3D has such.

```MATLAB
[c3d, all_pf] = ezc3dRead('my_c3d_with_force_plate_data.c3d');

pf_1 = all_pf(1); % Select the first platform
```

This gives you a structure containing information on the force platform and its data.

```MATLAB
% ...

pf_1.unit_force             % Units of forces
pf_1.unit_moment            % Units of moments
pf_1.unit_position          % Units of center of pressure

pf_1.cal_matrix             % Calibration matrix
pf_1.corners                % Position of the corners
pf_1.origin                 % Position of the origin

pf_1.force                  % Force data
pf_1.moment                 % Moment data
pf_1.center_of_pressure     % Center of pressure data
pf_1.Tz                     % Moment at center of pressure data

% ...
```

## Octave
The Octave binder is almost line for line based on the MATLAB binder. Therefore, everything which is presented in the MATLAB section applies the same to the Octave binder.


## Python 3
Python (https://www.python.org/) is a scripting language that has taken more and more importance over the past years. So much that now it is one of the preferred language of the scientific community. Its simplicity yet its large power to perform a large variety of tasks makes it a certainty that its popularity won't decrease for the next few years.

To interface the C++ code with Python, SWIG is a great tool. It very rapidly creates an interface in the target language with minimal code to write. However, the resulting code in the target language can be far from being easy to use. In effect, it gives a mixed-API not far from the original C++ language, which may not comply to best practices of the target language. Whilst this is useful to rapidly create an interface, it lacks user friendliness. EZC3D interfaces the C++ code using SWIG, but add a more pythonic layer on top of it. This top layer is not mandatory for the user (it is possible to call directly the SWIG interface via `ezc3d.ezc3d` instead of `ezc3d.c3d`), but the time lost to organize the data into a dictionary is insignificant compared to the ease of use this interface provides. I therefore strongly suggest to use this python interface. 

Please note, to navigate the c3d structure provided by the interface, the easiest way is to use the attribute (using the autocompletion if your IDE allows it). As an alternative, you can access all the properties using the dictionary notation and get the keys using the `keys()` method since. 

### Create an empty yet valid C3D structure
To create a new valid yet empty C3D, just call the `ezc3d.c3d()` method without any argument. 
```python3
from ezc3d import c3d
c = c3d()
print(c['parameters']['POINT']['USED']['value'][0]);  # Print the number of points used
```

#### Understanding the output <!-- omit from toc -->
The `c3d` instance returned by the `c3d('path_to_c3d.c3d')` mimics the internal structure of a C3D, that is a `header` section, a `parameters` section and a `data` section. 
The `header` section is a standard read-only section and mostly contain redundant information with `parameters`. 
The `parameters` section is partly standard, partly based on the data of the file. 
If one wants to modify anything relating to the meta data of the c3d file, this is the section to modify.
The `data` section is a standard section containing the data for the points, the analogs and the rotations. 

To access the data, one can use the dictionary notation or the dot notation.
```python3
from ezc3d import c3d
c = c3d()
print(c['parameters']['POINT']['USED']['value'][0])
print(c.parameters.POINT.USED['value'][0])
````
The dictionary notation better reflects the internal structure of the C3D class, which makes it more reliable. 
However, it is less convenient to use as one needs to check the existing "keys" during programming.
The dot notation is mostly an accessor to the dictionary notation. 


### Read a C3D
To read a C3D file you simply to call the `ezc3d.c3d()` with the path to c3d as the first argument.
```python3
from ezc3d import c3d
c = c3d('path_to_c3d.c3d')
print(c['parameters']['POINT']['USED']['value'][0]);  # Print the number of points used
point_data = c['data']['points']
points_residuals = c['data']['meta_points']['residuals']
analog_data = c['data']['analogs']
```
> Please note that the shape of `point_data` is 4xNxT, where 4 represent the components XYZ1 (the 3D coordinates of the point add with a 1 so it can be used with homogeneous matrices), N is the number of points and T is the number of frames. 
> Similarly, and to be consistent with the point shape, the shape of `analog_data` are 1xNxT, where 1 is the value, N is the number of analogous data and T is the number of frames. 
> The `meta_point` dictionary contains information about the residuals as provided from the data acquisition system: `residuals` are the mean error of the point (a negative value meaning that the point is invalid, usually because of occlusion) and `camera_masks` being a collection of flags if the cameras had seen the point or not (unless specified in the parameter section, the cameras are the seven first, this collection of flags is limited to 7 boolean values). The dimensions of the former are 1xNxT and the dimensions of the latter are 7xNxT.

Additionally, a default `c3d(..., ignore_bad_formatting=False)` flag can be set to `true` so files with bad formatting are read even though they are not formatted properly. This must be used with caution as it can result in a segmentation fault, depending on the reason the formatting is bad.

### Write a C3D
To write a C3D to a file, you must call the `write` method of a c3d dictionary. This method waits for the path of the C3D to write. Please note that the header is actually ignored since it is fully constructed from required parameters. 

The example that follows constructs a new C3D from scratch, adding data and adding a custom parameter.
```python3
import numpy as np

import ezc3d

# Load an empty c3d structure
c3d = ezc3d.c3d()

# Fill it with random data
c3d['parameters']['POINT']['UNITS']['value'] = ['m']
c3d['parameters']['POINT']['RATE']['value'] = [100]
c3d['parameters']['POINT']['LABELS']['value'] = ('point1', 'point2', 'point3', 'point4', 'point5')
c3d['data']['points'] = np.random.rand(4, 5, 100)
c3d['data']['points'][1, :, :] = 2
c3d['data']['points'][2, :, :] = 3

c3d['parameters']['ANALOG']['RATE']['value'] = [1000]
c3d['parameters']['ANALOG']['LABELS']['value'] = ('analog1', 'analog2', 'analog3', 'analog4', 'analog5', 'analog6')
c3d['data']['analogs'] = np.random.rand(1, 6, 1000)
c3d['data']['analogs'][0, 0, :] = 4
c3d['data']['analogs'][0, 1, :] = 5
c3d['data']['analogs'][0, 2, :] = 6
c3d['data']['analogs'][0, 3, :] = 7
c3d['data']['analogs'][0, 4, :] = 8
c3d['data']['analogs'][0, 5, :] = 9

# Add a custom parameter to the POINT group
c3d.add_parameter("POINT", "newParam", [1, 2, 3])

# Add a custom parameter a new group
c3d.add_parameter("NewGroup", "newParam", ["MyParam1", "MyParam2"])

# Write the data
c3d.write("path_to_c3d.c3d")
```
> Please note that the shape of `point_data` is 4xNxT, where 4 represent the components XYZ1 (the 3D coordinates of the point add with a 1 so it can be used with homogeneous matrices), N is the number of points and T is the number of frames. 
> Similarly, and to be consistent with the point shape, the shape of `analog_data` are 1xNxT, where 1 is the value, N is the number of analogous data and T is the number of frames. 
> The `meta_point` dictionary contains information about the residuals as provided from the data acquisition system: `residuals` are the mean error of the point (a negative value meaning that the point is invalid, usually because of occlusion, the default value is 0.0) and `camera_masks` being a collection of flags if the cameras had seen the point or not (unless specified in the parameter section, the cameras are the seven first, this collection of flags is limited to 7 boolean values, the default values are `False` for all the cameras). The dimensions of the former are 1xNxT and the dimensions of the latter are 7xNxT. If no `meta_point` are provided, the default values are used. 

### Force platform filter
One can access the force platform if their C3D has such.

```python
import ezc3d
c3d = ezc3d.c3d('my_c3d_with_force_plate_data.c3d', extract_forceplat_data=True);

pf_0 = c3d["data"]["platform"][0]  # Select the first platform
```

As seen, this adds a dictionary in the data where are all the information and data are stored.
The data are in numpy array format.

```python
# ...

pf_0['unit_force']          # Units of forces
pf_0['unit_moment']         # Units of moments
pf_0['unit_position']       # Units of center of pressure

pf_0['cal_matrix']          # Calibration matrix
pf_0['corners']             # Position of the corners
pf_0['origin']              # Position of the origin

pf_0['force']               # Force data
pf_0['moment']              # Moment data
pf_0['center_of_pressure']  # Center of pressure data
pf_0['Tz']                  # Moment at center of pressure data

# ...
```

## R
R (<https://www.r-project.org/about.html>) is a programming language popular for statistical analyses and data visualization.

The `c3dr` package (<https://github.com/ropensci/c3dr>) provides an R interface for EZC3D. For more details visit the package repository.

# How to contribute
You are very welcome to contribute to the project! There are two main ways to contribute. 

The first way is to actually code new features for EZC3D. The easiest way to do so is to fork the project, make the modifications and then open a pull request to the main project. Don't forget to add your name to the contributor in the documentation of the page if you do so!

The second way is to provide me with non-working C3D files (See the C3D Softwares section below for more details). There is another repository for test files in the pyomeca (https://github.com/pyomeca/ezc3d_c3dTestFiles). You can fork this project, add your C3D in according to the recommendations and pull request it. This will be greatly appreciated by me and the biomechanics community!

## Using the test suite
EZC3D is tested with the test suite from google `gtest` (https://github.com/google/googletest). 

If you want to add or change some tests, you are very welcome to do so (actually it makes me very happy!). You should compile EZC3D with the `BUILD_TESTS` options turned on. The google test suite should download itself automatically. 

Afterwards, you can create a new test with the following function declaration
```c++
TEST(NameOfTestStructure, NameOfTest) {
  // Your test here...
}
```

You are invited to write tests for true positive, false positive, true negative and false negative using different combinations of `EXPECT_EQ` (or `EXPECT_FLOAT_EQ` if you compare float-precision numbers), `EXPECT_NE`, `ASSERT_TRUE`, `EXPECT_THROW` and `EXPECT_NO_THROW`. For a complete explanation of the google test suite, please refer to one of the numerous tutorials on the web.

I also implemented some useful functions such as `compareHeader(myFirstC3d, mySecondC3d)` and `compareData(myFirstC3d, mySecondC3d)` which strictly compares header and data respectively. If you expect differences though, these function are no use and you should copy-paste the content of them in your test (and change whatever is expected to be different). It is also possible to create a fully filled structure using the `fillC3D(c3dTestStruct& c3dStruc, bool withPoints, bool withAnalogs)` function and it can be tested with the `defaultHeaderTest` and `defaultParametersTest` function. Again, if you expect differences with the default setting, you should not use these default testing functions, but copy the relevant part in you extra test. 

## Running the tests
To run the test, navigate to the `test` folder in your build folder and run the `ezc3d_test` binary. Please note that if you are on Windows, you will have to copy all the necessary dll next to this binary. 

## Tests for the binders
If you add a new feature that exposes something to the user, you are welcomed to implement them in the binders (Matlab/Octave and Python). 

### Matlab/Octave
Running the tests for Matlab/Octave is as simple as running the tests script in `{PATH_TO_EZC3D_ROOT_FOLDER}/test/python3`. Please note that the `PATH_TO_EZC3D_ROOT_FOLDER` is the main folder (for instance downloaded from GitHub, and not the build folder); please also note that on Windows, you will have to copy the dll in the build folder. 

### Python
For Python, the tests should be run using `pytest` from the build directory with the following command: `pytest -v {PATH_TO_EZC3D_ROOT_FOLDER}/test/python3`. Please note that the `PATH_TO_EZC3D_ROOT_FOLDER` is the main folder (for instance downloaded from GitHub, and not the build folder); please also note that on Windows, you will have to copy the dll in the build folder. 

# Supported generated C3D
The software companies have loosely implemented the C3D standard proposed by http://C3D.org. Hence, there are some workaround that must be incorporated to the code to be able to read the C3D created using third-party softwares. So far, C3D from four different companies were tested. Vicon (https://www.vicon.com/), Qualisys (https://www.qualisys.com/), Optotrak (https://www.ndigital.com/msci/products/optotrak-certus/) and BTS Bioengineering (https://www.btsbioengineering.com/). But I am sure there is plenty of other obscure companies or simply cases that were not tested from these companies (simply because I don't have C3D to test). If you find yourself with a bug when trying to read a C3D that should work, please open an issue and provide me with the corresponding C3D (see How to contribute). 

# Documentation
## EZC3D
The documentation is automatically generated using Doxygen (http://www.doxygen.org/). You can compile it yourself if you want (by setting `BUILD_DOC` to `ON`). Otherwise, you can access a copy of it that I try to keep up-to-date in the Documentation project of pyomeca (https://pyomeca.github.io/Documentation/) by selecting `ezc3d` or by directly accessing it (https://pyomeca.github.io/Documentation/ezc3d/index.html). 

## C3D format
The C3D format is maintained by http://c3d.org. They provide recommendation on how to implement reader/writer for the format. There is a copy of the documentation PDF in the `doc` folder. You are also welcome to have a look at a newer version if they ever create an update. 

# Support
Despite my efforts to make a bug-free library, EZC3D may fails sometimes. If it does, please refer to the section below to know what to do. 

## Report issues
In the event you are experiencing problems with EZC3D, please have a look in the [known issues](#seek-support-and-known-issues). If it doesn't help, you are probably experiencing a new bug, you are therefore very welcome to report it. The preferred way is to open an issue in the GitHub repository (https://github.com/pyomeca/ezc3d/issues). Please report the OS you are working on, the version of EZC3D you are using (if you have compiled yourself EZC3D, you will find the version number in the CMakeList.txt file, otherwise it is the version number of the binary you downloaded), and a precise description of what the problem is. Usually, the best description is to provide a non-working c3d file with the piece of code that fails and a copy of the error message. It may happen, for privacy reasons, that the c3d cannot be distributed. If it is the case, just state it as such, and I will reach out to you so we can find a solution. 

## Known issues
This section reports some issues that are likely to occur. I will fill it over time. Please have a look here before reporting, as it may help you fix your problem much faster.

### Slow C3D opening
If you experience a slow C3D opening (more than 10 seconds), even for a huge C3D file. You may be in one of two cases. 

First, make sure you are using EZC3D compiled with optimizations (RelWithDebInfo or Release). Indeed, the way C3D files are formatted implies back and fourth memory allocations between points and analogs. If the optimizations are turned off, it may take a little while to perform. 

If you actually are using a released level of optimization, you may actually be experiencing a bug. You are therefore welcome to send me the long-to-open C3D file so I can optimize few things by myself. Everyone will benefit!

### Non-working C3D
The C3D format allows for some pretty old and probably useless stuff. For example, you are allowed to store the points in the form of integers instead of floating points that you would scale afterwards. While it may have made sense many years ago, it is very unlikely anyone would need this nowadays. Hence, and because I did not have any examples of such C3D to test, I decided to ignore these features (you would know easily since the code raises a `not implemented exception`). However, at some point, for some reason, you may need these features. If so, you are welcomed to open an issue and to provide me with the non-working C3D. I will make my best to add the feature ASAP. 

Moreover, as stated before, some (all?) companies were pretty loose in their implementation of the C3D standard. Actually, the standard itself states how much you don't need to follow it, which is kind of strange, to say the least. Because of that, entire sections that are supposed to be mandatory may be missing, or checksum may have the wrong value (these are real omissions...), or anything which hasn't happened yet may occurs. There is no way for me, of course, to know that in advance, hence these exceptions are not implemented yet. If you encounter such files (the exception raised may be of any nature, but the most probable is segmentation fault), again do not hesitate to open an issue and to provide me with the non-working C3D. 

# Cite
If you use EZC3D, we would be grateful if you could cite it as follows:

```bibtex
@article{michaudBiorbd2021,
  title = {ezc3d: An easy C3D file I/O cross-platform solution for {{C}}++, {{Python}} and {{MATLAB}}},
  shorttitle = {ezc3d},
  author = {Michaud, Benjamin and Begon, Mickaël},
  date = {2021-02-21},
  journaltitle = {Journal of Open Source Software},
  volume = {6},
  pages = {2911},
  issn = {2475-9066},
  doi = {10.21105/joss.02911},
  url = {https://joss.theoj.org/papers/10.21105/joss.02911},
  urldate = {2021-02-21},
  abstract = {Michaud et al., (2021). ezc3d: An easy C3D file I/O cross-platform solution for C++, Python and MATLAB. Journal of Open Source Software, 6(58), 2911, https://joss.theoj.org/papers/10.21105/joss.02911},
  langid = {english},
  number = {58}
}
```
