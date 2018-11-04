# EZC3D
EZC3D is an easy to use reader, modifier and writter for C3D format files. It is written en C++ with proper binders for Python and MATLAB scripting langages. 

C3D (http://c3d.org) is a format specifically designed to store biomechanics data. Hence many biomechanics softwares can produce C3D files in order to share data. However, there is a lack in the biomechanics community of an easy to use, free and open source library to read, modify and write them as needed when it gets to the data analysis. There was at some point the BTK project (https://github.com/Biomechanical-ToolKit/BTKCore) that was targeting this goal, but the project is now obsolete. 

EZC3D addresses these issues. It offers a comprehensive and light API to read and write C3D files. The source code is written in C++ allowing to be compiled and used by higher level langages thanks to SWIG (http://www.swig.org/). Still, proper interface are written on top of the SWIG binder in order to facilitate the experience of the coders in their respective langages. 

So, without further ado, let's begin C3Ding!

# How to install
There are two main ways to install EZC3D on your computer: installing the binaries from Anaconda (easiest) or compiling the source code yourself (more versatile and up to date).

## Anaconda (For Windows and Linux, Mac is coming)
The easiest way to install EZC3D is to download the binaries from anaconda (https://anaconda.org/) repositories. The project is host on the pyomeca channel (https://anaconda.org/pyomeca/ezc3d).

After having install properly an anaconda client [my suggestion would be Miniconda (https://conda.io/miniconda.html)] and loaded the desired environment to install EZC3D in, just type the following command:
```bash
conda install -c pyomeca ezc3d
```
The binaries and includes of the core of EZC3D will be installed in `bin` and `include` folders of the environment respectively. Moreover, the Python3 binder will also be installed in the environment.

## Compiling (For Windows, Linux and Mac)
The main drawback with downloading the pre-compiled version from Anaconda is that this version may be out-of-date. Moreover, since it is already compiled, it doesn't allow you to modify EZC3D if you need it. Therefore, a more versatile way to enjoy EZC3D is to compile it by yourself.

EZC3D comes with a CMake (https://cmake.org/) project. If you don't know how to use CMake, you will find many examples via Internet. The main variables to set are:

> `CMAKE_INSTALL_PREFIX` Which is the `path/to/install` EZC3D in. If you compile the Python3 binder, a valid installation of Python with Numpy should be installed relatived to this path.
>
> `BUILD_SHARED_LIBS` If you wan to build ezc3d in a shared `TRUE` or static `FALSE` library manner. Default is `TRUE`.
>
> `CMAKE_BUILD_TYPE` Which type of build you want. Options are `Debug`, `RelWithDebInfo`, `MinSizeRel` or `Release`. This is relevant only for the build done using the `make` command. Please note that you may experience a slow EZC3D library if you compile it without any optimization (i.e. `Debug`) especially on Windows. 
>
> `BUILD_EXAMPLE` If you want `TRUE` or not `FALSE` to build the C++ example. Default is `TRUE`.
> 
> `BUILD_TESTS` If you want `ON` or not `OFF` to build the tests of the project. Please note that this will download gtest (https://github.com/google/googletest). Default is `OFF`.
> 
> `BUILD_DOC` If you want `ON` or not `OFF` to build the documentation of the project. Default is `OFF`.
> 
> `BINDER_PYTHON3` If you want `ON` or not `OFF` to build the Python binder. Default is `OFF`.
> 
> `Python3_EXECUTABLE`  If `BINDER_PYTHON3` is set to `ON` then this variable should point to the Python executable. This python should have swig and Numpy installed with it. This variable should be found automatically.
> 
> `SWIG_EXECUTABLE`  If `BINDER_PYTHON3` is set to `ON` then this variable should point to the SWIG executable. This variable should be found automatically.
> 
> `BINDER_MATLAB` If you want `ON` or not `OFF` to build the MATLAB binder. Default is `OFF`.
> 
> `MATLAB_ROOT_DIR` If `BINDER_MATLAB` is set to `ON` then this variable should point to the root path of MATLAB directory. Please note that the MATLAB binder is based on MATLAB R2018a API and won't compile on earlier versions. This variable should be found automatically.
> 
> `MATLAB_ezc3d_INSTALL_DIR` If `BINDER_MATLAB` is set to `ON` then this variable should point to the path where you want to install ezc3d. Typically, this is {MY DOCUMENTS}/MATLAB. The default value is the toolbox folder of MATLAB. Please note that if you leave the default value, you will probably need to grant administrator rights to the installer. 

# How to use
The aim of EZC3D is to be, indeed, eazy to use. Still, it is a C++ library and therefore requires so time to adapt. This section aims to help you level up as fast as possible, in order to enjoy EZC3D as fast as possible

## The C++ API
The core code is written in C++, meaning you can fully create from scratch, read and write C3D from C++. There is an example code in the folder example that can be used as template to perform all the day-to-day tasks. Moreover, the test files in the tests folder can also be very useful. 

The informations that follows are the basics. 

### Create an empty yet valid C3D structure
To create a new valid yet empty C3D, just call the `c3d` class without parameter.
```C++
ezc3d::c3d c3d;
```

### Read a C3D
To read a C3D file you simply have to call the `c3d` class with a path
```C++
ezc3d::c3d c3d("path_to_c3d.c3d");
```
Please note that on Windows, the path must be `/` or `\\` separated, for obvious reasons. 

### Write a C3D
A `c3d` class is able to write itself to a file using the method `write`
```C++
ezc3d::c3d c3d;
c3d.write("path_to_c3d.c3d")
```

### Navigating into the C3D class
The C3D class mimics the C3D structures as defined by the standard, that is separated into a `header`, a `parameters` and a `data` class. You can get a const-reference to these classes by simply calling their names (see below for more specific examples)

#### Get a value from the header 
To retrieve some information from the header, just call the `header` class and then the specific information you are interested in. If for example, you want to get the frame rate of the cameras, you should do as follow:
```C++
ezc3d::c3d c3d("path_to_c3d.c3d");
float pointRate(c3d.header().frameRate());
```
Please note that the names mimics those used by the C3D format as described by the c3d.org documentation. For more information on what you can get from the header, please refer to the documentation on [header](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1Header.html).

#### Set a value to the header
It is not possible from outside to add, remove or even modify the header directly. The reason for that is that the header has a very specific formatting to be compliant to the standard. Therefore, the header will update itself if needed when the parameters class is modify. If it doesn't this is a bug that should be reported. 

#### Get a parameter
Parameters in C3D are arranged in a GROUP:PAMETER manner and the classes in EZC3D mimic this arrangement. Therefore a particular parameter always stands inside of a group. For example, if you are interested in the labels of the points, you can navigate up to the POINT group and then to the LABELS parameter. 
```C++
ezc3d::c3d c3d;
std::vector<std::string> point_labels(c3d.parameters().group("POINT").parameter("LABELS").valuesAsString());
for (size_t m = 0; m < point_labels.size(); ++m){
  std::cout << point_labels[m] << std::endl;
}
```
For more information on what you can get from the parameters, please refer to the documentation on [parameters](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1ParametersNS_1_1Parameters.html).

#### Set a parameter 
To set a parameter into a group, you must call an accessor method provided into the `c3d` class. The first parameter of the function is the name of the group to set the new parameter in, and the second parameter of the function is the actual parameter to set.
```C++
ezc3d::c3d c3d;
ezc3d::ParametersNS::GroupNS::Parameter param("name_of_my_new_parameter"); // Create a new parameter
param.set(2.0); // Give a value to the parameter
c3d.parameter("GroupName", param); // Add the parameter to the c3d structure
```
Please note that if this parameter already exist in the group named "GroupName", then this parameter is replaced by the new one. Otherwise, if it doesn't exist or the group doesn't exist, then it is added to the group. For more information on how to set a new parameter from `c3d` accessors methods, please refer to the documentation on [c3d](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1c3d.html).

#### Get data
Point and analogous data are the core of the C3D file. To understand the structure though it is essential to understand that everything is based on points. For example, the base frame rate the point frame rate, while the analogous data is based on the number of data per point frame. Therefor to get a particular point in time, you must get the data at a certain frame and specify which point you are interested in, while to get a particular analogous data you must also specify the subframe.
```C++
ezc3d::c3d c3d("path_to_c3d.c3d");
ezc3d::DataNS::Points3dNS::Point pt(new_c3d.c3d.data().frame(f).points().point(0));
pt.print();
ezc3d::DataNS::AnalogsNS::Channel channel(new_c3d.c3d.data().frame(0).analogs().subframe(0).channel("channel1"));
channel.print();
```
For more information on what you can get from the points, please refer to the documentation on [points](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1DataNS_1_1Points3dNS_1_1Points.html) or [analogs](https://pyomeca.github.io/Documentation/ezc3d/classezc3d_1_1DataNS_1_1AnalogsNS_1_1Analogs.html).

#### Set data 
There are two ways to add data to the data set. 

##### Using the c3d accessor
The first and prefered way is to add a frame via the accessors method of the class `c3d`. The parameter to send is the filled frame to add/replace to the data structure. 
Please note that the points and channel must have been declare to the parameters before adding them to the data set. This is so the whole c3d structure is properly harmonized. 
Please also note, for the same reason, that POINT:RATE and ANALOG:RATE must have been declared before adding points and analogs. 
Here is a full example that creates a new C3D, add points and analogs and print it to the console. 
```C++
// Create an empyt c3d
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

##### Using the nonConst reference
The second method is more designed for internal purpose. However, you may find yourself in situation where the normal method is just to long or restrictive for what you want to do. Then you can access directly the data via a nonConst reference. For example, you can add channels that way:
```C++
// Add a new analog to the c3d (one filled with zeros, the other one with data)
ezc3d::c3d c3d;

c3d.analog("new_analog1"); // Declare an empty channel
std::vector<ezc3d::DataNS::Frame> frames_analog;
ezc3d::DataNS::Frame frame;
// Fill the frame 
for (size_t sf = 0; sf < c3d.header().nbAnalogByFrame(); ++sf){
    ezc3d::DataNS::AnalogsNS::Channel newChannel("new_analogs2");
    newChannel.data(sf+1);
    ezc3d::DataNS::AnalogsNS::SubFrame subframes_analog;
    subframes_analog.channel(newChannel);
    frame.analogs_nonConst().subframe(subframes_analog); // The non-const reference makes it easier to add the subframe
}
for (size_t f=0; f<c3d.data().nbFrames(); ++f)
    frames_analog.push_back(frame);
c3d.analog(frames_analog);

// Print it
c3d.print();
```
Please note that this method by-passes some protection and may create invalid C3D if not used properly.

## MATLAB
MATLAB (https://www.mathworks.com/) is a prototyping langage largely used in industry and faily used by the biomecanical scientific community. Despite the growing popularity of Python as a free and open-source alternative or Octave as a very similar langage open-source, MATLAB remains an important player. Therefore EZC3D comes with a binder for MATLAB.

MATLAB stands for Matrix laboratory. As the name suggest, it is mainly used to perform operation on matrix. With that in mind, the binder was written to organize the point so it is easy to perform matrix multiplication on them. Hence, EZC3D works on MATLAB structure that separate the `header`, the `parameter` and the `data`. Into the `header` structure, you will find information on the `points`, the `analogs` and the `events`. Into the `parameter`, you will find all the groups and parameters as they appear in the C3D file. Finally, in the `data`, there is the `points` values organized into a 3d hypermatrix (XYZ x N_POINTS x N_FRAMES) and the `analogs` values organized into a 2d matrix (N_FRAMES x N_CHANNELS).

### Create an empty yet valid C3D structure
To create a new valid yet empty C3D, just call the `ezc3dRead` without any argument. 
```MATLAB
c3d = ezc3dRead();
disp(c3d.parameter.POINT.USED); % Print the number of points used
```

### Read a C3D
To read a C3D file you simply to call the `ezc3dRead` with the path to c3d as the first argument.
```MATLAB
c3d = ezc3dRead('path_to_c3d.c3d');
disp(c3d.parameter.POINT.USED); % Print the number of points used
```

### Write a C3D
To write a C3D to a file, you must call the `ezc3dWrite` function. This function waits for the path of the C3D to write and a valid structure. Please note that the header is actually ignore since it is fully constructed from required parameters. Hence, a valid structure may omit the header. Still, for simplicity, it is easier to send a structure created via the `ezc3dRead` function.
```MATLAB
% Create a valid structure to work on
c3d = ezc3dRead();

% Add a point to the structure. 
c3d.parameter.POINT.RATE = 100;
c3d.parameter.POINT.USED = c3d.parameter.POINT.USED + 1;
c3d.parameter.POINT.LABELS = [c3d.parameter.POINT.LABELS, 'NewMarkerName'];
c3d.data.points = rand(3,1,100);

% Write the C3D
ezc3dWrite('path_to_c3d.c3d', c3d);
```
## Python 3
Python (https://www.python.org/) is a scripting langage that has taken more and more importance over the past years. So much that now it is one of the prefered langage of the scientific community. It simplicity yet its large power perform a large variety of tasks makes it almost a certainty that its popularity won't decrease for the next years.

To interface the C++ code with Python, SWIG is a great tool. It creates very efficiently an interface in the target langage with minimal code to write. However, the resulting code in the target langage is far from being easy to use. Actually, it gives a mixed-API not far from the original C++ langage. When this is useful to rapidly create the interface, it lacks of user-friendlyness. EZC3D interface the C++ code using SWIG, but add a more pythonic layer on top of it. This top layer is not mandatory for the user (it is possible to call directly the SWIG interface via `ezc3d.ezc3d` instead of `ezc3d.c3d`), but the time lost to organized the data into a dictionary is insignificant compared to the ease of use this interface provides. I therefore strongly suggest to used this python interface. 

Please note, to navigate the c3d struture provided by the interface, the easiest way is to use the `keys()` method since this is a dictionary. 

### Create an empty yet valid C3D structure
To create a new valid yet empty C3D, just call the `ezc3d.c3d()` method without any argument. 
```python3
from ezc3d import c3d
c = c3d()
print(c['parameters']['POINT']['USED']['value'][0]);  # Print the number of points used
```

### Read a C3D
To read a C3D file you simply to call the `ezc3d.c3d()` with the path to c3d as the first argument.
```python3
from ezc3d import c3d
c = c3d('path_to_c3d.c3d')
print(c['parameters']['POINT']['USED']['value'][0]);  # Print the number of points used
```

### Write a C3D
To write a C3D to a file, you must call the `write` method of a c3d dictionnary. This method waits for the path of the C3D to write. Please note that the header is actually ignore since it is fully constructed from required parameters. 
```python3
import numpy as np

from ezc3d import c3d

# Create a valid dict to work on
c = c3d()

# Add a point to the structure. 
c['parameters']['POINT']['RATE']['value'] = [100]
c['parameters']['POINT']['USED']['value'] = [1]
c['parameters']['POINT']['LABELS']['value'] = ['NewMarkerName']
c['data']['points'] = np.ndarray((4,1,100));  # XYZ1 x N_POINTS x N_FRAMES

# Write the C3D
c.write('path_to_c3d.c3d')
```

# How to contribute
You are very welcome to contribute to the project! There are to main ways to contribute. 

The first way is to actually code new features to EZC3D. The easiest way to do so is to fork the project make the modifications and then open a pull request to the main project. Don't forget to add your name to the contributor in the documentation of the page if you do so!

The second way is to provide me with non-working C3D files (See the C3D Softwares section below for more details). There is another repository for test files in the pyomeca (https://github.com/pyomeca/ezc3d_c3dTestFiles). You can fork this project, add your C3D in according to the recommandations and pull request it. This will be greatly appreciated by me and the biomechanics community!

# Supported generated C3D
The software companies have loosely implemented the C3D standard proposed by http://C3D.org. Hence, there are some workaround that must be incorporated to the code to be able to read the C3D created using third-party softwares. So far, C3D from three different companies were tested. Vicon (https://www.vicon.com/), Qualisys (https://www.qualisys.com/) and Optotrak (https://www.ndigital.com/msci/products/optotrak-certus/). But I am sure there is plenty of other obscure companies or simply cases that were not tested from these companies (simply because I don't have C3D to test). If you find yourself with a bug when trying to read a C3D that should work, please open an issue and provide me with the corresponding C3D (see How to contribute). 

# Documentation
## C3D format
The C3D format is maintained by http://c3d.org. They provide recommandation on how to implement reader/writer for the format. There is a copy of the documentation PDF in the `doc` folder. You are also welcome to have a look at a newer version if they ever create an update. 

## EZC3D
The documentation is automatically generated using Doxygen (http://www.doxygen.org/). You can compile it youself if you want (by setting `BUILD_DOC` to `ON`). Otherwise, you can access a copy of it that I try to keep up-to-date in the Documentation project of pyomeca (https://pyomeca.github.io/Documentation/) by selecting `ezc3d`. 

# Troubleshoots
Despite my efforts to make a bug-free library, EZC3D may fails sometimes. If it does, please refer to the section below to know what to do. I will fill this section with the issue over time.

## Slow C3D opening
If you experience a slow C3D opening (more than 10 seconds), even for a huge C3D file. You may be in one of two cases. 

First, mak sure you are using EZC3D compiled with optimizations (RelWithDebInfo or Release). Indeed, the way C3D files are formated implies back and fourth memory allocations between points and analogs. If the optimization are turned off, it may take a little while to perform. 

If you actually are using a released level of optimization, you may actually experience a bug. You are therefore welcomed to send me the long to open C3D file so I can optimize few things by myself. Everyone will benefit!

## Non-working C3D
The C3D format allows for some pretty old and probably useless stuff. For example, you are allowed to store the points in the form of integers instead of floating points that you would scale afterwards. Since it may have make sense many years ago, it is very unlikely anyone would need this nowadays. Hence, and because I did not have any examples of such C3D to test, I decided to ignore these features (you would know easily since the code raises a `not implemented exception`). However, at some point, for some reason, you may need these features. If so, you are welcomed to open an issue and to provide me with the non-working  C3D. I will make my best to add the feature ASAP. 

Moreover, as stated before, some (all?) companies were pretty loose in their implementation of the C3D standard. Actually, the standard itself states how much you don't need to follow it, which it kind of strange, the least to say. Because of that, entire sections that are supposed to be mandatory may be missing, or checksum may have the wrong value (these are real omissions...), or anything which hasn't happened yet may occurs. There is no way for me, of course, to know that in advance, hence these exception are not implemented yet. If you encounter such files (the exception raised may be from any nature, but the most probable is segmentation fault), again do not hesitate to open an issue and to provide me with the non-working C3D. 

# Changes log
Version 0.1.0 - First working version of a C++ C3D reader. 

Version 0.2.0 - Reader and writer in C++, Python interface with SWIG for the reader, MATLAB interface for the reader and writer

Version 0.3.0 - Pythonic interface for the python reader and started to interface the writer. 
Version 0.3.1 - Documentation using Doxygen added for the C++ code, Major refactor of the code in order to harmonized it across the classes.
