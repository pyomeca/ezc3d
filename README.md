# EZC3D
EZC3D is an easy to use reader, modifier and writter for C3D format files. It is written en C++ with proper binders for Python and MATLAB scripting langages. 

C3D (http://c3d.org) is a format specifically designed to store biomechanics data. Hence many biomechanics softwares can produce C3D files in order to share data. However, there is a lack in the biomechanics community of an easy to use, free and open source library to read, modify and write them as needed when it gets to the data analysis. There was at some point the BTK project (https://github.com/Biomechanical-ToolKit/BTKCore) that was targeting this goal, but the project is now obsolete. 

EZC3D addresses these issues. It offers a comprehensive and light API to read and write C3D files. The source code is written in C++ allowing to be compiled and used by higher level langages thanks to SWIG (http://www.swig.org/). Still, proper interface are written on top of the SWIG binder in order to facilitate the experience of the coders in their respective langages. 

Without further ado, let's begin C3Ding!

# How to install
There are two main ways to install EZC3D on your computer: installing the binaries from Anaconda (easiest) or compiling the source code yourself (more versatile and up to date).

## Anaconda (For Windows and Linux, Mac is coming)
The easiest way to install EZC3D is to download the binaries from anaconda (https://anaconda.org/) repositories. The project is host on the pyomeca channel (https://anaconda.org/pyomeca/ezc3d).

After having install properly an anaconda client [my suggestion would be Miniconda (https://conda.io/miniconda.html)] and loaded the desired environment to install EZC3D in, just type the following command:
```bash
conda install -c pyomeca ezc3d
```
The binaries and includes of the core of EZC3D will be installed in `bin` and `include` folders of the environment respectively. Moreover, the Python3 binder will also be installed in the environment.

## Compiling
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

### Windows
particularité

# How to use
## C++

## MATLAB
(https://www.mathworks.com/)

## Python 3
(https://www.python.org/) 

# How to contribute
You are very welcome to contribute to the project, either 
Send c3d

# C3D Softwares
Loose approche
Vicon (https://www.vicon.com/) and Qualisys (https://www.qualisys.com/), to name o use this format

# Documentation

# Troubleshoots

# Changes log
