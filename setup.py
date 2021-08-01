from skbuild import setup

# TODO Move swig stuff to command line
setup(
    name="ezc3d",
    version="1.0.0",
    description="TODO",
    author="TODO",
    license="MIT",
    packages=['ezc3d'],
    cmake_args=['-DBUILD_EXAMPLE:BOOL=OFF', '-DBINDER_PYTHON3:BOOL=ON', R"-DSWIG_DIR=D:\KleineProjekte\ezc3d\swigwin-4.0.2\Lib", R"-DSWIG_EXECUTABLE=D:\KleineProjekte\ezc3d\swigwin-4.0.2\swig.exe", '-DCMAKE_INSTALL_BINDIR="ezc3d"'],
    #package_dir={"ezc3d": "binding/python3"},
    #cmake_install_dir="out",
)