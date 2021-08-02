from skbuild import setup

setup(
    # NOTE: Could still add stuff like homepage or author mail, but since this isn't used to redistribute, not important
    name="ezc3d",
    version="1.4.4", # NOTE: Can this be automatically extracted from CMakeLists.txt without manually regex'ing it out?
    description=" Easy to use C3D reader/writer for C++, Python and Matlab",
    author="Michaud, Benjamin and Begon, Mickaël",
    license="MIT",
    packages=['ezc3d'],
    cmake_args=[
        '-DBUILD_EXAMPLE:BOOL=OFF',
        '-DBINDER_PYTHON3:BOOL=ON',
        '-DCMAKE_INSTALL_BINDIR="ezc3d"',
        '-Dezc3d_BIN_FOLDER=ezc3d',
        '-Dezc3d_LIB_FOLDER=ezc3d'
    ],
)