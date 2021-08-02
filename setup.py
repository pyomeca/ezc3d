from skbuild import setup
import re
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

with open(f"{dir_path}/CMakeLists.txt") as file:
    for line in file:
        match = re.search("project\(ezc3d VERSION ([0-9].[0-9].[0-9])\)", line)
        if match is not None:
            version = match[1]
            break
    else:
        raise RuntimeError("Version not found")

setup(
    # NOTE: Could still add stuff like homepage or author mail, but since this isn't used to redistribute, not important
    name="ezc3d",
    version=version,
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