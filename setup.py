from skbuild import setup
import re
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


try:
    import pypandoc
    long_description = pypandoc.convert("README.md", "r")
except(IOError, ImportError):
    long_description = open("README.md").read()

with open(f"{dir_path}/CMakeLists.txt") as file:
    for line in file:
        match = re.search(re.compile("project\\(ezc3d VERSION (\\d*\\.\\d*\\.\\d*)\\)"), line)
        if match is not None:
            version = match[1]
            break
    else:
        raise RuntimeError("Version not found")

setup(
    # NOTE: Could still add stuff like homepage or author mail, but since this isn't used to redistribute, not important
    name="ezc3d",
    version=version,
    author="Michaud, Benjamin and Begon, Mickaël",
    description="Easy to use C3D reader/writer for C++, Python and Matlab",
    long_description=long_description,
    long_description_content_type= "text/markdown",
    url = "https://github.com/pyomeca/ezc3d",
    license="MIT",
    packages=["ezc3d"],
    cmake_args=[
        "-DBUILD_EXAMPLE:BOOL=OFF",
        "-DBINDER_PYTHON3:BOOL=ON",
        "-DCMAKE_INSTALL_BINDIR=ezc3d",
        "-DCMAKE_INSTALL_LIBDIR=ezc3d"
    ],
)