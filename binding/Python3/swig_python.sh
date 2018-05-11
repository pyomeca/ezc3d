#|/usr/bash

cd build
swig -python -c++ ../example.i
mv ../example.py .
mv ../example_wrap.cxx .
c++ -c -fPIC ../../../src/ezC3D.cpp -I../../../include
c++ -c -fPIC ../../../src/Data.cpp -I../../../include
c++ -c -fPIC ../../../src/Header.cpp -I../../../include
c++ -c -fPIC ../../../src/Parameters.cpp -I../../../include
c++ -c -fPIC example_wrap.cxx -I/home/pariterre/miniconda3/envs/pyomeca/include/python3.6m -I..
c++ -shared ezC3D.o Data.o Header.o Parameters.o example_wrap.o -o _example.so 
cd ..


