// File : ezc3d_python.i
%{
#define SWIG_FILE_WITH_INIT
#include "ezc3d.h"
#include "Header.h"
#include "Data.h"
#include "Parameters.h"
#include "RotationsInfo.h"
%}

%include "numpy.i"
%fragment("NumPy_Fragments");
%init %{
    import_array();
%}
%include <std_vector.i>

%apply (int* IN_ARRAY1, int DIM1) {(int* points, int nPoints)};
%apply (int* IN_ARRAY1, int DIM1) {(int* channels, int nChannels)};

%rename(console_print) print;

%inline %{
PyObject * _get_points(const ezc3d::c3d& c3d, const std::vector<int>& points)
{
    // Get the data
    size_t nPoints(points.size());
    size_t nFrames(c3d.data().nbFrames());
    double * data = new double[4 * nPoints * nFrames];
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nPoints; ++m){
            const ezc3d::DataNS::Points3dNS::Point& point(c3d.data().frame(f).points().point(points[m]));
            data[nPoints*nFrames*0+nFrames*m+f] = point.x();
            data[nPoints*nFrames*1+nFrames*m+f] = point.y();
            data[nPoints*nFrames*2+nFrames*m+f] = point.z();
            data[nPoints*nFrames*3+nFrames*m+f] = 1;
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 4;
    arraySizes[1] = nPoints;
    arraySizes[2] = nFrames;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyObject * _get_point_residuals(
            const ezc3d::c3d& c3d,
            const std::vector<int>& points)
{
    // Get the data
    size_t nPoints(points.size());
    size_t nFrames(c3d.data().nbFrames());
    double * data = new double[1 * nPoints * nFrames];
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nPoints; ++m){
            const ezc3d::DataNS::Points3dNS::Point& point(c3d.data().frame(f).points().point(points[m]));
            data[nFrames*m+f] = point.residual();
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 1;
    arraySizes[1] = nPoints;
    arraySizes[2] = nFrames;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyObject * _get_point_camera_masks(
            const ezc3d::c3d& c3d,
            const std::vector<int>& points)
{
    // Get the data
    size_t nPoints(points.size());
    size_t nFrames(c3d.data().nbFrames());
    bool * data = new bool[7 * nPoints * nFrames];
    for (size_t f = 0; f < nFrames; ++f){
        for (size_t m = 0; m < nPoints; ++m){
            const ezc3d::DataNS::Points3dNS::Point& point(c3d.data().frame(f).points().point(points[m]));
            const std::vector<bool>& cam(point.cameraMask());
            for (size_t c = 0; c<cam.size(); ++c){
                data[nPoints*nFrames*c+nFrames*m+f] = cam[c];
            }
        }
    }

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 7;
    arraySizes[1] = nPoints;
    arraySizes[2] = nFrames;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_BOOL, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyObject * _get_analogs(const ezc3d::c3d& c3d, const std::vector<int>& analogs)
{
    // Get the data
    size_t nAnalogs(analogs.size());
    size_t nFrames(c3d.data().nbFrames());
    size_t nSubframes(c3d.header().nbAnalogByFrame());
    double * data = new double[nAnalogs * nFrames * nSubframes];
    for (size_t f = 0; f < nFrames; ++f)
        for (size_t sf = 0; sf < nSubframes; ++sf)
            for (size_t a = 0; a < nAnalogs; ++a)
                data[
                    f * nSubframes + 
                    sf + 
                    a * nSubframes * nFrames
                ] = c3d.data().frame(f).analogs().subframe(sf).channel(analogs[a]).data();

    // Export them to Python Object
    int nArraySize = 3;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 1;
    arraySizes[1] = nAnalogs;
    arraySizes[2] = nFrames * nSubframes;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyObject * _get_rotations(
            const ezc3d::c3d& c3d,
            const std::vector<int>& rotations,
            const ezc3d::DataNS::RotationNS::Info& rotationInfo
        )
{
    size_t nRotations(rotations.size());
    size_t nFrames(c3d.data().nbFrames());
    size_t nSubframes(rotationInfo.ratio());
    double * data = new double[16 * nRotations * nFrames * nSubframes];
    for (size_t f = 0; f < nFrames; ++f)
        for (size_t sf = 0; sf < nSubframes; ++sf)
            for (size_t r = 0; r < nRotations; ++r){
                const ezc3d::DataNS::RotationNS::Rotation& currentData =
                        c3d.data().frame(f).rotations().subframe(sf).rotation(rotations[r]);
                for (size_t i = 0; i<4; ++i){
                    for (size_t j = 0; j<4; ++j){
                        data[
                            f * nSubframes + 
                            sf +
                            r * nSubframes * nFrames + 
                            j * nRotations * nSubframes * nFrames + 
                            i * 4 * nRotations * nSubframes * nFrames
                        ] = currentData(i, j);
                    }
                }
            }

    // Export them to Python Object
    int nArraySize = 4;
    npy_intp * arraySizes = new npy_intp[nArraySize];
    arraySizes[0] = 4;
    arraySizes[1] = 4;
    arraySizes[2] = nRotations;
    arraySizes[3] = nFrames * nSubframes;
    PyArrayObject * c = (PyArrayObject *)PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, data);
    delete[] arraySizes;

    // Give ownership to Python so it will free the memory when needed
    PyArray_ENABLEFLAGS(c, NPY_ARRAY_OWNDATA);

    return PyArray_Return(c);
}
%}

%inline %{
PyArrayObject *helper_getPyArrayObject( PyObject *input, int type) {
  PyArrayObject *obj;

  if (PyArray_Check( input )) {
    obj = (PyArrayObject *) input;
    // RO - read-only
    if (!PyArray_ISBEHAVED_RO( obj )) {
      PyErr_SetString( PyExc_TypeError, "not algned or not in machine byte order" );
      return NULL;
    }
    int conversion_done = 0;
    obj = (PyArrayObject *) obj_to_array_allow_conversion( input, type, &conversion_done );
    if (!obj) return NULL;
  } else {
    PyErr_SetString( PyExc_TypeError, "not an array" );
    return NULL;
  }
  return obj;
}
%}

%inline %{
    void _import_numpy_data(
            ezc3d::c3d *self,
            PyArrayObject *pointsData,
            PyArrayObject *residualsData,
            PyArrayObject *cameraMasksData,
            PyArrayObject *analogData,
            PyArrayObject *rotationData
        ){
        const size_t nbFrames = PyArray_DIM(pointsData, 2);
        const size_t nbPoints = PyArray_DIM(pointsData, 1);
        const size_t nbAnalog = PyArray_DIM(analogData, 1);
        const size_t nbAnalogFrames = PyArray_DIM(analogData, 2);
        const size_t nbAnalogSubframes = nbFrames == 0 ? 0 : nbAnalogFrames / nbFrames;
       
        size_t nbRotation;
        size_t nbRotationFrames;
        size_t nbRotationSubframes;
        if (rotationData){
            nbRotation = PyArray_DIM(rotationData, 2);
            nbRotationFrames = PyArray_DIM(rotationData, 3);
            nbRotationSubframes = nbRotationFrames / nbFrames;
        }

        ezc3d::DataNS::Points3dNS::Points pts;
        ezc3d::DataNS::Points3dNS::Point pt;

        ezc3d::DataNS::AnalogsNS::Channel c;
        ezc3d::DataNS::AnalogsNS::SubFrame analogsSubframe;
        ezc3d::DataNS::AnalogsNS::Analogs analogs;

        ezc3d::DataNS::RotationNS::Rotation rot;
        ezc3d::DataNS::RotationNS::SubFrame rotationsSubframe;
        ezc3d::DataNS::RotationNS::Rotations rotations;

        std::vector<ezc3d::DataNS::Frame> allFrames;
        ezc3d::DataNS::Frame currFrame;


        for(size_t f = 0; f < nbFrames; ++f){
            for(size_t i = 0; i < nbPoints; ++i){
                const double x = *static_cast<double*>(PyArray_GETPTR3(pointsData, 0, i, f));
                const double y = *static_cast<double*>(PyArray_GETPTR3(pointsData, 1, i, f));
                const double z = *static_cast<double*>(PyArray_GETPTR3(pointsData, 2, i, f));
                pt.set(x, y, z);

                const double res = *static_cast<double*>(PyArray_GETPTR3(residualsData, 0, i, f));
                pt.residual(res);

                std::vector<bool> cameraMask;
                for(size_t j = 0; j < 7; ++j){
                    const int cam = *static_cast<int*>(PyArray_GETPTR3(cameraMasksData, j, i, f));
                    cameraMask.push_back(cam != 0);
                }   

                pt.cameraMask(cameraMask);
                pts.point(pt, i);
            }

            for(size_t sf = 0; sf < nbAnalogSubframes; ++sf){
                for(size_t i = 0; i < nbAnalog; ++i){
                    double data = *static_cast<double*>(PyArray_GETPTR3(analogData, 0, i, nbAnalogSubframes * f + sf));
                    c.data(data);
                    analogsSubframe.channel(c, i);
                }
                analogs.subframe(analogsSubframe, sf);
            }

            if (rotationData){
                for(size_t sf = 0; sf < nbRotationSubframes; ++sf){
                    for(size_t r = 0; r < nbRotation; ++r){
                        double elem00 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 0, 0, r, nbRotationSubframes * f + sf));
                        double elem10 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 1, 0, r, nbRotationSubframes * f + sf));
                        double elem20 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 2, 0, r, nbRotationSubframes * f + sf));
                        double elem30 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 3, 0, r, nbRotationSubframes * f + sf));
                        double elem01 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 0, 1, r, nbRotationSubframes * f + sf));
                        double elem11 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 1, 1, r, nbRotationSubframes * f + sf));
                        double elem21 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 2, 1, r, nbRotationSubframes * f + sf));
                        double elem31 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 3, 1, r, nbRotationSubframes * f + sf));
                        double elem02 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 0, 2, r, nbRotationSubframes * f + sf));
                        double elem12 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 1, 2, r, nbRotationSubframes * f + sf));
                        double elem22 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 2, 2, r, nbRotationSubframes * f + sf));
                        double elem32 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 3, 2, r, nbRotationSubframes * f + sf));
                        double elem03 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 0, 3, r, nbRotationSubframes * f + sf));
                        double elem13 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 1, 3, r, nbRotationSubframes * f + sf));
                        double elem23 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 2, 3, r, nbRotationSubframes * f + sf));
                        double elem33 = *static_cast<double*>(PyArray_GETPTR4(rotationData, 3, 3, r, nbRotationSubframes * f + sf));
                        double reliability =
                                std::isnan(elem00 + elem01 + elem02 + elem03 +
                                           elem10 + elem11 + elem12 + elem13 +
                                           elem20 + elem21 + elem22 + elem23 +
                                           elem30 + elem31 + elem32 + elem33)
                                ? -1. : 0.;

                        rot.set(elem00, elem01, elem02, elem03,
                                elem10, elem11, elem12, elem13,
                                elem20, elem21, elem22, elem23,
                                elem30, elem31, elem32, elem33,
                                reliability);
                        rotationsSubframe.rotation(rot, r);
                    }
                    rotations.subframe(rotationsSubframe, sf);
                }
            }

            currFrame.add(pts, analogs, rotations);
            allFrames.push_back(currFrame);
        }
        self->frames(allFrames);
    }
%}

%extend ezc3d::c3d
{
    // Extend c3d class to "import" data from numpy-arrays into object efficiently
    void import_numpy_data(
            PyObject *pointsData,
            PyObject *residualsData,
            PyObject *cameraMasksData,
            PyObject *analogData,
            PyObject *rotationsData
        ){
        PyArrayObject *pointsDataArr = helper_getPyArrayObject(pointsData, NPY_DOUBLE);
        PyArrayObject *residualsDataArr = helper_getPyArrayObject(residualsData, NPY_DOUBLE);
        PyArrayObject *cameraMasksDataArr = helper_getPyArrayObject(cameraMasksData, NPY_DOUBLE);
        PyArrayObject *analogDataArr = helper_getPyArrayObject(analogData, NPY_DOUBLE);
        PyArrayObject *rotationDataArr = nullptr;
        if (rotationsData != Py_None)
            rotationDataArr = helper_getPyArrayObject(rotationsData, NPY_DOUBLE);
        _import_numpy_data(self, pointsDataArr, residualsDataArr, cameraMasksDataArr, analogDataArr, rotationDataArr);
    }

    // Extend c3d class to get an easy accessor to data points
    PyObject * get_points(){
        std::vector<int> points;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            points.push_back(i);
        return _get_points(*self, points);
    }

    PyObject * get_points(int* points, int nPoints)
    {
        std::vector<int> _points;
        for (int i = 0; i < nPoints; ++i)
            _points.push_back(points[i]);
        return _get_points(*self, _points);
    }

    PyObject * get_points(int point)
    {
        std::vector<int> points;
        points.push_back(point);
        return _get_points(*self, points);
    }


    // Extend c3d class to get an easy accessor to data point residuals
    PyObject * get_point_residuals(){
        std::vector<int> points;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            points.push_back(i);
        return _get_point_residuals(*self, points);
    }

    PyObject * get_point_residuals(int* points, int nPoints)
    {
        std::vector<int> _points;
        for (int i = 0; i < nPoints; ++i)
            _points.push_back(points[i]);
        return _get_point_residuals(*self, _points);
    }

    PyObject * get_point_residuals(int point)
    {
        std::vector<int> points;
        points.push_back(point);
        return _get_point_residuals(*self, points);
    }



    // Extend c3d class to get an easy accessor to data point camera masks
    PyObject * get_point_camera_masks(){
        std::vector<int> points;
        for (int i = 0; i < self->header().nb3dPoints(); ++i)
            points.push_back(i);
        return _get_point_camera_masks(*self, points);
    }

    PyObject * get_point_residuals(int* points, int nPoints)
    {
        std::vector<int> _points;
        for (int i = 0; i < nPoints; ++i)
            _points.push_back(points[i]);
        return _get_point_camera_masks(*self, _points);
    }

    PyObject * get_point_residuals(int point)
    {
        std::vector<int> points;
        points.push_back(point);
        return _get_point_camera_masks(*self, points);
    }


    // Extend c3d class to get an easy accessor to data points
    PyObject * get_analogs(){
        std::vector<int> channels;
        for (int i = 0; i < self->header().nbAnalogs(); ++i)
            channels.push_back(i);
        return _get_analogs(*self, channels);
    }

    PyObject * get_analogs(int* channels, int nChannels)
    {
        std::vector<int> _channels;
        for (int i = 0; i < nChannels; ++i)
            _channels.push_back(channels[i]);
        return _get_analogs(*self, _channels);
    }

    PyObject * get_analogs(int channel)
    {
        std::vector<int> channels;
        channels.push_back(channel);
        return _get_analogs(*self, channels);
    }


    // Extend c3d class to get an easy accessor to data points
    PyObject * get_rotations(){
        // Get the data
        ezc3d::DataNS::RotationNS::Info rotationInfo(*self);
        std::vector<int> rotations;
        for (int i = 0; i < rotationInfo.used(); ++i)
            rotations.push_back(i);
        return _get_rotations(*self, rotations, rotationInfo);
    }

    PyObject * get_rotations(int* rotations, int nRotations)
    {
        ezc3d::DataNS::RotationNS::Info rotationInfo(*self);
        std::vector<int> _rotations;
        for (int i = 0; i < nRotations; ++i)
            _rotations.push_back(rotations[i]);
        return _get_rotations(*self, _rotations, rotationInfo);
    }

    PyObject * get_rotations(int rotation)
    {
        ezc3d::DataNS::RotationNS::Info rotationInfo(*self);
        std::vector<int> rotations;
        rotations.push_back(rotation);
        return _get_rotations(*self, rotations, rotationInfo);
    }
}


%extend ezc3d::Matrix
{
    PyObject* to_array(){
        size_t nRows($self->nbRows());
        size_t nCols($self->nbCols());
        int nArraySize(2);
        npy_intp * arraySizes = new npy_intp[nArraySize];
        arraySizes[0] = nRows;
        arraySizes[1] = nCols;

        double * mat = new double[nRows*nCols];
        unsigned int k(0);
        for (unsigned int i=0; i<nRows; ++i){
            for (unsigned int j=0; j<nCols; ++j){
                mat[k] = (*$self)(i,j);
                ++k;
            }
        }
        PyObject* output = PyArray_SimpleNewFromData(nArraySize,arraySizes,NPY_DOUBLE, mat);
        PyArray_ENABLEFLAGS((PyArrayObject *)output, NPY_ARRAY_OWNDATA);
        delete[] arraySizes;
        return output;
    };
}

%include ../ezc3d.i



