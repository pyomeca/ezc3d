#ifndef PARAMETER_H
#define PARAMETER_H
///
/// \file Parameter.h
/// \brief Declaration of Parameter class
/// \author Pariterre
/// \version 1.0
/// \date October 17th, 2018
///

#include "ezc3d.h"

///
/// \brief Parameter of a C3D file
///
class EZC3D_API ezc3d::ParametersNS::GroupNS::Parameter{
    //---- CONSTRUCTOR ----//
public:
    ///
    /// \brief Create an empty parameter
    /// \param name The name of the parameter
    /// \param description The description of the parameter
    ///
    Parameter(
            const std::string &name = "",
            const std::string &description = "");


    //---- STREAM ----//
public:
    ///
    /// \brief Print the parameter and its values
    ///
    void print() const;

    ///
    /// \brief Write the parameter to an opened file
    /// \param f Already opened fstream file with write access
    /// \param groupIdx Index of the group that this particular parameter is in
    /// \param dataStartPosition The position in the file where the data start (special case for POINT:DATA_START parameter)
    ///
    /// Write the parameter and its values to a file
    ///
    void write(
            std::fstream &f,
            int groupIdx,
            std::streampos &dataStartPosition) const;

protected:
    ///
    /// \brief Write a matrix of parameter to a file
    /// \param f The file stream already opened with write access
    /// \param dim The dimension of the matrix of parameter
    /// \param currentIdx Internal variable that keep track where it is in the matrix recursive calls. It should be set to 0 when called the first time.
    /// \param cmp Internal variable that keep track where it is in the parameter recursive calls. It should be set to 0 when called the first time.
    /// \return The internal variable cmp. It should be ignore by the user.
    ///
    size_t writeImbricatedParameter(
            std::fstream &f,
            const std::vector<size_t> &dim,
            size_t currentIdx=0, size_t cmp=0) const;

public:
    ///
    /// \brief Read and store a parameter from an opened C3D file
    /// \param c3d C3D reference to copy the data in
    /// \param params Reference to a valid parameter
    /// \param file The file stream already opened with read access
    /// \param nbCharInName The number of character of the parameter name
    /// \return The position in the file of the next Group/Parameter
    ///
    int read(
            c3d &c3d,
            const Parameters &params,
            std::fstream &file,
            int nbCharInName);


    //---- METADATA ----//
protected:
    std::string _name; ///< The name of the parameter
    std::string _description; ///< The description of the parameter
    bool _isLocked; ///< The lock status of the parameter
    ezc3d::DATA_TYPE _data_type; ///< The type of data (int, float or string) the parameter is

public:
    ///
    /// \brief Get the name of the parameter
    /// \return The name of the parameter
    ///
    const std::string& name() const;

    ///
    /// \brief Set the name of the parameter
    /// \param name The name of the parameter
    ///
    void name(
            const std::string& name);

    ///
    /// \brief Get the description of the parameter
    /// \return The description of the parameter
    ///
    const std::string& description() const;

    ///
    /// \brief Set the description of the parameter
    /// \param description The description of the parameter
    ///
    void description(
            const std::string& description);

    ///
    /// \brief Get the locking status of the parameter
    /// \return The locking status of the parameter
    ///
    bool isLocked() const;

    ///
    /// \brief Set the locking status of the parameter to true
    ///
    void lock();

    ///
    /// \brief Set the locking status of the parameter to false
    ///
    void unlock();


    //---- DATA DIMENSIONS ----//
protected:
    std::vector<size_t> _dimension; ///< Mapping of the data vector to the matrix of parameter

    ///
    /// \brief longestElement Compute the longest element in the holder. This is interesting solely for char
    /// \return The longest element. If 0 is return, then no element is present
    ///
    size_t longestElement() const;
public:
    ///
    /// \brief Return the vector mapping the dimension of the parameter matrix
    /// \return The vector mapping the dimension of the parameter matrix
    ///
    /// A parameter can have multiple values organized in vector or matrix. This parameter keeps track
    /// of the dimension of the parameter. For example, if dimension == {2, 3, 4}, then the values of the
    /// parameters are organized into a 3-dimensional fashion with rows == 2, columns == 3 and sheets == 4.
    ///
    const std::vector<size_t>& dimension() const;

    ///
    /// \brief Check if the dimension parameter is consistent with the values
    /// \param dataSize The size of the STL vector of the parameter
    /// \param dimension The dimension mapping of the parameter matrix
    /// \return If the dimension is consistent with the parameter size
    ///
    /// The dimension of a parameter would be considered consistent if the product of all its values
    /// equals the dataSize. For example, if dimension == {2, 3, 4}, a consistent dataSize would be
    /// 24 (2*3*4).
    ///
    bool isDimensionConsistent(
            size_t dataSize,
            const std::vector<size_t>& dimension) const;


    //---- DATA ----//
protected:
    bool _isEmpty; ///< If the parameter is empty

    ///
    /// \brief Set the empty flag to true if the parameter is actually empty
    ///
    void setEmptyFlag();

    std::vector<int> _param_data_int; ///< Parameter values if the parameter type is DATA_TYPE::BYTE of DATA_TYPE::INT
    std::vector<double> _param_data_double; ///< Parameter values if the parameter type is DATA_TYPE::FLOAT
    std::vector<std::string> _param_data_string; ///< Parameter values if the parameter type is DATA_TYPE::CHAR

public:
    ///
    /// \brief Return the type of the data
    /// \return The type of the data
    ///
    ezc3d::DATA_TYPE type() const;

    ///
    /// \brief Return the vector of values of the parameter
    /// \return The vector of values of the parameter
    ///
    /// Return the vector of values that was previously stored into the parameter.
    ///
    /// Throw an std::invalid_argument if the asked type was wrong (i.e. not DATA_TYPE::BYTE).
    ///
    const std::vector<int>& valuesAsByte() const;

    ///
    /// \brief Return the vector of values of the parameter
    /// \return The vector of values of the parameter
    ///
    /// Return the vector of values that was previously stored into the parameter.
    ///
    /// Throw an std::invalid_argument if the asked type was wrong (i.e. not DATA_TYPE::INT).
    ///
    const std::vector<int>& valuesAsInt() const;

    ///
    /// \brief Return the vector of values of the parameter
    /// \return The vector of values of the parameter
    ///
    /// Return the vector of values that was previously stored into the parameter.
    ///
    /// Throw an std::invalid_argument if the asked type was wrong (i.e. not DATA_TYPE::FLOAT).
    /// Internally, float are converted to double and converted back to float when written to the file
    ///
    const std::vector<double>& valuesAsDouble() const;

    ///
    /// \brief Return the vector of values of the parameter
    /// \return The vector of values of the parameter
    ///
    /// Return the vector of values that was previously stored into the parameter.
    ///
    /// Throw an std::invalid_argument if the asked type was wrong (i.e. not DATA_TYPE::STRING).
    ///
    const std::vector<std::string>& valuesAsString() const;

    ///
    /// \brief Set the integer scalar value for the parameter
    /// \param data The integer data
    ///
    /// Set the scalar value for a parameter assuming this value is an integer.
    ///
    void set(
            int data);

    ///
    /// \brief Set the size_t scalar value as integer for the parameter
    /// \param data The size_t scalar data
    ///
    /// Set the scalar value as integer for a parameter assuming this value is an size_t.
    ///
    void set(
            size_t data);

    ///
    /// \brief Set the integer vector of values for the parameter
    /// \param data The integer data
    /// \param dimension The vector mapping of the dimension of the parameter matrix
    ///
    /// Set the values for a parameter assuming these values are integers.
    /// If no dimension mapping is provided, it assumes to be a scalar if the data is the size of 1 or a vector
    /// if the data has a size higher than 1.
    ///
    void set(
            const std::vector<int>& data,
            const std::vector<size_t>& dimension = {});

    ///
    /// \brief Set the float scalar value for the parameter
    /// \param data The float data
    ///
    /// Set the scalar value for a parameter assuming this value is a float.
    ///
    void set(
            float data);

    ///
    /// \brief Set the double scalar value as float for the parameter
    /// \param data The double data
    ///
    /// Set the scalar value as a flot for a parameter assuming this value is a double.
    ///
    void set(
            double data);

    ///
    /// \brief Set the float vector of values for the parameter
    /// \param data The float data
    /// \param dimension The vector mapping of the dimension of the parameter matrix
    ///
    /// Set the values for a parameter assuming these values are floats.
    /// If no dimension mapping is provided, it assumes to be a scalar if the data is the size of 1 or a vector
    /// if the data has a size higher than 1.
    ///
    void set(
            const std::vector<double>& data,
            const std::vector<size_t>& dimension = {});

    ///
    /// \brief Set the single string value for the parameter
    /// \param data The string data
    ///
    void set(
            const std::string& data);

    ///
    /// \brief Set the vector of strings for the parameter
    /// \param data The vector of strings
    /// \param dimension The vector mapping of the dimension of the parameter matrix
    ///
    /// Set the vector of strings for a parameter.
    /// If no dimension mapping is provided, it assumes to be a single string if the data is the size of 1 or a vector
    /// if the data has a size higher than 1.
    ///
    void set(
            const std::vector<std::string>& data,
            const std::vector<size_t>& dimension = {});

};


#endif
