---
title: 'ezc3d: An easy C3D I/O solution in C++, Python and MATLAB'
tags:
  - C3D
  - C++
  - Python
  - Matlab
authors:
  - name: Benjamin Michaud
    orcid: 0000-0002-5031-1048
    affiliation: 1
  - name: Mickaël Begon
    orcid: 0000-0002-4107-9160
    affiliation: 1
affiliations:
 - name: École de Kinésiologie et de Sciences de l'Activité Physique, Université de Montréal
   index: 1
date: October 1st, 2020
bibliography: paper.bib
---

# Summary
The *c3d* format [@crampC3dOrg2019] is an open-source standard extensively used in the field of biomechanics.
The main data collection and data analysis software can read and export them. 
It was initially designed to store three-dimensional point data and analog data (e.g., force platform or EMG).
Nowadays, by stretching the standard, companies have managed to include all sorts of theoretically non-standard biomechanical data, including for instance IMU data.
To match the needs of the community, Motion Lab Systems---who created and maintains the *c3d* format---updates the standard to match the biomechanical needs and more exotic usage of the format.

Despite being extensively used by the biomechanics community, there are surprisingly few alternatives when it comes to manipulate *c3d* files outside analyses software. 
This forces scientists to develop *ad hoc* solutions for each project, which usually involves redeveloping file I/O algorithms for each software in-house CSV file they use. 
While it would make sense to develop a portable solution once for all, as offered by the cross-platform *c3d* format, the binary nature of this format discourages them from digging into the trouble of developing such a solution.
To our knowledge, `BTK` [@barreBiomechanicalToolKitBTKCore2020] is the most mature (if not, the only) biomechanics library that provides an API to read and write these files.
Unfortunately, despite its open-sourced nature, the project has been mostly abandoned since~2016.
It is becoming more and more out-of-date as it has not been following the changes in the standard of the format nor that it will follow those to come.
Unfortunately, due to the numerous and tightly interconnected biomechanics modules, updating `BTK` without breaking it is a hard task.

Introducing the open-source `ezc3d` library which provides a light and comprehensive API to easily read and write *c3d* files. 
For the lay users, `ezc3d` is an up-to-date solution to manage *c3d* files that complies with the latest recommendation of the standard.
It also supports in-house implementations of the main biomechanics software, that is currently: Vicon, Qualisys, Optotrak, BTS and XSens. 
Fast file I/O is achieved thanks to the C++ core .
MATLAB and Python3 interfaces are also conveniently provided so one can implement `ezc3d` in their current workflow without difficulties.
In addition, since the *c3d* standard allows for multiple ways to store force platform data, a force platform analysis module is provided.
The main feature of this module is to reorient forces, moments and centre of pressure in more common reference frames so they can be directly interpreted by the user. 

# The dependencies
The `ezc3d` library was originally designed to be a dependencies-free library.
This allows for a user to easily link `ezc3d` with their projects. 
This also eases things to provide a cross-platform library, that is for Windows, Linux and Mac. 
Thus, by default, no dependencies are needed to compile and to use the API.

By nature biomechanics data are matrix based data. 
In-house linear algebra solutions were therefore developed to store and manipulate such data.
However, in-house solutions will never be as effective as dedicated linear algebra libraries. 
Hence, a fast accessor option was added for those who require an even faster file reading solution.
This fast accessor option relies on the highly effective `Eigen` linear algebra library [@eigenweb].

# Current usage of `ezc3d`
The library got the attention of the two most important software in biomechanics, Anybody [@rasmussenChapterAnyBodyModeling2019] and Opensim [@sethOpenSimSimulatingMusculoskeletal2018].
One of the main programmers of the former posted and still maintains the conda-forge recipe so `ezc3d` can be easily installed using conda as well as automatically being kept up to date.
For the latter, since the 4.0 version, Opensim decided to embrace the *c3d* format file by adding the capability to read them.
After trying different existing solutions, `ezc3d` was chosen as the default *c3d* reader back end.

# References

