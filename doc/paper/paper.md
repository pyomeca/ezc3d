---
title: '`ezc3d`: An easy C3D file I/O cross-platform solution for C++, Python and MATLAB'
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
 - name: École de Kinésiologie et des Sciences de l'Activité Physique, Université de Montréal
   index: 1
date: February 18th, 2021
bibliography: paper.bib
---
# Summary
This work introduces the open source `ezc3d` library which provides a light and comprehensive API to easily read and write *c3d* files. 
The C++ core provides a fast file I/O library, and convenient MATLAB and Python3 interfaces are also provided so researchers can smoothly implement `ezc3d` in their current workflow.
It supports *c3d* files from the main biomechanics software, namely: Vicon, Qualisys, Optotrak, BTS and XSens. 
In addition, since the *c3d* standard allows for multiple ways to store force platform data, a force platform analysis module is provided.
The main feature of this module is to express forces and moments in more common reference frames---that is, expressed in the global reference frame calculated at either the origin or at the centre of pressure---so they can be directly interpreted by the user. 

# Statement of need
The *c3d* binary format [@crampC3dOrg2019] is an open source standard extensively used in the field of biomechanics.
Most of the software for biomechanical data collection and data analysis can read and export *c3d* files. 
Initially, this format was designed to store three-dimensional point and analog data (e.g., force platform or EMG).
Nowadays, by stretching the standard, companies have managed to integrate new types of data, including for instance IMU data.
To match the needs of the community, Motion Lab Systems---which created and maintains the *c3d* format---regularly updates the standard to match the biomechanical needs and to include new usages of the format.
 
Despite being extensively used by the biomechanics community, there are surprisingly few alternatives when it comes to manipulate *c3d* files outside analyses software. 
This has forced scientists to develop *ad hoc* solutions for each project which requires writing file I/O codes for each software file format. 
While it would make sense to develop a portable solution once and for all, the binary nature of the *c3d* format discourages biomechanists from digging into the trouble of developing such a solution.
To our knowledge, `BTK` [@barreBiomechanicalToolKitBTKCore2020] is the most mature (if not, the only) biomechanics library that provides an API to read and write *c3d* files.
Unfortunately, despite its open sourced nature, the project has been mostly abandoned since~2016.
It gets more and more out-of-date as it does not implement the changes in the standard of the format.
Unfortunately, due to the intricate connections of its modules, it proved difficult to update `BTK`.
The `ezc3d` toolbox is an up-to-date solution that will fill all your *c3d* management needs. 
 
# The dependencies
The `ezc3d` library was designed to be a dependency-free library such that the lay users could easily link `ezc3d` with their project.
Thus, by default, no dependency is needed to compile nor to use the API.
Moreover, thanks to the absence of external library requirements that could change at any time, `ezc3d` will remain available on all major operating systems, namely Windows, Linux and Mac. 
 
By nature, biomechanics data are matrix-based data. 
A linear algebra solution was therefore developed to store and manipulate such data.
Our solution will, however, never be as effective as those from dedicated linear algebra libraries.
Hence, thanks to the highly effective `Eigen` linear algebra library [@eigenweb], a fast accessor module is also available.
 
# Current usage of `ezc3d`
The library got the attention of two major modeling frameworks in biomechanics, namely the Anybody Modeling System [@rasmussenChapterAnyBodyModeling2019] and Opensim [@sethOpenSimSimulatingMusculoskeletal2018].
One of the employees of the former prepared and maintains the `conda-forge` recipe for `ezc3d`, so it can be easily installed and updated using `Conda`.
Since version 4.0, Opensim embraced the *c3d* format file by providing the capability to read *c3d* files.
The `ezc3d` library was chosen as the default *c3d* reader back end.
 
# Acknowledgment
The authors would like to thank François Bailly for its help in the writing of the paper.
Also, many thanks to Sahel Locher who designed the lovely `ezc3d` logo! 
 
 
# References
