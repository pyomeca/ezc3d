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
The *c3d* format [@C3DORGBiomechanics] is an opensource standard extensively used in the field of biomechanics.
Indeed, main data collection and data analyses software can export/read them natively. 
It was initially designed to hold three-dimensional point and analog (such as forceplate and EMG) data.
Nowadays, by stretching the standard, companies have managed to include all sorts of theoretically non-c3d-
compliant biomechanical data, including for instance IMU data.
To match the needs of the community, Motion Lab Systems---who created and maintains the *c3d* format---updates the standard when needed.

Despite being extensively used by the biomechanics community, there are surprisingly few alternatives when it comes to manipulate *c3d* files outside analyses software. 
To our knowledge, *BTK* [@barreBiomechanicalToolKitBTKCore2020] is the most mature (if not, the only) biomechanics library that provides an API to read and write these files.
Unfortunately, despite its open-source nature, the project has been mostly abandoned since~2016.
Hence, *BTK* has not been following the changes in the standard of the format nor that it will follow those to come.
Unfortunately, due to the numerous modules that have been developed over the years which are tightly connected, updating *BTK* without breaking it is a hard task.

Introducing the open-source *ezc3d* library which provides a comprehensive API to easily read and write *c3d* files. 
For the lay users, *ezc3d* is therefore an up-to-date solution to read and write *c3d* files that complies to latest standard. 
Moreover, *ezc3d* can read in-house implementations of main biomechanics software, that is currently: Vicon, Qualisys, Optotrak, BTS and XSens. 
The core of *ezc3d* is written in C++ allowing for fast file I/O.
MATLAB and Python3 interfaces are provided so the biomechanics community can implement *ezc3d* in their current workflow without difficulties.
In addition, since the *c3d* standard allows for multiple storage of force platform data, a force platform analysis module is provided.
The main feature of this module is to reorient forces, moment and centre of pressure in more common reference frames so they can be directly interpreted by the user. 

# The dependencies
The *ezc3d* library was originally designed to be a dependencies-free library.
This allows to easily link *ezc3d* with the user projects. 
Thus, by default, no dependendies is needed to compile and to use the API.
That said, if one requires a faster file reading solution, a fast accessor option is avaible that relies on *eigen* linear algebra library [@eigenweb].

# Current usage of `ezc3d`
OpenSim---one of the most important software in biomechanics---has recently decided to add the capability to read *c3d* files and decided to go with the *ezc3d* library for their default backend.

# References
