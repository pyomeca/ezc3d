#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Redcat"
__version__ = ".001"
__date__ = "3_14_2019"

import os
from ezc3d import c3d
import tkinter
from tkinter import messagebox


class C3DData:
    """
    Parameters:
            FullPath:       String of the file path
    Return:
            GetC3DData:     c3d Object
    """

    def __init__(self, parent, fullPath):

        self.RetrieveC3dData(fullPath)

    def RetrieveC3dData(self, fullPath):

        # Create some dictionaries to store data
        self.Gen = {
            "PathName": [],
            "FileName": [],
            "SubjName": [],
            "SubjMass": [],
            "SubjHeight": [],
            "ModelUsed": [],
            "NumbItems": [],
            "Vid_FirstFrame": [],
            "Vid_LastFrame": [],
            "Vid_SampRate": [],
            "Analog_FirstFrame": [],
            "Analog_LastFrame": [],
            "Analog_SampRate": [],
            "Analog_NumbChan": [],
            "AnalogUnits": [],
            "PointsUnit": [],
            "AnglesUnit": [],
            "ForcesUnit": [],
            "MomentsUnit": [],
            "PowersUnit": [],
            "ScalarsUnit": [],
            "SubjLLegLength": [],
            "SubjRLegLength": [],
            "ForcePlateOrigin": [],
        }

        self.Labels = {
            "PointsName": [],
            "Markers": [],
            "Angles": [],
            "Scalars": [],
            "Powers": [],
            "Forces": [],
            "Moments": [],
            "AnalogsName": [],
            "EventLabels": [],
            "AnalysisNames": [],
        }

        self.Data = {
            "AllPoints": [],
            "Markers": [],
            "Angles": [],
            "Scalars": [],
            "Powers": [],
            "Forces": [],
            "Moments": [],
            "Analogs": [],
        }

        # returns a handle to RetrieveC3dData
        self.c3d = c3d(os.path.join(fullPath))

        self.GetHeader(fullPath)
        self.GetSubjects()
        self.GetPoint()
        self.GetAnalog()
        self.GetForcePlatForm()
        self.GetEventContext()
        self.GetAnalysis()
        self.GetProcessing()

    def GetHeader(self, fullPath):
        if "header" in self.c3d:
            self.Gen["PathName"] = fullPath  # get the fullPath to the file
            self.Gen["FileName"] = os.path.basename(fullPath)  # get only the file name
            if "points" in self.c3d["header"]:  # if points exist
                self.Gen["NumbItems"] = self.c3d["header"]["points"][
                    "size"
                ]  # get number items (markers, angles, moments, etc.)
                self.Gen["Vid_FirstFrame"] = self.c3d["header"]["points"]["first_frame"]  # get the cameras first frame
                self.Gen["Vid_LastFrame"] = self.c3d["header"]["points"]["last_frame"]  # get the cameras last frame
                self.Gen["Vid_SampRate"] = self.c3d["header"]["points"]["frame_rate"]  # get the cameras sample rate
            if "analogs" in self.c3d["header"]:  # if analogs exist
                self.Gen["Analog_FirstFrame"] = self.c3d["header"]["analogs"][
                    "first_frame"
                ]  # get the analogs first frame
                self.Gen["Analog_LastFrame"] = self.c3d["header"]["analogs"]["last_frame"]  # get the analogs last frame
                self.Gen["Analog_SampRate"] = self.c3d["header"]["analogs"]["frame_rate"]  # get the analogs sample rate
                self.Gen["Analog_NumbChan"] = self.c3d["header"]["analogs"][
                    "size"
                ]  # get the analogs number of channels

    def GetSubjects(self):
        if "SUBJECTS" in self.c3d["parameters"]:
            if self.c3d["parameters"]["SUBJECTS"]["USED"]["value"][0] > 0:  # if exist data
                self.Gen["ModelUsed"] = self.c3d["parameters"]["SUBJECTS"]["MARKER_SETS"]["value"]  # get the model used
                self.Gen["SubjName"] = self.c3d["parameters"]["SUBJECTS"]["NAMES"][
                    "value"
                ]  # get the name of the subject

    def GetPoint(self):
        if "POINT" in self.c3d["parameters"]:
            if self.c3d["parameters"]["POINT"]["USED"]["value"][0] > 0:  # if exist data
                self.Labels["PointsName"] = self.c3d["parameters"]["POINT"]["LABELS"][
                    "value"
                ]  # get all the labels (markers, angles, etc.)
                self.Gen["PointsUnit"] = self.c3d["parameters"]["POINT"]["UNITS"]["value"][
                    0
                ]  # get the units of the coodinates
                self.Data["AllPoints"] = self.GetPointData("PointsName")  # get data of all items (markers, angles,etc.)

                ExtList = list()  # temp list
                if "TYPE_GROUPS" in self.c3d["parameters"]["POINT"]:  # determine if groups exist (model outputs)
                    Groups = self.c3d["parameters"]["POINT"]["TYPE_GROUPS"]["value"]  # groups available in C3D
                    PGroups = ["ANGLES", "FORCES", "MOMENTS", "POWERS", "SCALARS"]  # possible groups available in C3D
                    GroupsInC3D = sorted(list(set(PGroups) & set(Groups)))  # determine the real groups in C3D

                    for k in range(0, len(GroupsInC3D)):

                        self.Labels[GroupsInC3D[k].title()] = self.GetLabels(GroupsInC3D[k])  # store group labels

                        # get the points for each item in TYPE_GROUPS available
                        MyList = self.GetLabels(GroupsInC3D[k])  # temp list
                        self.Data[GroupsInC3D[k].title()] = self.GetData_Groups(
                            MyList
                        )  # store the data for each group available in c3d

                        ExtList.extend(MyList)  # create a temp list of all labels in groups
                        self.GetUnits(GroupsInC3D[k])  # get the units of each group

                self.Labels["Markers"] = sorted(
                    list(set(self.Labels["PointsName"]) ^ set(ExtList))
                )  # add only the labels for markers
                self.Data["Markers"] = self.GetData_Groups(self.Labels["Markers"])  # add vectors XYZ of markers
            else:
                # hide tk main window
                root = tkinter.Tk()
                root.withdraw()
                messagebox.showwarning("Warning", "No coordinates available in C3D File", parent=root)

    def GetAnalog(self):
        if "ANALOG" in self.c3d["parameters"]:
            if self.c3d["parameters"]["ANALOG"]["USED"]["value"][0] > 0:  # if data exist
                self.Labels["AnalogsName"] = self.c3d["parameters"]["ANALOG"]["LABELS"]["value"]  # get channel names
                self.Gen["AnalogUnits"] = self.c3d["parameters"]["ANALOG"]["UNITS"]["value"]  # get analog units
                self.Data["Analogs"] = self.GetAnalogData("AnalogsName")

    def GetForcePlatForm(self):
        if "FORCE_PLATFORM" in self.c3d["parameters"]:
            if self.c3d["parameters"]["FORCE_PLATFORM"]["USED"]["value"][0] > 0:  # if exist data
                self.Gen["ForcePlateOrigin"] = self.c3d["parameters"]["FORCE_PLATFORM"]["ORIGIN"][
                    "value"
                ]  # get the origin of force plate

    def GetEventContext(self):
        if "EVENT_CONTEXT" in self.c3d["parameters"]:
            if self.c3d["parameters"]["EVENT_CONTEXT"]["USED"]["value"][0] > 0:  # if exist data
                self.Labels["EventLabels"] = self.c3d["parameters"]["EVENT_CONTEXT"]["LABELS"][
                    "value"
                ]  # get the origin of force plate

    def GetAnalysis(self):
        if "ANALYSIS" in self.c3d["parameters"]:
            if self.c3d["parameters"]["ANALYSIS"]["USED"]["value"][0] > 0:  # if exist data
                self.Labels["AnalysisNames"] = self.c3d["parameters"]["ANALYSIS"]["NAMES"][
                    "value"
                ]  # get the origin of force plate

    def GetProcessing(self):
        if "PROCESSING" in self.c3d["parameters"]:
            self.Gen["SubjHeight"] = self.c3d["parameters"]["PROCESSING"]["Height"][
                "value"
            ]  # get the height of the subject
            self.Gen["SubjMass"] = self.c3d["parameters"]["PROCESSING"]["Bodymass"][
                "value"
            ]  # get the mass of the subject
            self.Gen["SubjLLegLength"] = self.c3d["parameters"]["PROCESSING"]["LLegLength"][
                "value"
            ]  # get the LLegLenght of the subject
            self.Gen["SubjRLegLength"] = self.c3d["parameters"]["PROCESSING"]["RLegLength"][
                "value"
            ]  # get the RLegLenght of the subject

    def GetAnalogData(self, Name):
        data = self.c3d["data"]["analogs"]
        dicts = {}
        for k in range(0, data.shape[1]):
            ListStr = self.Labels[Name][k]  # iterate over list and get keys
            dicts[ListStr] = data[0][k]
        return dicts

    def GetPointData(self, Name):
        data = self.c3d["data"]["points"]
        dicts = {}
        for k in range(0, data.shape[1]):
            ListStr = self.Labels[Name][k]
            dicts[ListStr] = data[0:4, k]
        return dicts

    def GetLabels(self, Names):
        if Names in self.c3d["parameters"]["POINT"]:
            return self.c3d["parameters"]["POINT"][Names]["value"]

    def GetData_Groups(self, Item):
        dicts = {}
        for key in Item:
            dicts[key] = self.Data["AllPoints"][key]
        return dicts

    def GetUnits(self, GroupsInC3D):
        GroupsUnit = ["AnglesUnit", "ForcesUnit", "MomentsUnit", "PowersUnit", "ScalarsUnit"]  # temp variable
        Units = ["ANGLE_UNITS", "FORCE_UNITS", "MOMENT_UNITS", "POWER_UNITS", "SCALAR_UNITS"]  # temp variable
        for kk in range(0, len(GroupsUnit)):
            if GroupsUnit[kk].lower().find(GroupsInC3D.lower(), 0, len(GroupsInC3D)) != -1:
                self.Gen[GroupsUnit[kk]] = self.c3d["parameters"]["POINT"][Units[kk]]["value"][
                    0
                ]  # add the group units to the header
                break


if __name__ == "__main__":

    C3DData(None, path_to_c3d)
