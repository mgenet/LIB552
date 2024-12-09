#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2020-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import glob
import ipywidgets
import itkwidgets
import xml.etree.ElementTree as ET

import vtk
import myVTKPythonLibrary as myvtk

import LIB552 as lib

################################################################################


class DolfinPVDReader:
    def __init__(self, pvd_folder, pvd_filename):
        self.pvd_folder   = pvd_folder
        self.pvd_filename = pvd_filename

        tree = ET.parse(self.pvd_folder+"/"+self.pvd_filename)
        root = tree.getroot()
        collections = root[0]
        self._timesteps = [float(child.attrib["timestep"]) for child in collections]
        self._files = [child.attrib["file"] for child in collections]
        
    def get_ugrid(self, index):
        return myvtk.readUGrid(self.pvd_folder+"/"+self._files[index])


class UGridSeriesReader:
    def __init__(self, ugrids_folder, ugrids_filebasename, ugrids_extension="vtu"):
        self.ugrids_folder       = ugrids_folder
        self.ugrids_filebasename = ugrids_filebasename
        self.ugrids_extension    = ugrids_extension

        self._files = sorted(glob.glob(self.ugrids_folder+"/"+self.ugrids_filebasename+"_[0-9]*."+self.ugrids_extension))
        assert (len(self._files) > 0), "There must be at least one mesh. Aborting."
        self._timesteps = range(len(self._files))

    def get_ugrid(self, index):
        return myvtk.readUGrid(self._files[index])


class DisplacementViewer:
    def __init__(self, pvd_folder=".", pvd_filename=None, ugrids_folder=".", ugrids_filebasename=None, state_label="time"):
        self._state_label = state_label

        if (pvd_filename is not None):
            self._reader = lib.DolfinPVDReader(pvd_folder, pvd_filename)
        elif (ugrids_filebasename is not None):
            self._reader = lib.UGridSeriesReader(ugrids_folder, ugrids_filebasename)
        else:
            assert (0), "Need to provide \"pvd_filename\" or \"ugrids_filebasename\". Aborting."

        self._U_warp = vtk.vtkWarpVector()
        self._U_warp.SetInputData(self._reader.get_ugrid(0))
        self._U_warp.SetScaleFactor(1.)
        self._U_warp.Update()
        self._viewer = itkwidgets.view(geometries=[self._U_warp.GetOutput()])

    def _warp(self, index=0, factor=1.):
        self._U_warp.SetInputData(self._reader.get_ugrid(index))
        self._U_warp.SetScaleFactor(factor)
        self._U_warp.Update()
        self._viewer.geometries = [self._U_warp.GetOutput()]

    def show(self):
        slider = ipywidgets.interactive(
            self._warp, 
            index=(0, len(self._reader._timesteps)-1, 1), 
            factor=(0, 10, 0.1), continuous_update=True)
        i_slider, f_slider = slider.children[:2]
        
        state_label = ipywidgets.Label(
            value=self._state_label+": {:e}".format(self._reader._timesteps[i_slider.value]))

        def _update_label(change):
            state_label.value = self._state_label+": {:e}".format(self._reader._timesteps[int(change["new"])])
        
        i_slider.observe(_update_label, names=["value"])
        
        def _next(b):
            i_slider.value += 1

        def _prev(b):
            i_slider.value -= 1

        b_prev = ipywidgets.Button(description="<")
        b_prev.on_click(_prev)
        b_next = ipywidgets.Button(description=">")
        b_next.on_click(_next)

        grid = ipywidgets.GridspecLayout(2, 4)
        grid[0, 0] = b_prev
        grid[0, 1] = i_slider
        grid[0, 2] = b_next
        grid[0, 3] = state_label
        grid[1, 1] = f_slider

        b_prev.layout.width =  "50px"
        b_next.layout.width =  "50px"
        grid.layout.width   = "750px"
        grid.layout.padding =   "0px"
        
        i_slider.description = self._state_label[:4]+" index"
        f_slider.description = "disp factor"

        return ipywidgets.VBox([self._viewer, grid])
