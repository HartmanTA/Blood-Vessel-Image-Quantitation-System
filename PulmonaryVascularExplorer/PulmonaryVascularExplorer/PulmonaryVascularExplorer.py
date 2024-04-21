import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import numpy as np
import ExtractCenterline


#
# PulmonaryVascularExplorer
#


class PulmonaryVascularExplorer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PulmonaryVascularExplorer")  
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Pulmonary Vascular Explorer")]
        self.parent.dependencies = []  #TODO figure out how to make it a requirement that VMTK is a downloaded extension before code will run
        self.parent.contributors = ["Hugo Bachaumard, Tyler Hartman, Aaron McCelllan (College of Charleston)"] 
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/HartmanTA/Blood-Vessel-Image-Quantitation-System">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

#TODO finish 
def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # lungVesselSeg1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="lungVesselSeg",
        sampleName="lungVesselSeg1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "lungVesselSeg1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="lungVesselSeg1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="lungVesselSeg1",
    )

    # lungVesselSeg2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="lungVesselSeg",
        sampleName="lungVesselSeg2",
        thumbnailFileName=os.path.join(iconsPath, "lungVesselSeg2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="lungVesselSeg2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="lungVesselSeg2",
    )


#
# PulmonaryVascularExplorerParameterNode
#


@parameterNodeWrapper
class PulmonaryVascularExplorerParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to be used
    """

    inputVolume: vtkMRMLScalarVolumeNode
    subdivideInputSurface : bool = False
    targetNumberOfPoints : Annotated[float, WithinRange(500, 5000)] = 5000
    decimationAggressiveness : Annotated[float, WithinRange(2, 8)] = 4

#
# PulmonaryVascularExplorerWidget
#


class PulmonaryVascularExplorerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PulmonaryVascularExplorer.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PulmonaryVascularExplorerLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.batchProcessPushButton.connect("clicked(bool)", self.onBatchProcessButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[PulmonaryVascularExplorerParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume:
            self.ui.applyButton.toolTip = _("Find Vessel Segmentation")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            #print(self.ui.DirectoryPath.currentPath)
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.decimationSlider.value, 
                               self.ui.numOfPointsSpinBox.value, self.ui.subDivideInputSurfaceCheckBox.checked, self.ui.DirectoryPath.currentPath)

    def onBatchProcessButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Put all volumes within a list
            listOfSceneVolumes = list(slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode'))

            # Compute output on all volumes loaded in scene
            self.logic.processAll(listOfSceneVolumes, self.ui.decimationSlider.value, 
                               self.ui.numOfPointsSpinBox.value, self.ui.subDivideInputSurfaceCheckBox.checked, self.ui.DirectoryPath.currentPath)
    


#
# PulmonaryVascularExplorerLogic
#


class PulmonaryVascularExplorerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
    
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        
        


    def getParameterNode(self):
        return PulmonaryVascularExplorerParameterNode(super().getParameterNode())

    
    def vesselFinder(self, inputVolumeAsArray):
        
        nonZeroes = np.nonzero(inputVolumeAsArray)
        visited = np.zeros_like(inputVolumeAsArray, dtype=int)
        nzLen = len(nonZeroes[0])
        v = 1
        for t in range(nzLen-1):
            if not visited[nonZeroes[0][t]][nonZeroes[1][t]][nonZeroes[2][t]]:
                currentVal = inputVolumeAsArray[nonZeroes[0][t]][nonZeroes[1][t]][nonZeroes[2][t]]
                currentPoint = [(nonZeroes[0][t], nonZeroes[1][t], nonZeroes[2][t])]
                while currentPoint:
                    depth, row, col = currentPoint.pop()
                    if (0 <= depth < len(inputVolumeAsArray) and
                        0 <= row < len(inputVolumeAsArray[0]) and 
                        0 <= col < len(inputVolumeAsArray[0][0]) and 
                        not (visited[depth][row][col] > 0) and
                        (inputVolumeAsArray[depth][row][col] == currentVal)):
                        visited[depth][row][col] = v
                        currentPoint.extend([(depth + dd, row + dr, col + dc) for dd, dr, dc in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]])
                v += 1
        
        return visited
    
    def batchSaE(self, labelMap, centroid):
        saeSet = list()
        vSet = np.unique(labelMap)
        for val in vSet:
            indeces = np.rot90(np.where(labelMap == val))
            saeSet.append(self.startAndEnd(indeces, centroid))
        return saeSet

    def centroidFinder(self, a):
        non_zero_indices = np.nonzero(a)
        iTot = np.sum(non_zero_indices[0])
        jTot = np.sum(non_zero_indices[1])
        kTot = np.sum(non_zero_indices[2])
        num_non_zero = len(non_zero_indices[0])
        if num_non_zero > 0:
            iAvg = iTot / num_non_zero
            jAvg = jTot / num_non_zero
            kAvg = kTot / num_non_zero
        else:
            iAvg, jAvg, kAvg = 0, 0, 0
        return [iAvg, jAvg, kAvg]

    def startAndEnd(self, vessel, centroid):
        distances = np.linalg.norm(vessel - centroid, axis=1)
        startPoint = np.argmin(distances)
        endPoint = np.argmax(distances)
        return [vessel[startPoint], vessel[endPoint]]


    def findAllCenterlines(self, labelMap, subdivideInputSurface, centroid, decimationAggressiveness, targetNumberOfPoints):
        import ExtractCenterline
        extractLogic = ExtractCenterline.ExtractCenterlineLogic()
        
        segVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(segVol, labelMap)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segVol, segmentationNode)
        a = list(segmentationNode.GetSegmentation().GetSegmentIDs())
        startsAndEnds = []
        startsAndEnds = self.batchSaE(labelMap, centroid)
        endPointsMarkupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Centerline endpoints")
        i = 0
        while (i < len(a)):
        #  for seg in a:
            try:
                seg = a[i]
                print(seg)
        # Preprocess the surface
                inputSurfacePolyData = extractLogic.polyDataFromNode(segmentationNode, seg)
                thisSandE = [startsAndEnds[int(seg.lstrip("Label_")) - 1][0], startsAndEnds[int(seg.lstrip("Label_")) - 1][1]]
                preprocessedPolyData = extractLogic.preprocess(inputSurfacePolyData, targetNumberOfPoints, decimationAggressiveness, subdivideInputSurface)
        # Auto-detect the endpoints
        # TODO: Add Startpoint/inlet specification 
                endPointsMarkupsNode.AddControlPoint(thisSandE[0])
                endPointsMarkupsNode.AddControlPoint(thisSandE[1])
                endPointsMarkupsNode.SetNthControlPointSelected (0, 1 == 1)
                networkPolyData = extractLogic.extractNetwork(preprocessedPolyData, endPointsMarkupsNode)
                startPointPosition = None
                endpointPositions = extractLogic.getEndPoints(networkPolyData, startPointPosition)
                endPointsMarkupsNode.RemoveAllControlPoints()
                for position in endpointPositions:
                    endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))
            
             # Extract the centerline
                centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Centerline curve")
                centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(preprocessedPolyData, endPointsMarkupsNode)
                centerlinePropertiesTableNode = None
                extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode)
                i += 1
            except:
                print("I made a mistake")
                i += 1
                continue


    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                decimationAggressiveness: float,
                targetNumberOfPoints: float,
                subdivideInputSurface: bool = False,
                path: str = "") -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be used 
        """
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")
        
        print(path)
        #perform processing
        inputVolumeAsArray = slicer.util.arrayFromVolume(inputVolume)
        self.findAllCenterlines(self.vesselFinder(inputVolumeAsArray), subdivideInputSurface, self.centroidFinder(inputVolumeAsArray), decimationAggressiveness, targetNumberOfPoints)
    
        

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

    def processAll(self,
                allInputVolumes: vtkMRMLScalarVolumeNode,
                decimationAggressiveness: float,
                targetNumberOfPoints: float,
                subdivideInputSurface: bool = False,
                path: str = "") -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param allInputVolumes: All volumes loaded in scene 
        """
        #if not inputVolume:
      #      raise ValueError("Input volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        for x in allInputVolumes:
            inputVolumeAsArray = slicer.util.arrayFromVolume(x)
            self.findAllCenterlines(self.vesselFinder(inputVolumeAsArray), subdivideInputSurface, self.centroidFinder(inputVolumeAsArray), decimationAggressiveness, targetNumberOfPoints)

        #perform processing
        
        
    
        

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")