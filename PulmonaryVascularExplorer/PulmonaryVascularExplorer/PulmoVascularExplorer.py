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
# PulmoVascularExplorer
#


class PulmoVascularExplorer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("PulmoVascularExplorer")  
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "PulmonaryVascularExplorer")]#TODO figure out way to give the extension its own section in the 3d slicer dropdown
        self.parent.dependencies = []  #TODO figure out how to make it a requirement that VMTK is a downloaded extension before code will run
        self.parent.contributors = ["Hugo Bachaumard, Tyler Hartman, Aaron McCelllan (College of Charleston)"] 
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This module is intended to aid in centerline
extraction for blood vessels in the lungs
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This module was developed by students at the College of Charleston Aaron McClellan, Hugo Bachaumard, and Tyler Hartman under the advisement of Dr. Brummer and uses the Vascular Modeling Toolkit in order to extract the centerline of each vessel tree
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

    # PulmoVascularExplorer1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PulmoVascularExplorer",
        sampleName="PulmoVascularExplorer1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "PulmoVascularExplorer1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="PulmoVascularExplorer1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="PulmoVascularExplorer1",
    )

    # PulmoVascularExplorer2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="PulmoVascularExplorer",
        sampleName="PulmoVascularExplorer2",
        thumbnailFileName=os.path.join(iconsPath, "PulmoVascularExplorer2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="PulmoVascularExplorer2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="PulmoVascularExplorer2",
    )


#
# PulmoVascularExplorerParameterNode
#


@parameterNodeWrapper
class PulmoVascularExplorerParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to be used
    """

    inputVolume: vtkMRMLScalarVolumeNode
    subdivideInputSurface : bool = False
    targetNumberOfPoints : Annotated[float, WithinRange(500, 5000)] = 5000
    decimationAggressiveness : Annotated[float, WithinRange(2, 8)] = 4

#
# PulmoVascularExplorerWidget
#


class PulmoVascularExplorerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/PulmoVascularExplorer.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = PulmoVascularExplorerLogic()

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

    def setParameterNode(self, inputParameterNode: Optional[PulmoVascularExplorerParameterNode]) -> None:
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
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.decimationSlider.value, 
                               self.ui.numOfPointsSpinBox.value, self.ui.subDivideInputSurfaceCheckBox.checked,
                               self.ui.makeTables.checked, self.ui.createModels.checked, self.ui.MinVesselSize.value)

    def onBatchProcessButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Put all volumes within a list
            listOfSceneVolumes = list(slicer.mrmlScene.GetNodesByClass('vtkMRMLScalarVolumeNode'))

            # Compute output on all volumes loaded in scene
            self.logic.processAll(listOfSceneVolumes, self.ui.decimationSlider.value, 
                               self.ui.numOfPointsSpinBox.value, self.ui.subDivideInputSurfaceCheckBox.checked,
                               self.ui.makeTables.checked, self.ui.createModels.checked, self.ui.MinVesselSize.value)
    


#
# PulmoVascularExplorerLogic
#


class PulmoVascularExplorerLogic(ScriptedLoadableModuleLogic):
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
        return PulmoVascularExplorerParameterNode(super().getParameterNode())

    
    def calculate_horton_strahler_number(self, vtkTable, rowCount, childrenColumnIndex):
        """Calculate Horton-Strahler number for each node."""
        hsnColumn = vtk.vtkIntArray()
        hsnColumn.SetName("HortonStrahlerNumber")
        hsnColumn.SetNumberOfTuples(rowCount)
        vtkTable.AddColumn(hsnColumn)
        
        cellIdToIndex = {vtkTable.GetValue(rowIndex, 0).ToString(): rowIndex for rowIndex in range(rowCount)}
        
        def compute_hsn(selfcellId):
            rowIndex = cellIdToIndex[cellId]
            childrenString = vtkTable.GetValue(rowIndex, childrenColumnIndex).ToString()
            
            if childrenString == "None":
                return 1  # Leaf node
            
            childrenIds = childrenString.split(" and ")
            childHSNs = [compute_hsn(childId) for childId in childrenIds]
            
            if not childHSNs:
                return 1
            
            max_hsn = max(childHSNs)
            if childHSNs.count(max_hsn) > 1:
                return max_hsn + 1
            return max_hsn
    
        for rowIndex in range(rowCount):
            cellId = vtkTable.GetValue(rowIndex, 0).ToString()
            hsn = compute_hsn(cellId)
            hsnColumn.SetValue(rowIndex, hsn)

    def updateTableWithHortonStrahlerNumbers(self, tableNode):
        """Add or update Horton-Strahler numbers in the given table."""
        vtkTable = tableNode.GetTable()
        rowCount = vtkTable.GetNumberOfRows()
        childrenColumnIndex = vtkTable.GetColumnIndex("Children")
        calculate_horton_strahler_number(vtkTable, rowCount, childrenColumnIndex)
        tableNode.Modified()  # Notify the system that the table has been updated

    def updateTableWithCoordinates(self, tableNode):
        """Populate the table with spatial and descriptive information about centerlines."""
        vtkTable = tableNode.GetTable()
        rowCount = vtkTable.GetNumberOfRows()
        labelNumber = int(tableNode.GetName().split('_')[-1])

        # Initialize and add columns
        columns = {
            "CellId": vtk.vtkStringArray(),
            "TreeId": vtk.vtkIntArray(),
            "Diameter": vtk.vtkDoubleArray(),
            "start_x": vtk.vtkDoubleArray(),
            "start_y": vtk.vtkDoubleArray(),
            "start_z": vtk.vtkDoubleArray(),
            "end_x": vtk.vtkDoubleArray(),
            "end_y": vtk.vtkDoubleArray(),
            "end_z": vtk.vtkDoubleArray(),
            "Parent": vtk.vtkStringArray(),
            "Children": vtk.vtkStringArray(),
            "ParentCellId": vtk.vtkStringArray()
        }
        for key, arr in columns.items():
            arr.SetName(key)
            arr.SetNumberOfTuples(rowCount)
            vtkTable.AddColumn(arr)

        # Radius index and positions dictionary for parent-child mapping
        radiusIndex = vtkTable.GetColumnIndex("Radius")
        positions = {}

        for rowIndex in range(rowCount):
            cellId = f"{labelNumber}.{rowIndex}"
            columns["CellId"].SetValue(rowIndex, cellId)
            columns["TreeId"].SetValue(rowIndex, labelNumber)
            columns["Diameter"].SetValue(rowIndex, 2 * vtkTable.GetValue(rowIndex, radiusIndex).ToDouble())
            columns["ParentCellId"].SetValue(rowIndex, "None")  # Default value

            # Handle coordinates
            startPoint = vtkTable.GetValueByName(rowIndex, 'StartPointPosition').ToString().split()
            endPoint = vtkTable.GetValueByName(rowIndex, 'EndPointPosition').ToString().split()
            for i, coord in enumerate(["x", "y", "z"]):
                columns[f'start_{coord}'].SetValue(rowIndex, float(startPoint[i]))
                columns[f'end_{coord}'].SetValue(rowIndex, float(endPoint[i]))

            positions[rowIndex] = {
                'start': (float(startPoint[0]), float(startPoint[1]), float(startPoint[2])),
                'end': (float(endPoint[0]), float(endPoint[1]), float(endPoint[2])),
                'cellId': cellId
            }

        # Map endpoints to start points for parent-child relationships
        startPointMap = {pos['start']: [] for pos in positions.values()}
        for pos in positions.values():
            startPointMap[pos['start']].append(pos['cellId'])

        for idx, pos in positions.items():
            children = startPointMap.get(pos['end'], [])
            children = [child for child in children if child != pos['cellId']]
            columns["Parent"].SetValue(idx, "Yes" if children else "No")
            childrenString = " and ".join(children) if children else "None"
            columns["Children"].SetValue(idx, childrenString)
            for child in children:
                childIndex = next((i for i in range(rowCount) if columns["CellId"].GetValue(i) == child), None)
                if childIndex is not None:
                    columns["ParentCellId"].SetValue(childIndex, pos['cellId'])

        tableNode.Modified()  # Notify system of updates

    def check_asymmetry_by_max_features(self, tableNode):
        """Check for asymmetry based on maximum features of children."""
        vtkTable = tableNode.GetTable()
        rowCount = vtkTable.GetNumberOfRows()

        # Indices for columns
        nodeIdIndex = vtkTable.GetColumnIndex("CellId")
        parentColumnIndex = vtkTable.GetColumnIndex("Parent")
        radiusIndex = vtkTable.GetColumnIndex("Radius")
        lengthIndex = vtkTable.GetColumnIndex("Length")

        # New column for asymmetry type, using strings to represent True, False, and NA
        asymmetryColumn = vtk.vtkStringArray()
        asymmetryColumn.SetName("Asymmetry")
        asymmetryColumn.SetNumberOfTuples(rowCount)
        vtkTable.AddColumn(asymmetryColumn)

        # Initialize all rows with "NA" first
        for i in range(rowCount):
            asymmetryColumn.SetValue(i, "NA")

        # Find bifurcating parents and check their children
        for rowIndex in range(rowCount):
            if vtkTable.GetValueByName(rowIndex, "Parent").ToString() == "Yes":
                childrenString = vtkTable.GetValueByName(rowIndex, "Children").ToString()
                childrenIds = childrenString.split(" and ") if childrenString != "None" else []

                if len(childrenIds) == 2:
                    childRadii = []
                    childLengths = []

                    for childId in childrenIds:
                        childIndex = next((i for i in range(rowCount) if vtkTable.GetValue(i, nodeIdIndex).ToString() == childId), None)
                        if childIndex is not None:
                            childRadii.append(vtkTable.GetValue(childIndex, radiusIndex).ToDouble())
                            childLengths.append(vtkTable.GetValue(childIndex, lengthIndex).ToDouble())

                    # Check the asymmetry condition
                    if len(set(childRadii)) == 2 and len(set(childLengths)) == 2:
                        max_length_index = childLengths.index(max(childLengths))
                        max_radius_index = childRadii.index(max(childRadii))

                        if max_length_index == max_radius_index:
                            asymmetryColumn.SetValue(rowIndex, "True")
                        else:
                            asymmetryColumn.SetValue(rowIndex, "False")

    def consolidateTablesIntoMaster(self):
        """Consolidates all centerline tables into one master table."""
        masterTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "MasterCenterlineTable")
        masterTable = masterTableNode.GetTable()

        # Retrieve all centerline tables
        tableNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLTableNode")
        tableNodes.InitTraversal()
        tableNode = tableNodes.GetNextItemAsObject()
        
        firstTable = True
        while tableNode:
            if "Centerline_Table_Label_" in tableNode.GetName():
                vtkTable = tableNode.GetTable()
                if firstTable:
                    # Copy the structure of the first table
                    masterTable.DeepCopy(vtkTable)
                    firstTable = False
                else:
                    # Append rows from subsequent tables
                    numRows = vtkTable.GetNumberOfRows()
                    for i in range(numRows):
                        newRow = masterTable.InsertNextBlankRow()
                        for j in range(vtkTable.GetNumberOfColumns()):
                            masterTable.SetValue(newRow, j, vtkTable.GetValue(i, j))
            tableNode = tableNodes.GetNextItemAsObject()

        masterTableNode.Modified()  # Notify the system that the master table has been updated
        print("All centerline tables consolidated into the master table.")

    def processAllCenterlineTables(self):
        """Process all centerline tables in the scene to update them with new metrics and information."""
        tableNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLTableNode")
        tableNodes.InitTraversal()
        tableNode = tableNodes.GetNextItemAsObject()

        while tableNode:
            if "Centerline_Table_Label_" in tableNode.GetName():
                updateTableWithCoordinates(tableNode)
                check_asymmetry_by_max_features(tableNode)
                updateTableWithHortonStrahlerNumbers(tableNode)
                tableNode.Modified()  # Notify the system that the table has been updated
            tableNode = tableNodes.GetNextItemAsObject()
        consolidateTablesIntoMaster()

    
    
    
    
    def vesselFinder(self, inputVolumeAsArray, minVesselSize):
        import numpy as np
        nonZeroes = np.nonzero(inputVolumeAsArray)
        visited = np.zeros_like(inputVolumeAsArray, dtype=float)
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
                cou = np.count_nonzero(visited == v)
                if (cou < minVesselSize):
                    visited[visited == t] = 0
                else:
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


    def findAllCenterlines(self, labelMap, subdivideInputSurface, centroid, decimationAggressiveness, targetNumberOfPoints, makeTables, makeModels):
        import ExtractCenterline
        extractLogic = ExtractCenterline.ExtractCenterlineLogic()
        centerlineCurveNode = None
        centerlinePropertiesTableNode = None
        segVol = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(segVol, labelMap)
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(segVol, segmentationNode)
        a = list(segmentationNode.GetSegmentation().GetSegmentIDs())
        startsAndEnds = []
        startsAndEnds = self.batchSaE(labelMap, centroid)
        i = 0
        if not (makeTables and makeModels):
            return
        while (i < len(a)):
        #  for seg in a:
            try:
                endPointsMarkupsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "VesselTreeEndpoints_" + str(i + 1))
                seg = a[i]
                print(seg)
        # Preprocess the surface
                inputSurfacePolyData = extractLogic.polyDataFromNode(segmentationNode, seg)
                thisSandE = [startsAndEnds[int(seg.lstrip("Label_")) - 1][0], startsAndEnds[int(seg.lstrip("Label_")) - 1][1]]
                preprocessedPolyData = extractLogic.preprocess(inputSurfacePolyData, targetNumberOfPoints, decimationAggressiveness, subdivideInputSurface)
        # Auto-detect the endpoints
        # TODO: Add Startpoint/inlet specification 
                networkPolyData = extractLogic.extractNetwork(preprocessedPolyData, endPointsMarkupsNode)
                startPointPosition = thisSandE[0]
                endpointPositions = extractLogic.getEndPoints(networkPolyData, startPointPosition)
                endPointsMarkupsNode.RemoveAllControlPoints()
                for position in endpointPositions:
                    endPointsMarkupsNode.AddControlPoint(vtk.vtkVector3d(position))
            
             # Extract the centerline
                centerlineCurveNode = None
                if makeModels:
                    centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode", "Centerline_curve_" + str(i + 1))
                centerlinePolyData, voronoiDiagramPolyData = extractLogic.extractCenterline(preprocessedPolyData, endPointsMarkupsNode)
                centerlinePropertiesTableNode = None
                if makeTables:
                    centerlinePropertiesTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "VesselTree_" + str(i + 1))
                extractLogic.createCurveTreeFromCenterline(centerlinePolyData, centerlineCurveNode, centerlinePropertiesTableNode)
                i += 1
            except:
                print("An error occured while processing vessel tree number: " + str(i + 1))
                i += 1
                continue


    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                decimationAggressiveness: float,
                targetNumberOfPoints: float,
                subdivideInputSurface: bool = False,
                makeTables: bool = True,
                makeModels: bool = True,
                minVesselSize: float = 3) -> None:
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

        #perform processing
        inputVolumeAsArray = slicer.util.arrayFromVolume(inputVolume)
        self.findAllCenterlines(self.vesselFinder(inputVolumeAsArray, minVesselSize), subdivideInputSurface, self.centroidFinder(inputVolumeAsArray), decimationAggressiveness, targetNumberOfPoints, makeTables, makeModels)
        processAllCenterlineTables()
        

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
        print(f"Processing completed in {stopTime-startTime:.2f} seconds")
#processAll errors if computer runs out of memory
    def processAll(self,
                allInputVolumes,
                decimationAggressiveness: float,
                targetNumberOfPoints: float,
                subdivideInputSurface: bool = False,
                makeTables: bool = True,
                makeModels: bool = True,
                minVesselSize: float = 3) -> None:
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
        if allInputVolumes:
            for x in allInputVolumes:
                inputVolumeAsArray = slicer.util.arrayFromVolume(x)
                self.findAllCenterlines(self.vesselFinder(inputVolumeAsArray, minVesselSize), subdivideInputSurface, self.centroidFinder(inputVolumeAsArray), decimationAggressiveness, targetNumberOfPoints, makeTables, makeModels)

        #perform processing
        
        
    
        

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")
