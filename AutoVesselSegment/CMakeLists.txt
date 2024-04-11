cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(AutoVesselSegment)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/AutoVesselSegment")
set(EXTENSION_CATEGORY "Examples")
set(EXTENSION_CONTRIBUTORS "Aaron McClellan (College of Charleston), Tyler Hartman, Hugo Bachaumard")
set(EXTENSION_DESCRIPTION "Automatically map blood vessel tree of the lungs.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/AutoVesselSegment.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/AutoVesselSegment/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(lungVesselSeg)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
