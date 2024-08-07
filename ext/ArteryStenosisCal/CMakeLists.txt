cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

# set(Slicer_DIR /opt/sr561/Slicer-build)
# set(Os)

find_package(Slicer COMPONENTS ConfigurePrerequisites REQUIRED)

# project(Artery Stenosis Calculator)
project(ArteryStenosisCal)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/ArteryStenosisCal")
set(EXTENSION_CATEGORY "Artery Stenosis")
set(EXTENSION_CONTRIBUTORS "Krit Cholapand (Chulalongkorn University)")
set(EXTENSION_DESCRIPTION "This is description")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/ArteryStenosisCal.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/ArteryStenosisCal/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
# add_subdirectory(StenosisCal)
# add_subdirectory(CreateGuideLinePython)
# add_subdirectory(DefaultPythonMod)
add_subdirectory(BVCreateGuideLine)
add_subdirectory(BVStenosisMeasurement)
add_subdirectory(BVPreprocessVolume)
add_subdirectory(BVAirSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
