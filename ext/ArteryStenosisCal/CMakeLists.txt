cmake_minimum_required(VERSION 3.13.4)

project(Artery Stenosis Stenosis Calculator)

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
add_subdirectory(StenosisCal)
add_subdirectory(CreateGuideLinePython)
add_subdirectory(DefaultPythonMod)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
