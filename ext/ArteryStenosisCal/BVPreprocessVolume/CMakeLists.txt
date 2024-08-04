#-----------------------------------------------------------------------------
set(MODULE_NAME BVPreprocessVolume)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )
file(GLOB MODULE_LIB_PYTHON_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_NAME}Lib/*.py")
foreach(python_script ${MODULE_LIB_PYTHON_SCRIPTS})
  list(APPEND MODULE_PYTHON_SCRIPTS ${python_script})
endforeach()

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
