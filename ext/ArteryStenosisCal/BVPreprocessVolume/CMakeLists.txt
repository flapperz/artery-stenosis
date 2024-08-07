#-----------------------------------------------------------------------------
set(MODULE_NAME BVPreprocessVolume)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )
# Collect all Python scripts in the ${MODULE_NAME}Lib directory
file(GLOB MODULE_LIB_PYTHON_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_NAME}Lib/*.py")
foreach(python_script ${MODULE_LIB_PYTHON_SCRIPTS})
  get_filename_component(python_script_name ${python_script} NAME)
  list(APPEND MODULE_PYTHON_SCRIPTS ${MODULE_NAME}Lib/${python_script_name})
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
