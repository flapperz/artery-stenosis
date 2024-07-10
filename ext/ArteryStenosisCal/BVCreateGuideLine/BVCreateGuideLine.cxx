// #include "itkImageFileWriter.h"

#include "itkPluginUtilities.h"

#include "BVCreateGuideLineCLP.h"

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//


int main( int argc, char * argv[] )
{
  PARSE_ARGS;
  
  
  
  std::ofstream rts;
  rts.open(returnParameterFile.c_str());
  rts << "flattenFiducials = 0.5,0.6,0.7" << std::endl;
  
  

  return EXIT_SUCCESS;
}
