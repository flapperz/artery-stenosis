// #include <vtkNew.h>
// #include <vtkMRMLMarkupsJsonStorageNode.h>
// #include <vtkMRMLMarkupsFiducialNode.h>
#include "itkPluginUtilities.h"

#include "MarkupsCliPocCLP.h"



int main(int argc, char *argv[])
{
    PARSE_ARGS;
    
    // vtkNew<vtkMRMLMarkupsFiducialNode> copiedFiducialNode;
    // copiedFiducialNode->SetName("seedsCopy");
    // for (unsigned int i = 0; i < 5; ++i)
    // {
    //     copiedFiducialNode->AddControlPoint(vtkVector3d(0.0, 0.0, 0.0));
    //     // toggle some settings
    //     if (i == 0)
    //     {
    //         copiedFiducialNode->SetNthControlPointLocked(i, true);
    //         copiedFiducialNode->SetNthControlPointSelected(i, false);
    //         copiedFiducialNode->SetNthControlPointVisibility(i, false);
    //     }
    // }
    
    // std::cout << outMarkupsFile << std::endl;

    // if (!outMarkupsFile.empty())
    // {
    //     vtkNew<vtkMRMLMarkupsJsonStorageNode> outMarkupsStorageNode;
    //     outMarkupsStorageNode->SetFileName(outMarkupsFile.c_str());

    //     outMarkupsStorageNode->UseLPSOn();
    //     outMarkupsStorageNode->WriteData(copiedFiducialNode.GetPointer());
    // }

    return EXIT_SUCCESS;
}