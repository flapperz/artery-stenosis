#include "itkImageFileWriter.h"

#include "itkPluginUtilities.h"

#include "BVCreateGuideLineCLP.h"

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//

namespace
{
  template <typename TPixel>
  int DoIt(int argc, char *argv[], TPixel)
  {
    PARSE_ARGS;

    const unsigned int Dimension = 3;

    typedef TPixel InputPixelType;
    typedef itk::Image<InputPixelType, Dimension> InputImageType;
    typedef itk::ImageFileReader<InputImageType> ReaderType;

    typename ReaderType::Pointer reader = ReaderType::New();
    typename InputImageType::Pointer image;

    reader->SetFileName(inputVolume.c_str());
    reader->Update();
    image = reader->GetOutput();

    typedef itk::Index<3> Index3Type;
    
    typename InputImageType::RegionType region = image->GetLargestPossibleRegion();
    typename InputImageType::SizeType imageSize = region.GetSize();
    std::cout << "size" << imageSize << std::endl;


    Index3Type seed{{307, 204, 160}};
    Index3Type target{{329, 190, 159}};
    
    // Get pixel value example
    // typename InputImageType::IndexType pixelIndex{{334, 186, 157}};
    // Index3Type pixelIndex{{334, 186, 157}};
    // typename InputImageType::PixelType pixelValue = image->GetPixel(pixelIndex);
    // std::cout << pixelValue << std::endl;

    typedef std::pair<double, Index3Type> qItemType;
    // std::set<qItemType, std::vector<qItemType>, tupleGreater<double, Index3Type>> pq;
    std::set<qItemType> pq;
    std::map<Index3Type, double> dist_map;
    std::map<Index3Type, Index3Type> pred_map;

    pq.insert({0, seed});
    dist_map[seed] = 0;

    bool is_reach = false;

    int it = 0;
    const int MAX_IT = 5e6;

    while (!pq.empty() && !is_reach && it < MAX_IT)
    {
      it += 1;

      qItemType u_pair = *pq.begin();
      pq.erase(pq.begin());

      double u_dist = u_pair.first;
      Index3Type u_idx = u_pair.second;
      
      if (u_idx == target)
      {
        is_reach = true;
        break;
      }
      
      if (dist_map.find(u_idx) != dist_map.end() && u_dist > dist_map[u_idx])
      {
        std::cout << "In here too wtf!!!" << std::endl;
        continue;
      }
      for (auto const& vi: {-1, 0, 1})
      {
        for (auto const &vj : {-1, 0, 1})
        {
          for (auto const &vk : {-1, 0, 1})
          {
            if (vi == 0 && vj == 0 && vk == 0)
              continue;
            Index3Type v_idx = u_idx;
            v_idx[0] += vi;
            v_idx[1] += vj;
            v_idx[2] += vk;
            
            int dist_uv = 1;
            // process
            double v_value = image->GetPixel(v_idx);
            v_value = v_value > -50 ? 2000000000 : v_value + 4000;

            double curr_cost = dist_uv * v_value;
            double dist = u_dist + curr_cost;
            // std::cout << "cost: " << curr_cost << " " << u_dist << " " << dist << std::endl;

            if (dist_map.find(v_idx) == dist_map.end() || dist < dist_map[v_idx])
            {
              dist_map[v_idx] = dist;
              pred_map[v_idx] = u_idx;
              pq.insert({dist, v_idx});
              // std::cout << "in here" << pq.size() << std::endl;
            }
          }
        }
      }
    }
    
    std::cout << "it: " << it << std::endl;
    
    if (! is_reach)
      return EXIT_FAILURE;

    it = 0;
    // crawl
    Index3Type crawler = target;
    Index3Type parent;
    std::vector<Index3Type> path;
    while (pred_map.find(crawler) != pred_map.end() && it < 512 + 275 + 512)
    {
      it += 1;
      parent = pred_map[crawler];
      path.push_back(parent);
      crawler = parent;
    }
    
    std::cout << path.size() << std::endl;
    
    // for (auto& idx: path)
    // {
    //   std::cout << idx << std::endl;
    // }

    return EXIT_SUCCESS;
  }
}

int main(int argc, char *argv[])
{
  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType pixelType;
  itk::ImageIOBase::IOComponentType componentType;
  
  try
  {
    itk::GetImageType(inputVolume, pixelType, componentType);

    // This filter handles all types on input, but only produces
    // signed types
    switch (componentType)
    {
    case itk::ImageIOBase::UCHAR:
      return DoIt(argc, argv, static_cast<unsigned char>(0));
      break;
    case itk::ImageIOBase::CHAR:
      return DoIt(argc, argv, static_cast<signed char>(0));
      break;
    case itk::ImageIOBase::USHORT:
      return DoIt(argc, argv, static_cast<unsigned short>(0));
      break;
    case itk::ImageIOBase::SHORT:
      return DoIt(argc, argv, static_cast<short>(0));
      break;
    case itk::ImageIOBase::UINT:
      return DoIt(argc, argv, static_cast<unsigned int>(0));
      break;
    case itk::ImageIOBase::INT:
      return DoIt(argc, argv, static_cast<int>(0));
      break;
    case itk::ImageIOBase::ULONG:
      return DoIt(argc, argv, static_cast<unsigned long>(0));
      break;
    case itk::ImageIOBase::LONG:
      return DoIt(argc, argv, static_cast<long>(0));
      break;
    case itk::ImageIOBase::FLOAT:
      return DoIt(argc, argv, static_cast<float>(0));
      break;
    case itk::ImageIOBase::DOUBLE:
      return DoIt(argc, argv, static_cast<double>(0));
      break;
    case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
    default:
      std::cerr << "Unknown input image pixel component type: ";
      std::cerr << itk::ImageIOBase::GetComponentTypeAsString(componentType);
      std::cerr << std::endl;
      return EXIT_FAILURE;
      break;
    }
  }

  catch (itk::ExceptionObject &excep)
  {
    std::cerr << argv[0] << ": exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
  }

  // std::ofstream rts;
  // rts.open(returnParameterFile.c_str());
  // rts << "flattenFiducials = 0.5,0.6,0.7" << std::endl;

  return EXIT_SUCCESS;
}
