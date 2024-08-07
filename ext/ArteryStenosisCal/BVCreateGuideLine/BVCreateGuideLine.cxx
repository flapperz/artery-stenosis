#include "itkImageFileWriter.h"

#include "itkPluginUtilities.h"

#include "BVCreateGuideLineCLP.h"

#ifdef NDEBUG
#define DEBUG_MSG(str) do { } while ( false )
#else
#define DEBUG_MSG(str) do { std::cout << str << std::endl; } while( false )
#endif

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//

namespace
{
  typedef itk::Index<3> Index3Type;

  std::vector<Index3Type> ParseFiducials(std::vector<int> raw)
  {
    std::vector<Index3Type> result;

    if (!raw.size() || raw.size() % 3 != 0)
      return result;

    for (int i=0; i<raw.size(); i+=3)
      result.push_back({raw[i], raw[i+1], raw[i+2]});

    return result;
  }
  
  std::string FlattenFiducials(std::vector<Index3Type> markers)
  {
    std::string result;

    if (!markers.size())
      return result;

    for (auto& e: markers)
    {
      result+=std::to_string(e[0])+",";
      result+=std::to_string(e[1])+",";
      result+=std::to_string(e[2])+",";
    }
    result.pop_back();

    return result;
  }
  
  template <typename TImage>
  int GetPathDijkstra(Index3Type seed, Index3Type dest, typename TImage::Pointer image, std::vector<Index3Type> &path)
  {
    typedef TImage ImageType;

    typedef std::pair<double, Index3Type> qItemType;
    // std::set<qItemType, std::vector<qItemType>, tupleGreater<double, Index3Type>> pq;
    std::set<qItemType> pq;
    std::map<Index3Type, double> dist_map;
    std::map<Index3Type, Index3Type> pred_map;

    pq.insert({0, seed});
    dist_map[seed] = 0;
    
    const double CONST_SQRT2 = 1.41421356237;
    const double CONST_SQRT3 = 1.73205080757;

    typename ImageType::RegionType region = image->GetLargestPossibleRegion();
    typename ImageType::SizeType imageSize = region.GetSize();
    DEBUG_MSG("image size:" << imageSize << std::endl);

    bool is_reach = false;


    int it = 0;
    const int MAX_IT = std::min((int)5e6, (int)region.GetNumberOfPixels());

    while (!pq.empty() && !is_reach && it < MAX_IT)
    {
      it += 1;

      qItemType u_pair = *pq.begin();
      pq.erase(pq.begin());

      double u_dist = u_pair.first;
      Index3Type u_idx = u_pair.second;

      if (u_idx == dest)
      {
        is_reach = true;
        break;
      }

      // u node distance should be set before add to pq
      if (u_dist != dist_map[u_idx])
      {
        DEBUG_MSG("Ignore revisiting");
        continue;
      }
      for (auto const &vi : {-1, 0, 1})
        for (auto const &vj : {-1, 0, 1})
          for (auto const &vk : {-1, 0, 1})
          {
            if (vi == 0 && vj == 0 && vk == 0)
              continue;
            Index3Type v_idx = u_idx;
            v_idx[0] += vi;
            v_idx[1] += vj;
            v_idx[2] += vk;
            if (!region.IsInside(v_idx))
              continue;

            double dist_uv = 1;
            switch (abs(vi) + abs(vj) + abs(vk))
            {
            case 2:
              dist_uv = CONST_SQRT2;
              break;
            case 3:
              dist_uv = CONST_SQRT3;
              break;
            default:
              break;
            }
            // process
            double v_value = image->GetPixel(v_idx);

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

    std::cout << "it: " << it << std::endl;

    if (!is_reach)
      return EXIT_FAILURE;

    it = 0;
    // crawl
    Index3Type crawler = dest;
    Index3Type parent;
    while (pred_map.find(crawler) != pred_map.end() && it < 512 + 275 + 512)
    {
      it += 1;
      parent = pred_map[crawler];
      path.push_back(parent);
      crawler = parent;
    }

    return EXIT_SUCCESS;
  }

  template <typename TPixel>
  int DoIt(int argc, char *argv[], TPixel tpixelVal)
  {
    PARSE_ARGS;

    std::vector<Index3Type> markers = ParseFiducials(inFlattenMarkersIJK);
    DEBUG_MSG("input markers");
    for (int i = 0; i < markers.size(); ++i)
    {
      DEBUG_MSG(markers[i]);
    }
    if (markers.size() < 2)
    {
      // TODO: better handling this case
      std::cerr << "input have less than 2 point markers" << std::endl;
      return EXIT_FAILURE;
    }

    const unsigned int Dimension = 3;

    typedef itk::Image<TPixel, Dimension> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;

    typename ReaderType::Pointer reader = ReaderType::New();
    typename ImageType::Pointer image;

    reader->SetFileName(inputVolume.c_str());
    reader->Update();
    image = reader->GetOutput();


    std::vector<Index3Type> path;
    // use seed as target / target as seed to avoid reveresing vecotr
    for (int i = 0; i < markers.size() - 1; ++i)
    {
      auto seed = markers[i + 1];
      auto dest = markers[i];
      std::vector<Index3Type> subPath;
      int retVal = GetPathDijkstra<ImageType>(seed, dest, image, subPath);
      if (retVal == EXIT_FAILURE)
      {
        std::cerr << "Can't find path between index:" << i << "to" << i+1 << std::endl;
        return EXIT_FAILURE;
      }
      path.insert(path.end(), subPath.begin(), subPath.end());
      DEBUG_MSG("dijkstra: " << retVal << " : " << subPath.size());
    }
    path.push_back(markers.back());
    DEBUG_MSG("path size: " << path.size());

    std::string flattenMarkersStr = FlattenFiducials(path);
    std::ofstream rts;
    rts.open(returnParameterFile.c_str());

    rts << "outFlattenMarkersIJK = " << flattenMarkersStr << std::endl;
    rts.close();

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
    std::cout << "the image type isssss:" << componentType << std::endl;
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

  return EXIT_SUCCESS;
}
