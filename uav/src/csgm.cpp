#include "csgm.h"

// STD
#include <iostream>

// OpenCV
#include <opencv2/imgproc.hpp>

// Tulipp
#include "config.hpp"

namespace tulipp
{
namespace usecase
{
namespace uav
{

//==================================================================================================
CSgm::CSgm()
{
}

//==================================================================================================
CSgm::~CSgm()
{
}

//==================================================================================================
cv::Mat CSgm::computeDepth(const cv::Mat &iFrame1, const cv::Mat &iFrame2, const float &iImgScale)
{
  cv::Mat depthMap;

  assert(iFrame1.size() == iFrame2.size());
  assert(iFrame1.type() == iFrame2.type());
  assert(iFrame1.type() == CV_32FC1);

  //--- assign frame sizes ---
  const cv::Size ORIGINAL_FRAME_SIZE = iFrame1.size();
  const cv::Size WORKING_FRAME_SIZE = cv::Size(ORIGINAL_FRAME_SIZE.width * iImgScale,
                                               ORIGINAL_FRAME_SIZE.height * iImgScale);

  cv::Mat workingFrame1, workingFrame2;

  //--- if scale factor is unequal to 1.0, resize ---
  if(std::fabs(iImgScale - 1.f) > 0.000001)
  {
    cv::resize(iFrame1, workingFrame1, WORKING_FRAME_SIZE);
    cv::resize(iFrame2, workingFrame2, WORKING_FRAME_SIZE);

    config::MAX_DISPARITY *= iImgScale;
  }
  else
  {
    workingFrame1 = iFrame1;
    workingFrame2 = iFrame2;
  }

  //--- assign disparity image ---
  cv::Mat leftDisparityImg = cv::Mat::zeros(WORKING_FRAME_SIZE, CV_32FC1);
  cv::Mat rightDisparityImg = cv::Mat::zeros(WORKING_FRAME_SIZE, CV_32FC1);

#ifdef VERBOSITY
  std::cout << "Computing Cost Volume..." << std::endl;
#endif

  //--- assign and compute cost volume ---
  costs::TCostVolume<float> leftCostVolume = costs::TCostVolume<float>(WORKING_FRAME_SIZE.width * WORKING_FRAME_SIZE.height,
                                        std::vector<float>(config::MAX_DISPARITY + 1, 0.f));
  costs::TCostVolume<float> rightCostVolume = costs::TCostVolume<float>(WORKING_FRAME_SIZE.width * WORKING_FRAME_SIZE.height,
                                        std::vector<float>(config::MAX_DISPARITY + 1, 0.f));
  computeCostVolume<float>(workingFrame1, workingFrame2, TO_LEFT, leftCostVolume);
  computeCostVolume<float>(workingFrame2, workingFrame1, TO_RIGHT, rightCostVolume);

#ifdef VERBOSITY
  std::cout << "Computing Disparity..." << std::endl;
#endif

  leftDisparityImg = computeDispImg(workingFrame1, workingFrame2, leftCostVolume);
  rightDisparityImg = computeDispImg(workingFrame2, workingFrame1, rightCostVolume);

#ifdef VERBOSITY
  std::cout << "Median Filtering..." << std::endl;
#endif
  //--- perform median filter ---
  cv::medianBlur(leftDisparityImg, leftDisparityImg, 3);
  cv::medianBlur(rightDisparityImg, rightDisparityImg, 3);

#ifdef VERBOSITY
  std::cout << "Consistency Check..." << std::endl;
#endif

  //--- perform consistency check ---
  performConsitencyCheck<float>(leftDisparityImg, rightDisparityImg, -1);

#ifdef VERBOSITY
  std::cout << "Computing depth..." << std::endl;
#endif

  //--- scale back to original size ---
  cv::resize(leftDisparityImg, leftDisparityImg, ORIGINAL_FRAME_SIZE, 0,0, cv::INTER_NEAREST);
  leftDisparityImg /= iImgScale;

  //--- reproject disparity into depth map ---
  depthMap = computeDepthFromDisparity1D<float>(leftDisparityImg, config::FOCAL_LENGTH,
                                                config::BASELINE);

  return depthMap;
}

//==================================================================================================
template<typename T>
void CSgm::computeCostVolume(const cv::Mat &iReferenceImg, const cv::Mat &iMatchingImg,
                             const ESamplingDirection iSamplingDirection,
                             costs::TCostVolume<T> &oCostVolume)
{
  assert(iReferenceImg.size() == iMatchingImg.size());
  assert(iReferenceImg.type() == iMatchingImg.type());
  assert(config::MAX_DISPARITY > 0);

  //--- set constants ---
  const cv::Size IMG_SIZE = iReferenceImg.size();

  //--- compute sampling offset depending on iSamplingDirection ---
  cv::Point2i samplingOffset(0,0);
  int samplingStartX = 0,
      samplingStartY = 0,
      samplingStopX = IMG_SIZE.width,
      samplingStopY = IMG_SIZE.height;
  switch (iSamplingDirection) {
    case TO_LEFT:
    {
      samplingOffset.x = -1;

      samplingStartX = config::MAX_DISPARITY;
    }
    break;

    case TO_RIGHT:
    {
      samplingOffset.x = 1;

      samplingStopX = IMG_SIZE.width - config::MAX_DISPARITY;
    }
    break;

    case TO_TOP:
    {
      samplingOffset.y = -1;

      samplingStartY = config::MAX_DISPARITY;
    }
    break;

    case TO_BOTTOM:
    {
      samplingOffset.y = 1;

      samplingStopY = IMG_SIZE.height - config::MAX_DISPARITY;
    }
    break;

    default:
    break;
  }

  //--- loop through image and sample along scan line ---
#ifdef OPENMP
  #pragma omp parallel for shared(oCostVolume) collapse(2)
#endif
  for(int y = samplingStartY; y < samplingStopY; y++)
  {
    for(int x = samplingStartX; x < samplingStopX; x++)
    {
      cv::Point2i refPx(x,y);
      cv::Point2i matchingPx;
      for(int d = 0; d <= config::MAX_DISPARITY; d++)
      {
        //--- calculate matching pixel by adding sampling offset and handling border ---
        matchingPx = refPx + d * samplingOffset;
        matchingPx.x = cv::borderInterpolate(matchingPx.x, IMG_SIZE.width, cv::BORDER_REFLECT);
        matchingPx.y = cv::borderInterpolate(matchingPx.y, IMG_SIZE.height, cv::BORDER_REFLECT);

        //--- compute matching costs and save into cost volume ---
        oCostVolume[y * IMG_SIZE.width + x][d] =
            costs::computeBirchfieldTomasie<float>(iReferenceImg, iMatchingImg, refPx, matchingPx);
      }
    }
  }
}

//==================================================================================================
template<typename T>
cv::Mat CSgm::computeDispImg(const cv::Mat &iReferenceImg, const cv::Mat &iMatchingImg,
                             costs::TCostVolume<T> const & iCostVolume)
{
  //--- set constants ---
  const cv::Size FRAME_SIZE = iReferenceImg.size();

  assert(iCostVolume.size() == FRAME_SIZE.width * FRAME_SIZE.height);

  //--- make image of datatype T ---
  cv::Mat disparityImg = cv::Mat(FRAME_SIZE, cv::DataType<T>::type);

  //--- assign aggregated cost volume ---
  costs::TCostVolume<T> aggrCostVol = costs::TCostVolume<T>(iCostVolume.size(),
                                                    std::vector<T>(iCostVolume[0].size(), T(0)));

  //--- loop through cost volume and aggregate ---
#ifdef OPENMP
  #pragma omp parallel for shared(disparityImg, iCostVolume, aggrCostVol) collapse(2)
#endif
  for(int y = 0; y < FRAME_SIZE.height; y++)
  {
    for(int x = 0; x < FRAME_SIZE.width; x++)
    {
      std::vector<T>& aggrCosts = aggrCostVol[y * FRAME_SIZE.width + x];
      aggrCosts = performSgmAggregation(iReferenceImg, iMatchingImg, {x,y}, iCostVolume);

      int bestCostIdx = std::min_element(aggrCosts.begin(), aggrCosts.end()) -
                        aggrCosts.begin();
      T dispRefinement = computeDispRefinement<T>(aggrCosts, bestCostIdx);

      T disparity = (T)bestCostIdx + dispRefinement;
      if(std::isnan(disparity) || std::isinf(disparity))
        disparity = 0.f;

      disparityImg.at<T>(y, x) = std::max(disparity,(T)0.001);
    }
  }

  return disparityImg;
}

//==================================================================================================
template<typename T>
std::vector<T> CSgm::performSgmAggregation(const cv::Mat &iReferenceImg, const cv::Mat &iMatchingImg,
                                           std::array<int,2> const &iPx,
                                           costs::TCostVolume<T> const &iCostVolume)
{
  //--- check if cost volumes are of correct size ---
  assert(iCostVolume.size() == (iReferenceImg.size().width * iReferenceImg.size().height));

  cv::Point2i pxPoint = cv::Point2i(iPx[0], iPx[1]);
  std::vector<std::vector<T>> paths;
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(0, -1))); // top path
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(1, -1))); // top-right path
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(1, 0)));  // right path
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(1, 1)));  // bottom-right path
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(0, 1)));  // bottom path
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(-1, 1))); // bottom-left path
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(-1, 0))); // left path
  paths.push_back(aggrAlongPath(iReferenceImg, iMatchingImg,
                                iCostVolume, pxPoint, pxPoint, cv::Point2i(-1, -1)));  // top-left path

  std::vector<T> aggrCosts = std::vector<T>(paths[0].size(), (T)0);
  std::for_each(paths.begin(), paths.end(), [&aggrCosts](std::vector<T> dispCosts){
                    for(uint i = 0; i < dispCosts.size(); i++)
                      aggrCosts[i] += dispCosts[i];
                  }
                );

  return aggrCosts;
}

//==================================================================================================
template<typename T>
std::vector<T> CSgm::aggrAlongPath(const cv::Mat &iReferenceImg, const cv::Mat &iMatchingImg,
                                   costs::TCostVolume<T> const &iCostVol, cv::Point2i const &centerPx,
                                   cv::Point2i const &pathPx, const cv::Point2i &r,
                                   int const pathCounter)
{

  //--- compute constants current pixel index ---
  const cv::Size FRAME_SIZE = iReferenceImg.size();
  const int CURRENT_PX_IDX = pathPx.y * FRAME_SIZE.width + pathPx.x;

  std::vector<T> currentMatchCosts = iCostVol[CURRENT_PX_IDX];

  //--- if pixel has reached boarder return only matching costs ---
  if(pathPx.x + r.x < 0 || pathPx.y + r.y < 0
     || pathPx.x + r.x > FRAME_SIZE.width -1 || pathPx.y + r.y > FRAME_SIZE.height -1
     || pathCounter == config::MAX_PATH_LENGTH)
    return currentMatchCosts;

  //--- get results next recursive step ---
  std::vector<T> recursiveResults = aggrAlongPath(iReferenceImg, iMatchingImg,
                                                  iCostVol, centerPx, pathPx + r, r, pathCounter + 1);

  //--- compute min path cost ---
  float minPathCost = *std::min_element(recursiveResults.begin(), recursiveResults.end());

  //--- adapt p2 ---
  float p2 = config::PENALTY_2 / (std::fabs(iReferenceImg.at<T>(centerPx.y, centerPx.x) -
                                         iMatchingImg.at<T>(pathPx.y, pathPx.x)));
  p2 = MAX(config::PENALTY_1, p2);

  //--- aggregate costs ---
  for(int d = 0; d < currentMatchCosts.size(); d++)
  {
    currentMatchCosts[d] += MIN(recursiveResults[d],
                                MIN(recursiveResults[MAX(d-1, 0)] + config::PENALTY_1,
                                    MIN(recursiveResults[MIN(d+1, recursiveResults.size()-1)]
                                          + config::PENALTY_1,
                                        minPathCost + p2
                                       )
                                    )
                                ) - minPathCost;
  }

  return currentMatchCosts;
}

//==================================================================================================
template<typename T>
T CSgm::computeDispRefinement(std::vector<T> const &iDispCosts, int const iBestCostIdx) {
  //--- C value of parabola is equl to minimum ---
  //NOTE if disparity differences are larger than 1 add tLeft and tRight

  //--- comput A value of parabola
  T parabA = ( iDispCosts[std::abs(iBestCostIdx - 1)] +
                 iDispCosts[cv::borderInterpolate(iBestCostIdx + 1, iDispCosts.size(),
                                                  cv::BORDER_REFLECT)] -
                 2.f * iDispCosts[iBestCostIdx] ) / 2.f;

  //--- compute B value of parabola
  T parabB = (iDispCosts[std::abs(iBestCostIdx  - 1)] -
                 parabA - iDispCosts[iBestCostIdx]) * -1;

  return std::min(std::max((-parabB / (2.f * parabA)),(T)-1),(T)1);
}

//==================================================================================================
template<typename T>
void CSgm::performConsitencyCheck(cv::Mat& ioDispImg1, cv::Mat const & iDispImg2, T const iInconsValue)
{
  assert(ioDispImg1.size() == iDispImg2.size());
  assert(ioDispImg1.type() == cv::DataType<T>::type);
  assert(ioDispImg1.type() == iDispImg2.type());

  const cv::Size FRAME_SIZE = ioDispImg1.size();

  //--- loop through disparity images and compare values ---
#ifdef OPENMP
  #pragma omp parallel for shared(ioDispImg1, iDispImg2) collapse(2)
#endif
  for(int y = 0; y < FRAME_SIZE.height; y++)
  {
    for(int x = 0; x < FRAME_SIZE.width; x++)
    {
      T disp1 = ioDispImg1.at<T>(y,x);
      if(x-disp1 < 0) continue;

      T disp2 = iDispImg2.at<T>(y,(int)std::rint(x - disp1));

      //--- if absolute differenc between both disparites is greater than 1, assign inconsistancy value ---
      if(std::fabs(disp1 - disp2) > 1.f)
        ioDispImg1.at<T>(y,x) = iInconsValue;
    }
  }
}

//==================================================================================================
template<typename T>
cv::Mat CSgm::computeDepthFromDisparity1D(const cv::Mat& iDispImg, const T& iFocalLengthX,
                                    const T& iBaseline)
{
  cv::Mat depthMap = cv::Mat::zeros(iDispImg.size(), cv::DataType<T>::type);

  assert(iDispImg.type() == depthMap.type());

  const cv::Size FRAME_SIZE = iDispImg.size();

#ifdef OPENMP
  #pragma omp parallel for shared(depthMap, iDispImg) collapse(2)
#endif
  for(int y = 0; y < FRAME_SIZE.height; y++)
  {
    for(int x = 0; x < FRAME_SIZE.width; x++)
    {
      T disparity = iDispImg.at<T>(y,x);

      //--- if disparity is near zero set depth to 0 ---
      if(disparity < (T)0.1 && disparity > -(T)0.1)
        depthMap.at<T>(y,x) = 0;
      else
        depthMap.at<T>(y,x) = (iFocalLengthX * iBaseline) / disparity;
    }
  }

  return depthMap;
}

} // namespace uav
} // namespace usecase
} // namespace tulipp
