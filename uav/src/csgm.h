#ifndef CSGM_H
#define CSGM_H

// OpenCV
#include <opencv2/core.hpp>

// Tulipp
#include "costs.hpp"

// STD
#include <array>
#include <cmath>

namespace tulipp
{
namespace usecase
{

/**
 * @brief Namespace containg reference implentation for TULIPP UAV usecase. Within this usecase
 * a stereo estimation is to be performed with two front-facing cameras in order to do collision
 * avoidance.
 * @author Ruf, Boitumelo <boitumelo.ruf@iosb.fraunhofer.de>
 * @copyright Fraunhofer Institute of Optronics, System Technologies and Image Exploitation, 2016
 */
namespace uav
{

/**
 * @brief Enumation of possible sampling directions along a scanline.
 */
enum ESamplingDirection
{
  TO_LEFT, TO_RIGHT, TO_TOP, TO_BOTTOM
};

/**
 * @brief Class performing depth/disparity estimation with Semi Globla Matching (SGM) aggregation and
 * BirchfieldTomasie cost functions.
 * @par Reference\n
 *  <a href="http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1467526&tag=1">SGM Paper</a>
 */
class CSgm
{

  //--- FUNCTION DECLERATION ---

public:

  /**
   * @brief Constructor
   */
  CSgm();

  /**
   * @brief Destructor
   */
  ~CSgm();

  /**
   * @brief Compute depth image from two input frames.
   * @param[in] iFrame1 Input frame 1.
   * @param[in] iFrame2 Input frame 2.
   * @param[in] iImgScale Factor with which to scale the input images.
   * @return Depth image corresponding to input frame 1
   */
  cv::Mat computeDepth(cv::Mat const &iFrame1, cv::Mat const &iFrame2, float const & iImgScale);

private:

  /**
   * @brief Compute costvolume by performing scanline sampling between the reference image and
   * a matching image. As cost function the BirchfieldTomasi costfunction is used.
   * @details For each pixel in the reference image the matching image is sampled with a disparity
   * in the absolute range of [0, config::MAX_DISPARITY] along the scanline. For each sampled matching
   * pixel, a cost value is computed indicating how similar these two pixels are. These cost values
   * are stored in a costvolume.
   * @param[in] iReferenceImg Reference image.
   * @param[in] iMatchingImg Matching image.
   * @param[in] iSamplingDirection Direction of sampling along the scanline
   * relative to reference image.
   * @param[out] oCostVolume Resulting cost volume of size [width x height][config::MAX_DISPARITY + 1]
   * @par Reference\n
   *  <a href="http://link.springer.com/article/10.1023/A:1008160311296">BirchfieldTomasi</a>
   */
  template<typename T>
  void computeCostVolume(cv::Mat const &iReferenceImg, cv::Mat const &iMatchingImg,
                         ESamplingDirection const iSamplingDirection,
                         costs::TCostVolume<T> &oCostVolume);

  /**
   * @brief Method to compute disparity image by aggregating the cost and pick the disparity
   * with the minimum aggregated costs.
   * Disparity image will be of type T and the cost aggregation is performed with the SGM path
   * aggeragtion. Additionaly the disparity is subpixel refined by fitting a prabola through the
   * best three costs.
   * @param[in] iReferenceImg Reference image.
   * @param[in] iMatchingImg Matching image.
   * @param[in] iCostVolume Cost volume to use for computation of disparity image.
   * @return Single channel disparity image of type T.
   */
  template<typename T>
  cv::Mat computeDispImg(cv::Mat const &iReferenceImg, cv::Mat const &iMatchingImg,
                         costs::TCostVolume<T> const & iCostVolume);

  /**
   * @brief Method to aggregate given costvolume with the SGM aggregation method.
   * Along eight paths around each pixel in the reference image the costs are agrgegated.
   * @param[in] iReferenceImg Reference image.
   * @param[in] iMatchingImg Matching image.
   * @param[in] iPx Pixel coordinates for which the cost volume is to be aggregated.
   * @param[in] iCostVolume Costvolume to aggregate of type T.
   * @return List of aggregated cost per disparity.
   * @par Reference\n
   *  <a href="http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1467526&tag=1">SGM Paper</a>
   */
  template<typename T>
  std::vector<T> performSgmAggregation(cv::Mat const &iReferenceImg, cv::Mat const &iMatchingImg,
                                       std::array<int,2> const &iPx,
                                       costs::TCostVolume<T> const &iCostVolume);

  /**
   * @brief Recursive function to compute aggregated matching cost along path r from pixel px with
   * disparity disp.
   * @param[in] iReferenceImg Reference image.
   * @param[in] iMatchingImg Matching image.
   * @param[in] iCostVol Cost volume.
   * @param[in] centerPx Coordinates of Center Pixel for which disparity is computed. This does not
   * change along path.
   * @param[in] pathPx Coordinates of currrent pixel that is looked at on the path r. This changes as
   * iterating along path.
   * @param[in] r Path direction
   * @param[in] pathCounter Counter along path to find length of path.
   * @return List of all costs for all disparities.
   * @par Reference\n
   *  <a href="http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1467526&tag=1">SGM Paper</a>
   */
  template<typename T>
  std::vector<T> aggrAlongPath(cv::Mat const &iReferenceImg, cv::Mat const &iMatchingImg,
                    costs::TCostVolume<T> const &iCostVol, cv::Point2i const &centerPx,
                    cv::Point2i const &pathPx, const cv::Point2i &r, int const pathCounter = 0);

  /**
   * @brief Method to compute depth refinement by fitting a parabola through minimum.
   * @param[in] iDispCosts List of all disparity costs.
   * @param[in] bestCostIdx Index of best cost.
   * @return Subpixel refinement in the range of [-1,1].
   */
  template<typename T>
  T computeDispRefinement(std::vector<T> const &iDispCosts, int const iBestCostIdx);

  /**
   * @brief Method to perform consistency check between two disparity images.
   * If a the absolute difference of the disparity between two pixels is larger than 1, the
   * give inconsistancy value is assinged.
   * @param[in,out] ioDispImg1 Input and output disparity image 1.
   * @param[in] iDispImg2 Disparity image against which the consistency check is to be performed.
   * @param[in] iInconsValue Value to be assigned for inconsistencies.
   */
  template<typename T>
  void performConsitencyCheck(cv::Mat& ioDispImg1, cv::Mat const & iDispImg2, T const iInconsValue);

  /**
   * @brief Method to compute depth map from a 1d disparity image.
   * As the disparity image is 1d it is assumed that the stereo images are rectified, and so
   * only the baseline is required to account for scaling abiguity.
   * @param[in] iDisparityImg
   * @param[in] iFocalLengthX
   * @param[in] iBaseline Baseline in meters
   * @return Single channel depth map of type T.
   */
  template<typename T>
  cv::Mat computeDepthFromDisparity1D(const cv::Mat& iDisparityImg, const T& iFocalLengthX,
                                      const T& iBaseline);

  //--- MEMBER DECLERATION ---

private:

};

} // namespace uav
} // namespace usecase
} // namespace tulipp

#endif // CSGM_H
