#ifndef COSTS_HPP
#define COSTS_HPP

// STD
#include <vector>

// OpenCV
#include <opencv2/core.hpp>

namespace tulipp
{
namespace usecase
{
namespace uav
{

/**
 * @brief Namespace holding cost function for SGM disparity computation.
 */
namespace costs
{

  /**
   * @brief Cost volume type.
   * Two dimensional array of type T. First dimension holds pixels, stored row by row
   * (i.e. (x,y) = y*width + x). Second row holds costs per pixel for each disparity.
   */
  template<typename T>
  using TCostVolume = std::vector<std::vector<T>>;

  /**
   * @brief Method to computer Birchfield and Tomasi matching cost between two pixel.
   * Absolute differene between two pixels, where the match is searched in the range of -0.5 and +0.5 pixels.
   * @param[in] frame1 Frame 1
   * @param[in] frame2 Frame 2
   * @param[in] px1 Pixel coordinate in frame 1 (x,y).
   * @param[in] px2 Pixel coordinate in frame 2 (x,y).
   * @return Absolute difference between both pixels.
   * @note No boarder handling is performed.
   * @par Reference\n
   *  <a href="http://link.springer.com/article/10.1023/A:1008160311296">BirchfieldTomasi</a>
   */
  template<typename T>
  static float computeBirchfieldTomasie(cv::Mat const &frame1, cv::Mat const &frame2,
                                        cv::Point2i const px1, cv::Point2i const px2)
  {
    //--- get values of center pixel and half left and half right ---
    float px1_val = frame1.at<T>(px1.y, px1.x);
    float px1_halfleft_val = ((frame1.at<T>(px1.y, px1.x) + frame1.at<T>(px1.y, MAX(px1.x-1, 0))) / 2.f);
    float px1_halfright_val = ((frame1.at<T>(px1.y, px1.x) + frame1.at<T>(px1.y, MIN(px1.x+1, frame1.size().width-1))) / 2.f);
    float val1_max = MAX(px1_val,MAX(px1_halfleft_val, px1_halfright_val)); // max intensity value
    float val1_min = MIN(px1_val,MIN(px1_halfleft_val, px1_halfright_val)); // min intensity value

    float px2_val = frame2.at<T>(px2.y, px2.x);
    float px2_halfleft_val = ((frame2.at<T>(px2.y, px2.x) + frame2.at<T>(px2.y, MAX(px2.x-1, 0))) / 2.f);
    float px2_halfright_val = ((frame2.at<T>(px2.y, px2.x) + frame2.at<T>(px2.y, MIN(px2.x+1,frame2.size().width-1))) / 2.f);
    float val2_max = MAX(px2_val,MAX(px2_halfleft_val, px2_halfright_val)); // max intensity value
    float val2_min = MIN(px2_val,MIN(px2_halfleft_val, px2_halfright_val)); // min intensity value

    //--- lambda for commputation dissimilarity ---
    auto dissim = [](float i1, float iMax2, float iMin2){ return MAX(0,MAX(i1 - iMax2, iMin2 - i1)); };

    //--- compute and return dissimilarity cost ---
    return MIN(dissim(px1_val, val2_max, val2_min), dissim(px2_val, val1_max, val1_min));
  }

} // namespace costs

} // namespace uav
} // namespace usecase
} // namespace tulipp

#endif // COSTS_HPP
