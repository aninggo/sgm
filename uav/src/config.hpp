#ifndef CONFIG_HPP
#define CONFIG_HPP
namespace tulipp
{
namespace usecase
{
namespace uav
{

/**
 * @brief Namespace holding configuration parameters of SGM disparity computation.
 */
namespace config
{

  /**
   * @brief Maximum absolute disparity for correspondence search.
   */
  int MAX_DISPARITY = 120;

  /**
   * @brief Penalty 1 of SGM.
   * @par Reference\n
   *  <a href="http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1467526&tag=1">SGM Paper</a>
   */
  float PENALTY_1 = 0.05;

  /**
   * @brief Penalty 2 of SGM.
   * @par Reference\n
   *  <a href="http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1467526&tag=1">SGM Paper</a>
   */
  float PENALTY_2 = 0.1;

  /**
   * @brief Maximum length of aggregation path. Set to -1 in order for the aggregation to go up to
   * iamge borders.
   */
  int MAX_PATH_LENGTH = 10;

  /**
   * @brief Horizontal Focal lenght of camera. Needed for reprojection of disparity to depth.
   */
  float FOCAL_LENGTH = 746.765;

  /**
   * @brief Baseline of stereo setup. Needed for reprojection of disparity to depth.
   */
  float BASELINE = 0.21726;

} // namespace config

} // namespace uav
} // namespace usecase
} // namespace tulipp

#endif // CONFIG_HPP
