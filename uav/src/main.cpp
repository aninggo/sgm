
// STD
#include <iostream>
#include <chrono>
#include <iomanip>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Tulipp
#include "csgm.h"

// macros
#define xstr(s) _str(s)
#define _str(s) #s

 // include namepsace
using namespace tulipp::usecase::uav;

//==================================================================================================
/**
 * @brief Method to colorize depthmap for visualization.
 *
 * Colorization is performed according HSV Color Space from 0 to 240 (0 = Red, 120 = Green,
 * 240 = Blue).
 *
 * @param[in] iDepthmap Depthmap to colorize.
 * @param[out] oScaleImage Cv Mat image showing a scale.
 * @return Colorized depthmap In format CV_8UC3.
 */
cv::Mat colorizeDepthmap(const cv::Mat &iDepthmap, cv::Mat& oScaleImage)
{
  const float MAX_DEPTH_TO_DISPLAY = 50;

  cv::Mat colorizedDepthmap;
  cv::Mat naThresholdMask = cv::Mat::zeros(iDepthmap.size(),CV_8UC1);
  cv::Mat maxThresholdMask = cv::Mat::zeros(iDepthmap.size(),CV_8UC1);

  // convert values to float and copy to colorized depthmap
  iDepthmap.convertTo(colorizedDepthmap, CV_32FC1);

  //--- clamp negative values to 0 ---
  cv::threshold(iDepthmap, naThresholdMask, 0., 1., cv::THRESH_BINARY_INV);
  naThresholdMask.convertTo(naThresholdMask, CV_8UC1);

  //--- NORMALIZE DEPTHMAP ---

  //--- get min max of depth map for areas which are outside of naThreshold ---
  double minVal, maxVal;
  std::stringstream minTxt, maxTxt;
  cv::minMaxLoc(colorizedDepthmap, &minVal, &maxVal, 0, 0, cv::Mat((naThresholdMask - 1) * -1));

  // get mask of pixels that are above maxVal
  cv::threshold(iDepthmap, maxThresholdMask, MAX_DEPTH_TO_DISPLAY, 1., cv::THRESH_BINARY);
  maxThresholdMask.convertTo(maxThresholdMask, CV_8UC1);

  colorizedDepthmap -= minVal;
  colorizedDepthmap /= (MAX_DEPTH_TO_DISPLAY-minVal);

  minTxt << std::fixed << std::setprecision(2) << minVal;
  maxTxt << std::fixed << std::setprecision(2) << MAX_DEPTH_TO_DISPLAY;

  //--- create scale image ---
  oScaleImage = cv::Mat::zeros(iDepthmap.size().height, 200, CV_32FC1);
  {
    for(int y = 10; y <= iDepthmap.size().height - 10; y++)
    {
      cv::line(oScaleImage, cv::Point2i(10, y), cv::Point2i(70, y),
               (1- (float)(y-10) / (float)(iDepthmap.size().height - 10)));
    }
  }

  // color according to HSV-Color-Space. Min 0 = Red, Max 240 = Blue.
  {
    std::vector<cv::Mat> depthmapChannels;
    std::vector<cv::Mat> scaleChannels;

    depthmapChannels.push_back(cv::Mat((-colorizedDepthmap + 1) * 240.f).clone()); // H
    depthmapChannels.push_back(cv::Mat(iDepthmap.size(), CV_32FC1, 1.f));    // S
    depthmapChannels.push_back(cv::Mat(iDepthmap.size(), CV_32FC1, 1.f));    // V

    scaleChannels.push_back(cv::Mat((-oScaleImage + 1) * 240.f).clone()); // H
    scaleChannels.push_back(cv::Mat(oScaleImage.size(), CV_32FC1, 1.f));    // S
    scaleChannels.push_back(cv::Mat(oScaleImage.size(), CV_32FC1, 1.f));    // V

    // set pixels above threshold ot black
    depthmapChannels.at(2).setTo(0, naThresholdMask);
    depthmapChannels.at(2).setTo(0, maxThresholdMask);

    // merge channels
    cv::merge(depthmapChannels, colorizedDepthmap);
    cv::merge(scaleChannels, oScaleImage);

    cv::cvtColor(colorizedDepthmap, colorizedDepthmap, CV_HSV2RGB);
    colorizedDepthmap *= 255.f;
    colorizedDepthmap.convertTo(colorizedDepthmap, CV_8UC3);

    cv::cvtColor(oScaleImage, oScaleImage, CV_HSV2RGB);
    oScaleImage *= 255.f;
    oScaleImage.convertTo(oScaleImage, CV_8UC3);
  }


  //--- FINISH SCALE IAMGE ---

  //--- white background ---
  cv::rectangle(oScaleImage,cv::Point2i(0,0), cv::Point2i(oScaleImage.size().width,10),
                CV_RGB(255,255,255), -1);
  cv::rectangle(oScaleImage,cv::Point2i(0,0), cv::Point2i(10,oScaleImage.size().height),
                CV_RGB(255,255,255), -1);
  cv::rectangle(oScaleImage,cv::Point2i(0,oScaleImage.size().height - 10),
                cv::Point2i(oScaleImage.size().width,oScaleImage.size().height),
                CV_RGB(255,255,255), -1);
  cv::rectangle(oScaleImage,cv::Point2i(70,0),
                cv::Point2i(oScaleImage.size().width,oScaleImage.size().height),
                CV_RGB(255,255,255), -1);

  //--- black border ---
  cv::rectangle(oScaleImage,cv::Point2i(10,10), cv::Point2i(70,oScaleImage.size().height - 10),
                CV_RGB(0,0,0));

  //--- write labels ---
  cv::line(oScaleImage, cv::Point2i(75, 10), cv::Point2i(90, 10),
            CV_RGB(0,0,0));
  cv::line(oScaleImage, cv::Point2i(75, oScaleImage.size().height - 10),
           cv::Point2i(90, oScaleImage.size().height - 10),
           CV_RGB(0,0,0));
  cv::putText(oScaleImage, maxTxt.str(), cv::Point2i(95, 15), cv::FONT_HERSHEY_COMPLEX, 0.5,
              CV_RGB(0,0,0));
  cv::putText(oScaleImage, minTxt.str(), cv::Point2i(95, oScaleImage.size().height - 5), cv::FONT_HERSHEY_COMPLEX, 0.5,
              CV_RGB(0,0,0));

  return colorizedDepthmap;
}

//==================================================================================================
/**
 * @brief Method to print synopsis of application.
 */
void printSynopsis()
{
  std::cout << "\033[31m"
            << "=========================================================================\n"
            << "Not enough parameters!\n"
            << "Synopsis: " << xstr(TARGET_NAME) << " <frame1> <frame2> <imgScale>"
            << "\033[0m"
            << std::endl;
}

//==================================================================================================
/**
 * @brief Main entry point of application
 * @param[in] argv Call parameters: <frame1> <frame2> <outputWidth> <ouputHeight>
 */
int main(int argc, const char* argv[])
{

  //--- check if enough parameters ar given ---
  if(argc < 3)
  {
    printSynopsis();
    return 0;
  }

  //--- read images ---
  cv::Mat frame1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat frame2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

  //--- check if reading was unsuccessful ---
  if(frame1.empty() || frame2.empty())
  {
    std::cout << "\033[31m"
              << "=========================================================================\n"
              << "Error reading image files!"
              << "\033[0m"
              << std::endl;

    return -1;
  }

  //--- scale if parameters are given ---
  float imgScale = 0.8;
  if(argc == 4)
    imgScale = std::min(std::stof(argv[3]), imgScale);


  //--- covnert images to float ---
  frame1.convertTo(frame1, CV_32FC1, 1.f/255.f);
  frame2.convertTo(frame2, CV_32FC1, 1.f/255.f);

  //--- compute start time ---
  auto startTime = std::chrono::steady_clock::now();

  //--- compute disparity image ---
  CSgm sgm;
  cv::Mat depthImg = sgm.computeDepth(frame1, frame2, imgScale);

  //--- compute total runtime ---
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                 (std::chrono::steady_clock::now() - startTime);

  //--- print runtime ---
  std::cout << "\033[32m"
            << "Runtime: " << duration.count() << "ms"
            << "\033[0m" << std::endl;

  //--- save results ---
  cv::FileStorage fs("depthImg.xml",cv::FileStorage::WRITE);
  fs << "depth" << depthImg;
  fs.release();

  cv::Mat depthScale;
  cv::Mat visDepthImg = colorizeDepthmap(depthImg, depthScale);
  cv::imwrite("depthImg.png", visDepthImg);
  cv::imwrite("depthScale.png", depthScale);
//  cv::imshow("depth image", visDepthImg);
//  cv::imshow("depth scale", depthScale);
//  cv::waitKey();

  return 0;
}
