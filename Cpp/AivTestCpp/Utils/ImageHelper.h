#pragma once
#include <opencv2/core/mat.hpp>
#include <vector>
#include <string>

class ImageHelper
{
public:
	static std::vector<cv::Mat> LoadImages(const std::vector<std::string>& fileNames);
	static std::vector<cv::Mat> ConvertToGrayImages(const std::vector<cv::Mat>& images);

	static cv::Mat Diff(const cv::Mat& image);
};

