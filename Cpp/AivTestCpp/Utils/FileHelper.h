#pragma once
#include <vector>
#include <string>
#include <opencv2/core/mat.hpp>

class FileHelper
{
public:
	static std::vector<std::string> LoadFileNames(const std::string& folderPath);
	static bool SaveImage(const std::string& folderPath, const std::string& fileName, const cv::Mat& image);
};