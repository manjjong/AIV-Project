#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
class ImageStitcher
{
private:
	const std::vector<cv::Mat> images;
	const std::vector<cv::Mat> grayImages;

private:
	// 스티칭 데이터 준비
	void resizeWithAspectRatio(const cv::Mat& image0, const cv::Mat& image1, const std::vector<int>& candidate0, const std::vector<int>& candidate1, cv::Mat& outImage, float& outRatio);
	std::vector<int> extractCandidates(const cv::Mat& gray_image, const cv::String& side);
	int findStitchPoint(const cv::Mat& image0, const cv::Mat& image1);

	void prepareStitchData(std::vector<cv::Mat>& outNewImages, std::vector<int>& outWidthGapList, std::vector<int>& outHeightDiffList);

private:
	// 스티칭 이미지 생성
	void calculateStitchDimensions(const std::vector<int>& widthGapList, int& outWidth, int& outHeight, std::vector<int>& outWidthList);
	cv::Mat createStitchedImage(const std::vector<cv::Mat>& newImages, const std::vector<int>& widthGapList, const std::vector<int>& heightDiffList);
public:
	ImageStitcher(const std::vector<cv::Mat>& images_, const std::vector<cv::Mat>& grayImages_);

	cv::Mat Run();
};

