#include "ImageHelper.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<Mat> ImageHelper::LoadImages(const vector<string>& fileNames)
{
    vector<Mat> images(fileNames.size());

    parallel_for_(Range(0, images.size()), [&](const Range& range) {
        for (int i = 0; i < range.end; ++i)
        {
            images[i] = cv::imread(fileNames[i]);
        }
    });
    return images;
}

vector<Mat> ImageHelper::ConvertToGrayImages(const vector<Mat>& images)
{
    vector<Mat> grayImages(images.size());

    parallel_for_(Range(0, images.size()), [&](const Range& range) {
        for (int i = range.start; i < range.end; ++i)
        {
            cv::Mat gray;
            cv::cvtColor(images[i], gray, COLOR_BGR2GRAY);
            grayImages[i] = gray;
        }
    });
    
    return grayImages;
}

cv::Mat ImageHelper::Diff(const cv::Mat& image)
{
    int len = image.rows;
    cv::Mat output(len - 1, 1, CV_32S);

    parallel_for_(Range(0, len - 1), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; ++i)
        {
            output.at<int>(i, 0) = static_cast<int>(image.at<uchar>(i + 1, 0)) - static_cast<int>(image.at<uchar>(i, 0));
        }
    });

    return output;
}
