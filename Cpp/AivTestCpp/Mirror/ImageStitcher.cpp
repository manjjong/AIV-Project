#include "ImageStitcher.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include "Utils/ImageHelper.h"

using namespace std;
using namespace cv;

ImageStitcher::ImageStitcher(const vector<Mat>& images_, const vector<Mat>& grayImages_) :
	images(images_), grayImages(grayImages_)
{
}

void ImageStitcher::resizeWithAspectRatio(const Mat& image0, const Mat& image1, const vector<int>& candidate0, const vector<int>& candidate1, Mat& outImage, float& outRatio)
{
    int diff0 = candidate0.back() - candidate0.front();
    int diff1 = candidate1.back() - candidate1.front();
    outRatio = static_cast<double>(diff0) / diff1;

    resize(image1, outImage, Size(image1.cols, static_cast<int>(image1.rows * outRatio)), 0, 0, INTER_LINEAR);

}

// 경계 후보 추출
vector<int> ImageStitcher::extractCandidates(const Mat& gray_image, const String& side)
{
    vector<int> candidates;
    Mat diff;

    if (side == "left") 
    {
        diff = ImageHelper::Diff(gray_image.col(0));
    }
    else if (side == "right") 
    {
        diff = ImageHelper::Diff(gray_image.col(gray_image.cols - 1));
    }

    for (int i = 0; i < diff.rows; ++i) 
    {
        if (abs(diff.at<int>(i, 0)) > 25) 
        {
            candidates.push_back(i);
        }
    }
    return candidates;
}

// 템플릿 매칭을 수행하여 가장 일치하는 위치를 반환
int ImageStitcher::findStitchPoint(const cv::Mat& image0, const cv::Mat& image1)
{
    int image0Width = image0.cols;
    int templateWidth = static_cast<int>(image0Width * 0.2f);
    
    vector<double> values(templateWidth + 1, 0.0);
    
    parallel_for_(Range(1, templateWidth + 1), [&](const Range& range)
    {
        for (int i = range.start; i < range.end; ++i)
        {
            // image0의 마지막 i 열과 image1의 첫 i 열 추출
            Mat img0 = image0.colRange(image0Width - i, image0Width);
            Mat img1 = image1.colRange(0, i);
            
            // 템플릿 매칭 수행
            Mat result;
            matchTemplate(img0, img1, result, TM_CCOEFF_NORMED);

            double minVal, maxVal;
            Point minLoc, maxLoc;

            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
            values[i] = maxVal;
        }
    });

    int maxIndex = max_element(values.begin(), values.end()) - values.begin();

    cout << "인덱스 : " << maxIndex << ", 값 : " << values[maxIndex] << endl;
    return maxIndex;
}


void ImageStitcher::prepareStitchData(vector<Mat>& outNewImages, vector<int>& outWidthGapList, vector<int>& outHeightDiffList)
{
    int vecSize = images.size();
    vector<float> ratioList(vecSize, 1);

    outNewImages.clear();
    outWidthGapList.clear();
    outHeightDiffList.clear();
    outNewImages.resize(vecSize);
    outWidthGapList.resize(vecSize);
    outHeightDiffList.resize(vecSize);

	images[0].copyTo(outNewImages[0]);

	for (int i = 0; i < vecSize - 1; ++i)
	{
        vector<int> candidate0 = extractCandidates(grayImages[i], "right");
        vector<int> candidate1 = extractCandidates(grayImages[i + 1], "left");

        // imshow("gray0", grayImages[i]);
        // imshow("gray1", grayImages[i + 1]);
        // waitKey();

        Mat newImage;
        float ratio;
        
        // 이전 비율 적용
        float prevRatio = ratioList[i];
        candidate0[0] = static_cast<int>(candidate0[0] * prevRatio);
        candidate0[candidate0.size() - 1] = static_cast<int>(candidate0[candidate0.size() - 1] * prevRatio);

        // 이미지 비율 적용
        resizeWithAspectRatio(images[i], images[i + 1], candidate0, candidate1, newImage, ratio);
        int newCandidate1 = static_cast<int>(candidate1[0] * ratio);

        // 폭 및 높이 차이 계산
        outNewImages[i + 1] = newImage;
        ratioList[i + 1] = ratio;
        outWidthGapList[i + 1] = findStitchPoint(grayImages[i], grayImages[i + 1]);
        outHeightDiffList[i + 1] = candidate0[0] - newCandidate1;
	}
}


void ImageStitcher::calculateStitchDimensions(const std::vector<int>& widthGapList, int& outWidth, int& outHeight, std::vector<int>& outWidthList)
{
    outWidthList.resize(widthGapList.size());

    for (int i = 0; i < widthGapList.size(); ++i)
    {
        outWidthList[i] = images[i].cols - widthGapList[i];
    }

    outWidth = std::accumulate(outWidthList.begin(), outWidthList.end(), 0);
    outHeight = images[0].rows;
}

cv::Mat ImageStitcher::createStitchedImage(const std::vector<cv::Mat>& newImages, const std::vector<int>& widthGapList, const std::vector<int>& heightDiffList)
{
    int width, height;
    vector<int> widthList;

    // 큰 도화지 생성 및 초기화
    calculateStitchDimensions(widthGapList, width, height, widthList);

    Mat stitchedImage(height, width, images[0].type());

    // 첫 번째 이미지 복사
    Rect roi(0, 0, images[0].cols, images[0].rows);
    images[0].copyTo(stitchedImage(roi));

    int prevWidth = images[0].cols;
    int prevHeightDiff = 0;

    for (int i = 1; i < images.size(); ++i)
    {
        int heightDiff = heightDiffList[i] + prevHeightDiff;        // 높이 차이
        int widthGap = widthGapList[i];                             // 넓이 차이
        int currWidth = prevWidth + widthList[i];                   // 현재 너비
        const Mat& image = newImages[i];                            // 현재 이미지

        // 배경 색 설정
        Rect bgRect(prevWidth, 0, widthList[i], height);
        stitchedImage(bgRect).setTo(image.at<Vec3b>(0, 0));

        if (heightDiff >= 0)
        {
            // 높이 차이가 양수일 경우
            Rect srcRect(widthGap, 0, image.cols - widthGap, height - heightDiff);
            Rect dstRect(prevWidth, heightDiff, srcRect.width, srcRect.height);
            image(srcRect).copyTo(stitchedImage(dstRect));
        }
        else
        {
            // 높이 차이가 음수일 경우
            Rect src_rect(widthGap, -heightDiff, image.cols - widthGap, height + heightDiff);
            Rect dst_rect(prevWidth, 0, src_rect.width, src_rect.height);
            image(src_rect).copyTo(stitchedImage(dst_rect));
        }

        // 다음 루프를 위한 너비와 높이 차이 업데이트
        prevWidth = currWidth;
        prevHeightDiff = heightDiff;
    }
    return stitchedImage;
}


Mat ImageStitcher::Run()
{
    vector<Mat> newImages;
    vector<int> widthGapList;
    vector<int> heightDiffList;

    // 스티칭 데이터 준비
    prepareStitchData(newImages, widthGapList, heightDiffList);

    // 스티칭 이미지 생성
    Mat stitchedImage = createStitchedImage(newImages, widthGapList, heightDiffList);

    return stitchedImage;
}
