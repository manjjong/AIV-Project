#include "FileHelper.h"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <regex>

namespace fs = std::filesystem;

using namespace std;
using namespace cv;

// 숫자를 기준으로 정렬하는 함수
bool naturalOrderCompare(const string& a, const string& b) 
{
    regex numberRegex("(\\d+)"); // 숫자 패턴
    smatch matchA, matchB;

    // 파일 이름에서 숫자를 추출
    string fileNameA = fs::path(a).stem().string(); // 파일 이름(확장자 제외)
    string fileNameB = fs::path(b).stem().string();

    regex_search(fileNameA, matchA, numberRegex);
    regex_search(fileNameB, matchB, numberRegex);

    if (matchA.empty() || matchB.empty()) {
        // 숫자가 없으면 기본 문자열 비교
        return a < b;
    }

    // 숫자 비교
    int numA = stoi(matchA.str());
    int numB = stoi(matchB.str());

    return numA < numB;
}

std::vector<std::string> FileHelper::LoadFileNames(const std::string& folderPath)
{
    vector<string> bmp_files;

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".bmp") {
            bmp_files.push_back(entry.path().string());
        }
    }

    sort(bmp_files.begin(), bmp_files.end(), naturalOrderCompare);

    return bmp_files;
}

bool FileHelper::SaveImage(const std::string& folderPath, const std::string& fileName, const cv::Mat& image)
{
    if (!fs::exists(folderPath)) 
    {
        if (!fs::create_directory(folderPath))
        {
            return false;
        }
    }

    imwrite(folderPath + "/" + fileName, image);
    return true;
}
