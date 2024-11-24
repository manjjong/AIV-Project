#include "FileHelper.h"
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <regex>

namespace fs = std::filesystem;

using namespace std;
using namespace cv;

// ���ڸ� �������� �����ϴ� �Լ�
bool naturalOrderCompare(const string& a, const string& b) 
{
    regex numberRegex("(\\d+)"); // ���� ����
    smatch matchA, matchB;

    // ���� �̸����� ���ڸ� ����
    string fileNameA = fs::path(a).stem().string(); // ���� �̸�(Ȯ���� ����)
    string fileNameB = fs::path(b).stem().string();

    regex_search(fileNameA, matchA, numberRegex);
    regex_search(fileNameB, matchB, numberRegex);

    if (matchA.empty() || matchB.empty()) {
        // ���ڰ� ������ �⺻ ���ڿ� ��
        return a < b;
    }

    // ���� ��
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
