#include <iostream>
#include "Utils/FileHelper.h"
#include "Utils/ImageHelper.h"
#include "Mirror/ImageStitcher.h"

using namespace std;
using namespace cv;


void DoImageStitch(const vector<Mat>& images, const vector<Mat>& grayImages)
{
	ImageStitcher stitcher(images, grayImages);
	Mat result = stitcher.Run();

	FileHelper::SaveImage("Result", "result.bmp", result);
}

int main(int argc, char** argv)
{
	if (argc < 3) {
		cerr << "Usage: " << argv[0] << "<image_folder_path>" << endl;
		return -1;
	}

	int choice = stoi(argv[1]);
	string folderPath = argv[2];

	vector<string> fileNames = FileHelper::LoadFileNames(folderPath);

	if (fileNames.size() == 0)
	{
		cerr << "Error: Folder does not exist: " << folderPath << endl;
		return -1;
	}

	vector<Mat> images = ImageHelper::LoadImages(fileNames);
	vector<Mat> grayImages = ImageHelper::ConvertToGrayImages(images);

	switch (choice)
	{
	case 2:
		DoImageStitch(images, grayImages);
		break;
	}
}