#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    Mat I, input_image;
    string path;

    while(getline(cin, path)){
	    input_image = imread(path.c_str(), 0); // Read the file as grayscale

	    input_image.convertTo(I, CV_8U);
	    imwrite( path.c_str(), I );
	    cout << path.c_str() << " " << CV_8U << I.depth() << " " << input_image.depth() << endl;
    }
}