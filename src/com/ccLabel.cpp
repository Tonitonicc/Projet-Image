
#include "../common.h"
#include "../tpConnectedComponents.h"
#include <iostream>
#include "CLI11.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    CLI::App app{"Connected Component Labelling 2 pass"};

    string inputImage = "binary.png";
    app.add_option("-I,--inputImage", inputImage, "Input image filename");

    string outputImage = "out.png";
    app.add_option("-O,--outputImage", outputImage, "Output image filename");

    bool showImages = false;
    app.add_flag("-S,--show", showImages, "Display input and output images in new windows");

    CLI11_PARSE(app, argc, argv);

    cout << "Reading image: " << inputImage << endl;
    Mat image = imreadHelper(inputImage);
    if(image.empty()) {
        cerr << "Error: Could not read image " << inputImage << endl;
        return -1;
    }
    cout << "Image read successfully. Size: " << image.size() << " Type: " << image.type() << endl;

    cout << "Applying ccLabel..." << endl;
    Mat res_image = ccLabel(image);
    cout << "ccLabel done. Result size: " << res_image.size() << " Type: " << res_image.type() << endl;

    cout << "Remapping labels..." << endl;
    Mat tmp = remap_labels(res_image);
    double min, max;
    cv::minMaxLoc(tmp, &min, &max);

    if(min < 0){
        cout << "Warning: a pixel has a label value lower than 0!" << endl;
    }
    if(max > 255){
        cout << "Warning: a pixel has a label value greater than 255!" << endl;
    }

    cout << "Converting to float..." << endl;
    Mat tmp2;
    res_image.convertTo(tmp2, CV_32FC1);
    cv::normalize(tmp2, res_image, 0.0, 1.0, NORM_MINMAX, CV_32FC1);
    
    cout << "Writing output image: " << outputImage << endl;
    imwriteHelper(res_image, outputImage);

    // maybe show result
    if (showImages) {
        cout << "Displaying images..." << endl;
        showimage(image, "Input Image");
        showimage(res_image, "Output Image");
        waitKey(0);
        destroyAllWindows();
    }

    cout << "Done!" << endl;
    return 0;
}
