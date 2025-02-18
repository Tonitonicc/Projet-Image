#include "tpHistogram.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

/**
    Inverse a grayscale image with float values.
    for all pixel p: res(p) = 1.0 - image(p)
*/
Mat inverse(Mat image)
{
    // clone original image
    Mat res = image.clone();
      /********************************************
                YOUR CODE HERE
    *********************************************/
    cv::subtract(1.0f, image, res);
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Thresholds a grayscale image with float values.
    for all pixel p: res(p) =
        | 0 if image(p) <= lowT
        | image(p) if lowT < image(p) <= hightT
        | 1 otherwise
*/
Mat threshold(Mat image, float lowT, float highT)
{
    Mat res = image.clone();
    assert(lowT <= highT);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    cv::threshold(image, res, highT, 1.0f, cv::THRESH_BINARY_INV);
    cv::threshold(res, res, lowT, 0.0f, cv::THRESH_BINARY);

    
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Quantize the input float image in [0,1] in numberOfLevels different gray levels.
    
    eg. for numberOfLevels = 3 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/3
        | 1/2 if 1/3 <= image(p) < 2/3
        | 1 otherwise

        for numberOfLevels = 4 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/4
        | 1/3 if 1/4 <= image(p) < 1/2
        | 2/3 if 1/2 <= image(p) < 3/4
        | 1 otherwise

        and so on for other values of numberOfLevels.

*/
Mat quantize(Mat image, int numberOfLevels)
{
    Mat res = image.clone();
    assert(numberOfLevels>0);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    float step = 1.0f / numberOfLevels;
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            float value = image.at<float>(i,j);
            int level = (int)(value / step);
            if(level >= numberOfLevels) level = numberOfLevels - 1;
            res.at<float>(i,j) = level * step;
            if(level == numberOfLevels - 1) res.at<float>(i,j) = 1.0f;
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Normalize a grayscale image with float values
    Target range is [minValue, maxValue].
*/
Mat normalize(Mat image, float minValue, float maxValue)
{
    Mat res = image.clone();
    assert(minValue <= maxValue);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);
    
    if(maxVal > minVal) {
        res = (image - minVal) * ((maxValue - minValue) / (maxVal - minVal)) + minValue;
    } else {
        res = Mat::ones(image.size(), image.type()) * minValue;
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Equalize image histogram with unsigned char values ([0;255])

    Warning: this time, image values are unsigned chars but calculation will be done in float or double format.
    The final result must be rounded toward the nearest integer 
*/
Mat equalize(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
    // Calculate histogram
    int histogram[256] = {0};
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            histogram[image.at<uchar>(i,j)]++;
        }
    }
    
    // Calculate cumulative histogram
    int cumulative[256] = {0};
    cumulative[0] = histogram[0];
    for(int i = 1; i < 256; i++) {
        cumulative[i] = cumulative[i-1] + histogram[i];
    }
    
    // Calculate mapping
    float scale = 255.0f / (image.rows * image.cols);
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            int pixel = image.at<uchar>(i,j);
            float newValue = scale * cumulative[pixel];
            res.at<uchar>(i,j) = cv::saturate_cast<uchar>(round(newValue));
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute a binarization of the input float image using an automatic Otsu threshold.
    Input image is of type unsigned char ([0;255])
*/
Mat thresholdOtsu(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
    // Calculate histogram
    float histogram[256] = {0};
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            histogram[image.at<uchar>(i,j)]++;
        }
    }
    
    // Normalize histogram
    float total = image.rows * image.cols;
    for(int i = 0; i < 256; i++) {
        histogram[i] /= total;
    }
    
    float maxVariance = 0;
    int threshold = 0;
    
    for(int t = 0; t < 256; t++) {
        // Calculate weights
        float w0 = 0, w1 = 0;
        float mean0 = 0, mean1 = 0;
        
        for(int i = 0; i < 256; i++) {
            if(i <= t) {
                w0 += histogram[i];
                mean0 += i * histogram[i];
            } else {
                w1 += histogram[i];
                mean1 += i * histogram[i];
            }
        }
        
        if(w0 > 0) mean0 /= w0;
        if(w1 > 0) mean1 /= w1;
        
        // Calculate between-class variance
        float variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);
        
        if(variance > maxVariance) {
            maxVariance = variance;
            threshold = t;
        }
    }
    
    cv::threshold(image, res, threshold, 255, cv::THRESH_BINARY);
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}
