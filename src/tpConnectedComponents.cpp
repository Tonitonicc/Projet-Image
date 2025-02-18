#include "tpConnectedComponents.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <map>
#include <stack>
using namespace cv;
using namespace std;


/**
    Performs a labeling of image connected component with 4 connectivity
    with a depth-first exploration.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccLabel(cv::Mat image)
{
    std::cout << "Starting ccLabel..." << std::endl;
    Mat binary;
    if(image.depth() == CV_32F) {
        std::cout << "Converting float image to binary..." << std::endl;
        image = image * 255;
        image.convertTo(binary, CV_8UC1);
    } else {
        std::cout << "Copying image to binary..." << std::endl;
        image.copyTo(binary);
    }
    std::cout << "Thresholding image..." << std::endl;
    threshold(binary, binary, 127, 255, THRESH_BINARY);
    
    std::cout << "Creating result matrix..." << std::endl;
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1);
    int currentLabel = 1;
    
    std::cout << "Starting image scan..." << std::endl;
    try {
        for(int i = 0; i < binary.rows; i++) {
            for(int j = 0; j < binary.cols; j++) {
                if(binary.at<uchar>(i, j) != 0 && res.at<int>(i, j) == 0) {
                    std::cout << "Found new component at (" << i << "," << j << ") with label " << currentLabel << std::endl;
                    
                    std::stack<std::pair<int, int>> stack;
                    stack.push(std::make_pair(i, j));
                    
                    while(!stack.empty()) {
                        int row = stack.top().first;
                        int col = stack.top().second;
                        stack.pop();
                        
                        if (row < 0 || row >= binary.rows || col < 0 || col >= binary.cols) 
                            continue;
                        if (binary.at<uchar>(row, col) == 0 || res.at<int>(row, col) != 0) 
                            continue;
                        

                        res.at<int>(row, col) = currentLabel;
                        

                        stack.push(std::make_pair(row-1, col)); 
                        stack.push(std::make_pair(row+1, col)); 
                        stack.push(std::make_pair(row, col-1)); 
                        stack.push(std::make_pair(row, col+1)); 
                    }
                    
                    currentLabel++;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during image scan: " << e.what() << std::endl;
        throw;
    }
    
    std::cout << "Converting to float..." << std::endl;
    Mat normalized;
    res.convertTo(normalized, CV_32FC1);
    normalize(normalized, normalized, 0, 1, NORM_MINMAX);
    
    std::cout << "ccLabel completed." << std::endl;
    return normalized;
}

/**
    Deletes the connected components (4 connectivity) containg less than size pixels.
*/
Mat ccAreaFilter(Mat image, int size)
{
    if (image.empty()) {
        cerr << "Erreur : Impossible de charger l'image d'entrée !" << endl;
        return Mat();
    }

    Mat binary;
    if (image.depth() == CV_32F) {
        image = image * 255; 
        image.convertTo(binary, CV_8UC1);
    } else {
        image.copyTo(binary);
    }
    threshold(binary, binary, 127, 255, THRESH_BINARY);

    Mat labels;
    int numLabels = connectedComponents(binary, labels, 8, CV_32S);
    cout << "Nombre total de labels détectés : " << numLabels - 1 << endl;

    map<int, int> labelCount;
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            if (label > 0) {
                labelCount[label]++;
            }
        }
    }

    cout << "=== Début des tailles des composants ===" << endl;
    for (const auto &pair : labelCount) {
        cout << "Label " << pair.first << " -> Taille : " << pair.second << " pixels" << endl;
    }
    cout << "=== Fin des tailles des composants ===" << endl;

    Mat res = Mat::zeros(binary.size(), CV_8UC1);
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            if (label > 0 && labelCount[label] >= size) {
                res.at<uchar>(i, j) = 255;
            }
        }
    }

    if (countNonZero(res) == 0) {
        cerr << "⚠️ Attention : L'image finale est vide après filtrage ! Aucun composant ne dépasse le seuil." << endl;
    }

    imwrite("filtered.png", res);
    cout << "✅ Image filtrée enregistrée sous 'filtered.png'" << endl;

    imshow("Filtered Image", res);
    waitKey(0);
    destroyAllWindows();

    return res;
}



/**
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccTwoPassLabel(cv::Mat image)
{
    std::cout << "Starting ccTwoPassLabel..." << std::endl;

    Mat binary;
    if(image.depth() == CV_32F) {
        image = image * 255;
        image.convertTo(binary, CV_8UC1);
    } else {
        image.copyTo(binary);
    }
    threshold(binary, binary, 127, 255, THRESH_BINARY);

    Mat labels = Mat::zeros(binary.size(), CV_32SC1);
    int currentLabel = 1;
    std::map<int, int> parent;

    auto findRoot = [&](int label) {
        while (parent[label] != label) {
            parent[label] = parent[parent[label]];
            label = parent[label];
        }
        return label;
    };

    std::cout << "First pass..." << std::endl;
    for (int i = 0; i < binary.rows; i++) {
        for (int j = 0; j < binary.cols; j++) {
            if (binary.at<uchar>(i, j) != 0) {
                int left = (j > 0) ? labels.at<int>(i, j - 1) : 0;
                int above = (i > 0) ? labels.at<int>(i - 1, j) : 0;

                if (left == 0 && above == 0) {
                    labels.at<int>(i, j) = currentLabel;
                    parent[currentLabel] = currentLabel;
                    currentLabel++;
                } else if (left != 0 && above == 0) {
                    labels.at<int>(i, j) = findRoot(left);
                } else if (left == 0 && above != 0) {
                    labels.at<int>(i, j) = findRoot(above);
                } else { 
                    int minLabel = std::min(findRoot(left), findRoot(above));
                    int maxLabel = std::max(findRoot(left), findRoot(above));

                    labels.at<int>(i, j) = minLabel;
                    parent[maxLabel] = minLabel;
                }
            }
        }
    }

    std::cout << "Union-Find full propagation..." << std::endl;
    for (auto &p : parent) {
        p.second = findRoot(p.first);
    }

    std::cout << "Second pass..." << std::endl;
    int maxLabel = 0; 
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            if (label != 0) {
                labels.at<int>(i, j) = findRoot(label);
                maxLabel = std::max(maxLabel, labels.at<int>(i, j));
            }
        }
    }

    if (maxLabel == 0) {
        std::cerr << "Error: All labels are zero. Check input image and labeling process." << std::endl;
        return Mat::zeros(binary.size(), CV_32FC1);
    }

    std::cout << "Normalizing labels..." << std::endl;
    Mat normalized;
    labels.convertTo(normalized, CV_32FC1);
    normalized /= maxLabel;

    std::cout << "ccTwoPassLabel completed." << std::endl;
    return normalized;
}
