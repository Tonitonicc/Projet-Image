#include "tpConnectedComponents.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <map>
using namespace cv;
using namespace std;


/**
    Performs a labeling of image connected component with 4 connectivity
    with a depth-first exploration.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    int currentLabel = 1;
    
    // Recursive function to label connected components
    std::function<void(int, int, int)> labelDFS = [&](int row, int col, int label) {
        if (row < 0 || row >= image.rows || col < 0 || col >= image.cols) return;
        if (image.at<uchar>(row, col) == 0 || res.at<int>(row, col) != 0) return;
        
        // Label current pixel
        res.at<int>(row, col) = label;
        
        // Visit 4-connected neighbors
        labelDFS(row-1, col, label); // up
        labelDFS(row+1, col, label); // down
        labelDFS(row, col-1, label); // left
        labelDFS(row, col+1, label); // right
    };
    
    // Scan the image
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            if(image.at<uchar>(i, j) != 0 && res.at<int>(i, j) == 0) {
                labelDFS(i, j, currentLabel);
                currentLabel++;
            }
        }
    }
    return res;
}

/**
    Deletes the connected components (4 connectivity) containg less than size pixels.
*/
cv::Mat ccAreaFilter(cv::Mat image, int size)
{
    Mat res = Mat::zeros(image.rows, image.cols, image.type());
    assert(size>0);
    // First, label all connected components
    Mat labels = ccLabel(image);
    
    // Count pixels for each label
    map<int, int> labelCount;
    for(int i = 0; i < labels.rows; i++) {
        for(int j = 0; j < labels.cols; j++) {
            int label = labels.at<int>(i, j);
            if(label > 0) {
                labelCount[label]++;
            }
        }
    }
    
    // Keep only components with size >= threshold
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            int label = labels.at<int>(i, j);
            if(label > 0 && labelCount[label] >= size) {
                res.at<uchar>(i, j) = image.at<uchar>(i, j);
            }
        }
    }
    return res;
}


/**
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccTwoPassLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    vector<int> labels(1, 0);  // labels[0] is unused
    map<int, vector<int>> equivalences;
    int currentLabel = 1;
    
    // First pass: assign temporary labels and record equivalences
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            if(image.at<uchar>(i, j) == 0) continue;
            
            vector<int> neighbors;
            // Check left and up neighbors
            if(j > 0 && res.at<int>(i, j-1) > 0)
                neighbors.push_back(res.at<int>(i, j-1));
            if(i > 0 && res.at<int>(i-1, j) > 0)
                neighbors.push_back(res.at<int>(i-1, j));
            
            if(neighbors.empty()) {
                // New label
                res.at<int>(i, j) = currentLabel;
                labels.push_back(currentLabel);
                currentLabel++;
            } else {
                // Use minimum neighbor label
                int minLabel = *min_element(neighbors.begin(), neighbors.end());
                res.at<int>(i, j) = minLabel;
                
                // Record equivalences
                for(int neighbor : neighbors) {
                    if(neighbor != minLabel) {
                        equivalences[minLabel].push_back(neighbor);
                        equivalences[neighbor].push_back(minLabel);
                    }
                }
            }
        }
    }
    
    // Resolve equivalences using union-find
    vector<int> parent(labels.size());
    for(int i = 0; i < parent.size(); i++) parent[i] = i;
    
    function<int(int)> find = [&](int x) {
        if(parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    };
    
    for(const auto& equiv : equivalences) {
        int label1 = equiv.first;
        for(int label2 : equiv.second) {
            int root1 = find(label1);
            int root2 = find(label2);
            if(root1 != root2)
                parent[root2] = root1;
        }
    }
    
    // Second pass: relabel using resolved equivalences
    for(int i = 0; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            if(res.at<int>(i, j) > 0) {
                res.at<int>(i, j) = find(res.at<int>(i, j));
            }
        }
    }
    return res;
}