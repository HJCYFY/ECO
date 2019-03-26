#pragma once
#include <sstream>
#include <fstream>
#include <string>
#include <memory>
#include "opencv2/core.hpp"
#include "feature.hpp"
using namespace std;
namespace Feature{

class CNf{
public:
    CNf(int _binSize);
    ~CNf();
    shared_ptr<Feature> extract(cv::Mat img);
private:
    shared_ptr<Feature> lookup_table(cv::Mat img);
    shared_ptr<Feature> integralImage(float* mat ,int height,int width);
    shared_ptr<Feature> average_feature_region(shared_ptr<Feature>& im, int height,int width);

    int binSize;
    int dims;
    int size;

    int width ;
    int height ;
    int stride ;

    int row,col;

    float* table_data;
};

}