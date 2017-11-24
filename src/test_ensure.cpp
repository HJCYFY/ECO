#include "eco_assert.h"
#include "feature_extraction/hog.h"
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace Feature;
using namespace std;
using namespace cv;
int main()
{
    Mat image(imread("/home/huajun/Desktop/river2.jpg",IMREAD_GRAYSCALE));
    // Mat image(imread("/home/huajun/Desktop/track_map.png",IMREAD_GRAYSCALE));
    
    // Mat image(imread("/home/huajun/Desktop/mm.jpeg",IMREAD_GRAYSCALE));


    Hog hog(HogVariantUoctti,9,6);
    // hog.set_use_bilinear_orientation_assignments(true);    
    hog.put_image(image);
}