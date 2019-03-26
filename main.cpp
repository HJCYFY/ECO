#include <iostream>
#include <fstream>
#include <memory>
#include "opencv2/opencv.hpp"
#include "fhog.hpp"
#include "cnf.hpp"
#include "tracker.hpp"
using namespace std;

using namespace cv;
using namespace Feature;
int main(int argv,char** argc)
{
    Mat im(imread("/home/huajun/Test/ECO/sequences/Crossing/img/0001.jpg",IMREAD_UNCHANGED));
    Track::Tracker a(204, 150, 50, 17, im);

    char name[100];
    for(int i =2;i<120;++i)
    {
        sprintf(name,"/home/huajun/Test/ECO/sequences/Crossing/img/%04d.jpg",i);
        im = imread(name,IMREAD_UNCHANGED);
        a.track(im);
    }
    cout<<" total time:"<<a.time<<" fps:"<<119/a.time<<endl;
    return 0;
}