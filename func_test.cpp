#include <iostream>
#include <fstream>
#include <memory>
#include "Eigen/Core"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "fhog.hpp"
#include "cnf.hpp"
#include "tracker.hpp"
#include "sample_space.hpp"
#include "scale_filter.hpp"
using namespace std;

int main()
{
    // debug targetlocalized
    string folder = "/home/huajun/WorkSpace/ECO/sequences/vot2016/girl/";
    string img1_name = folder + "00000001.jpg";
    string ground_truth = folder+ "groundtruth.txt";

    float x,y,x_min = 999,x_max=0,y_min=999,y_max=0;
    std::string line;
    std::ifstream cin(ground_truth);
    assert(cin.is_open());
    getline(cin,line);
    cout<<line<<endl;

    std::size_t found = line.find(",");
    while (found!=std::string::npos)
    {
        line.replace ( found,1," ");
        found = line.find(",");
    }
    std::cout << "line " << line << '\n';

    std::istringstream read_data(line);
    while(read_data>>x)
    {
        read_data>>y;
        if(x<x_min)
            x_min = x;
        if(x>x_max)
            x_max = x;
        if(y<y_min)
            y_min = y;
        if(y>y_max)
            y_max = y;
    }
    float height = y_max - y_min;
    float width = x_max - x_min;
    cout<<" x_min "<<x_min<<" x_max "<<x_max<<endl;
    cout<<" y_min "<<y_min<<" y_max "<<y_max<<endl;
    cout<<" height "<<height<<" width "<<width<<endl;


    Mat im(imread(img1_name,IMREAD_UNCHANGED));
    Track::Tracker tracker(x_min, y_min, height, width, im);
    cout<<" initialed "<<endl;
    char name[100];
    int num_track = 1500;
    for(int i =2;i<(num_track-2);++i)
    {
        sprintf(name,"%08d.jpg",i);
        string pic_name = folder+name;
        im = imread(pic_name,IMREAD_UNCHANGED);
        tracker.track(im);
    }
    cout<<" total time:"<<tracker.time<<" fps:"<<(num_track-2)/tracker.time<<endl;
    return 0;
}