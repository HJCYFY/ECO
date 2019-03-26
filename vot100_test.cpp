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
    config::ConfigParser("./resource/config.cfg");
    string folder = "/home/huajun/WorkSpace/file/OTB100/dataset/TB-100/"+ config::file_name;
    char img1[100];
    int start = 1;
    sprintf(img1,"/img/%04d.jpg",start);
    string img1_name = folder + img1;
    std::ifstream cin1(img1_name);
    while(!cin1.is_open())
    {
        ++start;
        sprintf(img1,"/img/%04d.jpg",start);
        img1_name = folder + img1;
        cin1 = std::ifstream(img1_name);
        if(start>300)
        {
            cout<<img1_name<<endl;
            exit(0);
        }
    }
    
    string ground_truth = folder+ "/groundtruth_rect.2.txt";
    float x,y,h,w;
    std::string line;
    std::ifstream cin(ground_truth);
    assert(cin.is_open());
    int num_track = 0;
    while(getline(cin,line))
    {
        if(num_track == 0)
        {
            cout<<line<<endl;
            std::size_t found = line.find(",");
            while (found!=std::string::npos)
            {
                line.replace ( found,1," ");
                found = line.find(",");
            }
            std::istringstream read_data(line);
            while(read_data>>x)
            {
                read_data>>y;
                read_data>>w;
                read_data>>h;
            }
        }
        num_track++;
    }
    cout<<" num_track "<<num_track<<endl;
    num_track += start;
    cout<<" num_track "<<num_track<<endl;
    cout<<" start "<<start<<endl;
    Mat im(imread(img1_name,IMREAD_COLOR));
    Track::Tracker tracker(x, y, h, w, im);
    cout<<" initialed "<<endl;

    string ofs_name = "/home/huajun/Desktop/OTB100/" + config::file_name +".2.txt";
    ofstream ofs(ofs_name,ios::trunc);
    ofs<<x<<","<<y<<","<<w<<","<<h<<endl;
    char name[100];
    for(int i =start+1;i<num_track;++i)
    {
        sprintf(name,"/img/%04d.jpg",i);
        string pic_name = folder+name;
        cout<<pic_name<<endl;
        im = imread(pic_name,IMREAD_COLOR);
        tracker.track(im); 
        h = tracker.m_target_sz.height * tracker.m_currentScaleFactor;
        w = tracker.m_target_sz.width * tracker.m_currentScaleFactor;  
        x = tracker.m_pos.x - w/2;
        y = tracker.m_pos.y - h/2;
        ofs<<x<<","<<y<<","<<w<<","<<h<<endl;
    }
    ofs.close();
    cout<<" total time:"<<tracker.time<<" fps:"<<(num_track-start-1)/tracker.time<<endl;
    return 0;
}