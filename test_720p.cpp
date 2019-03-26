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
    string video_path = "/media/huajun/新加卷/data/dataset/";
    string groundtruth_path = "/media/huajun/新加卷/data/anno/";
    string video_name = "yuneec3.mp4";
    string gth_name = "yuneec3.txt";
	VideoCapture vidCap;
	Mat frame;
    vidCap.open(video_path+video_name);
    if (!vidCap.isOpened())  
    {
        printf("Can't open file\n");
        return 0;
    }
    vidCap >> frame;

	int x, y, width, height;
	char c;
    std::ifstream cin(groundtruth_path+gth_name);
    if(!cin.is_open())
    {
        cout<<" there is no groundtruth file "<<endl;
        return 0;
    }
    cin >> x >> c >> y >> c >> width >> c >> height;
    cin.close();

    Track::Tracker tracker(x, y, height, width, frame);
    cout<<" initialed "<<endl;
    int num_track =0;
    while(true)
    {
		vidCap >> frame;
		if (frame.empty()) 
            break;
        tracker.track(frame);
        num_track++;
    }
    cout<<" total time:"<<tracker.time<<" fps:"<<(num_track-2)/tracker.time<<endl;
    return 0;
}