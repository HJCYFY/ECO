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
    
    Mat im(imread("/home/huajun/Test/ECO/sequences/TB101/BlurCar1/img/0001.jpg",IMREAD_UNCHANGED));
    Track::Tracker tracker(250, 168, 106, 105, im);
    char name1[100];
    char name2[100];
    char name3[100];
    char name4[100];
    char name5[100];

    tracker.m_train->m_cn_projection_matrix.resize(10,3);
    tracker.m_train->m_hog_projection_matrix.resize(31,10);

    Matlab::read_mat("/home/huajun/Documents/hj1/cn_pm",tracker.m_train->m_cn_projection_matrix.rows(),tracker.m_train->m_cn_projection_matrix.cols(),tracker.m_train->m_cn_projection_matrix.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_pm",tracker.m_train->m_hog_projection_matrix.rows(),tracker.m_train->m_hog_projection_matrix.cols(),tracker.m_train->m_hog_projection_matrix.data());
    cout<<"SSS"<<endl;
    for(int i=2;i<500;++i)
    {
        sprintf(name1,"/home/huajun/Test/ECO/sequences/TB101/BlurCar1/img/%04d.jpg",i);
        sprintf(name4,"/home/huajun/Documents/hj1/cn_hf%d",i);
        sprintf(name5,"/home/huajun/Documents/hj1/hog_hf%d",i);
        im = imread(name1,IMREAD_UNCHANGED);
        Matlab::read_mat(name4,tracker.m_train->m_cn_hf1);
        Matlab::read_mat(name5,tracker.m_train->m_hog_hf1);  
        tracker.target_localization(im);
        tracker.m_scalefilter->Update(im, tracker.m_pos, tracker.m_target_sz, tracker.m_currentScaleFactor);
        Size target_sz = tracker.m_target_sz * tracker.m_currentScaleFactor;
        cv::Rect rect_position_vis(tracker.m_pos.x - target_sz.width/2,tracker.m_pos.y-target_sz.height/2,target_sz.width,target_sz.height);
        MatrixXcf extend = MatrixXcf::Zero(409/2+1,409);

        extend.topLeftCorner(tracker.m_scores_fs_sum.rows(),tracker.m_scores_fs_sum.cols()/2+1) = tracker.m_scores_fs_sum.leftCols(tracker.m_scores_fs_sum.cols()/2+1);
        extend.topRightCorner(tracker.m_scores_fs_sum.rows(),tracker.m_scores_fs_sum.cols() - tracker.m_scores_fs_sum.cols()/2-1) = 
                                                            tracker.m_scores_fs_sum.rightCols(tracker.m_scores_fs_sum.cols()-tracker.m_scores_fs_sum.cols()/2-1);
        MatrixXf score;
        Matlab::ifft2(extend,score,409,409);
        MatrixXf score_shift(409,409);
        score_shift.topLeftCorner(204,204) = score.bottomRightCorner(204,204);
        score_shift.topRightCorner(204,205) = score.bottomLeftCorner(204,205);
        score_shift.bottomLeftCorner(205,204) = score.topRightCorner(205,204);
        score_shift.bottomRightCorner(205,205) = score.topLeftCorner(205,205);
        float min = score_shift.minCoeff();
        float max = score_shift.maxCoeff();
        MatrixXi i_im = ((score_shift.array() - min) * 255 /(max-min)).cast<int>();

        Mat weight;
        eigen2cv(i_im,weight);
        weight.convertTo(weight, CV_8UC1);
        Mat im_color;
        applyColorMap(weight, im_color, COLORMAP_HSV);
        Size resp_sz;
        resp_sz.height = round(tracker.m_support_sz.height*tracker.m_currentScaleFactor);
        resp_sz.width = round(tracker.m_support_sz.width*tracker.m_currentScaleFactor);

        resize(im_color,im_color,resp_sz);
        int left=0,right=0,top=0,bottom=0;
        if( tracker.m_sample_pos.x-resp_sz.width/2 < 0)
        {
            left = resp_sz.width/2 - tracker.m_sample_pos.x;
        }
        if(tracker.m_sample_pos.x-resp_sz.width/2 + resp_sz.width > im.cols -1)
        {
            right = (tracker.m_sample_pos.x-resp_sz.width/2 + resp_sz.width) - im.cols + 1;
        }
        if( tracker.m_sample_pos.y-resp_sz.height/2 < 0)
        {
            top = resp_sz.height/2 - tracker.m_sample_pos.y;
        }
        if(tracker.m_sample_pos.y-resp_sz.height/2 + resp_sz.height > im.rows -1)
        {
            bottom = (tracker.m_sample_pos.y-resp_sz.height/2 + resp_sz.height)-im.rows +1;
        }

        Mat ROI1 = im(Rect(tracker.m_sample_pos.x-resp_sz.width/2 + left ,tracker.m_sample_pos.y-resp_sz.height/2 + top,resp_sz.width - left - right,resp_sz.height - top - bottom));//(im,rect_position_vis);
        Mat ROI2 = im_color(Rect(left,top,resp_sz.width - left - right,resp_sz.height - top - bottom));
        addWeighted(ROI1, 0.5, ROI2, 0.5, 0.0, ROI1);
        rectangle(im, rect_position_vis, Scalar(0,255,0), 1);
        imshow("ret",im);
        waitKey(100);
    }
}