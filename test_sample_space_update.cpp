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
    // // debug targetlocalized
    Mat im(imread("/home/huajun/Test/ECO/sequences/Crossing/img/0001.jpg",IMREAD_UNCHANGED));
    Track::Tracker tracker(204, 150, 50, 17, im);
    
    char name1[100];
    char name2[100];
    char name3[100];
    char name4[100];
    char name5[100];
    char name6[100];
    char name7[100];
    Array4D<complex<float> > cn_xlf_proj(41,21,3,1);
    Array4D<complex<float> > hog_xlf_proj(27,14,10,1);

    Matlab::read_mat("/home/huajun/Documents/hj1/cn_samplesf",tracker.m_cn_samplesf);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_samplesf",tracker.m_hog_samplesf); 
    tracker.m_samplespace->gram_matrix(0,0) = 38.4257011;
    for(int i=2;i<68;++i)
    {        
        sprintf(name1,"/home/huajun/Documents/hj1/cn_xlf_proj%d",i);
        sprintf(name2,"/home/huajun/Documents/hj1/hog_xlf_proj%d",i);

        Matlab::read_mat(name1,cn_xlf_proj);
        Matlab::read_mat(name2,hog_xlf_proj);
        tracker.m_samplespace->update_sample_space_model(tracker.m_cn_samplesf,tracker.m_hog_samplesf,cn_xlf_proj,hog_xlf_proj);

        cout<<" update "<<i<<endl;
        sprintf(name5,"/home/huajun/Documents/hj2/prior_weights%d_",i);
        sprintf(name6,"/home/huajun/Documents/hj2/gram_matrix%d_",i);
        sprintf(name7,"/home/huajun/Documents/hj2/distance_matrix%d_",i);;
        Matlab:: write_mat(name5,tracker.m_samplespace->prior_weights.rows(),tracker.m_samplespace->prior_weights.cols(),tracker.m_samplespace->prior_weights.data());
        Matlab:: write_mat(name6,tracker.m_samplespace->gram_matrix.rows(),tracker.m_samplespace->gram_matrix.cols(),tracker.m_samplespace->gram_matrix.data());
        Matlab:: write_mat(name7,tracker.m_samplespace->distance_matrix.rows(),tracker.m_samplespace->distance_matrix.cols(),tracker.m_samplespace->distance_matrix.data());
    
        if(tracker.m_samplespace->merged_sample_id >= 0)
        {
            Map<MatrixXcf>(tracker.m_cn_samplesf.data,tracker.m_cn_samplesf.stride12*tracker.m_cn_samplesf.dim3,tracker.m_cn_samplesf.dim4).middleCols(tracker.m_samplespace->merged_sample_id,1) = 
                tracker.m_samplespace->cn_merged_sample.getVector(); 
            Map<MatrixXcf>(tracker.m_hog_samplesf.data,tracker.m_hog_samplesf.stride12*tracker.m_hog_samplesf.dim3,tracker.m_hog_samplesf.dim4).middleCols(tracker.m_samplespace->merged_sample_id,1) = 
                tracker.m_samplespace->hog_merged_sample.getVector(); 
        }
        if(tracker.m_samplespace->new_sample_id >= 0)
        {
            Map<MatrixXcf>(tracker.m_cn_samplesf.data,tracker.m_cn_samplesf.stride12*tracker.m_cn_samplesf.dim3,tracker.m_cn_samplesf.dim4).middleCols(tracker.m_samplespace->new_sample_id,1) = 
                cn_xlf_proj.getVector(); 
            Map<MatrixXcf>(tracker.m_hog_samplesf.data,tracker.m_hog_samplesf.stride12*tracker.m_hog_samplesf.dim3,tracker.m_hog_samplesf.dim4).middleCols(tracker.m_samplespace->new_sample_id,1) = 
                hog_xlf_proj.getVector(); 
        }

        if( tracker.m_samplespace->merged_sample_id >= 0 || tracker.m_samplespace->new_sample_id >= 0)
        {
            sprintf(name3,"/home/huajun/Documents/hj2/cn_samplesf%d_",i);
            sprintf(name4,"/home/huajun/Documents/hj2/hog_samplesf%d_",i);
            Matlab:: write_mat(name3,tracker.m_cn_samplesf);
            Matlab:: write_mat(name4,tracker.m_hog_samplesf);
        }

    }
}