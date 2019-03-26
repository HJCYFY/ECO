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
    // Mat im(imread("/home/huajun/Test/ECO/sequences/Crossing/img/0001.jpg",IMREAD_UNCHANGED));
    // Track::Tracker tracker(204, 150, 50, 17, im);

    Track::Train train(41, 21, 3, 27, 14, 10);

    train.m_CG_opts.maxit = config::init_CG_iter/config::init_GN_iter;
    train.m_CG_opts.CG_use_FR = true;
    train.m_CG_opts.CG_standard_alpha = true;
    train.m_reg_filter.resize(3,7);
    train.m_cn_projection_matrix.resize(10,3);
    train.m_hog_projection_matrix.resize(31,10);
    train.m_cn_yf.resize(41,21);
    train.m_hog_yf.resize(27,14);
    train.m_cn_proj_energy.resize(10,3);
    train.m_hog_proj_energy.resize(31,10);
    Array4D<complex<float> > cn_xlf(41, 21, 10, 1);
    Array4D<complex<float> > hog_xlf(27, 14, 31, 1);
    train.m_cn_sample_energy.resize(41, 21, 3, 1);
    train.m_hog_sample_energy.resize(27, 14, 10, 1);
    train.m_reg_energy = 0.0763157;
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_pm",train.m_cn_projection_matrix.rows(),train.m_cn_projection_matrix.cols(),train.m_cn_projection_matrix.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_pm",train.m_hog_projection_matrix.rows(),train.m_hog_projection_matrix.cols(),train.m_hog_projection_matrix.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_xlf",cn_xlf);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_xlf",hog_xlf);
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_yf",train.m_cn_yf.rows(),train.m_cn_yf.cols(),train.m_cn_yf.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_yf",train.m_hog_yf.rows(),train.m_hog_yf.cols(),train.m_hog_yf.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/reg_filter",train.m_reg_filter.rows(),train.m_reg_filter.cols(),train.m_reg_filter.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_sample_energy",train.m_cn_sample_energy);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_sample_energy",train.m_hog_sample_energy);
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_proj_energy",train.m_cn_proj_energy.rows(),train.m_cn_proj_energy.cols(),train.m_cn_proj_energy.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_proj_energy",train.m_hog_proj_energy.rows(),train.m_hog_proj_energy.cols(),train.m_hog_proj_energy.data()); 

    train.train_joint(cn_xlf,hog_xlf);


    Matlab::write_mat("/home/huajun/Documents/hj2/cn_hf1",train.m_cn_hf1);
    Matlab::write_mat("/home/huajun/Documents/hj2/hog_hf1",train.m_hog_hf1); 
    Matlab::write_mat("/home/huajun/Documents/hj2/cn_pm",train.m_cn_projection_matrix.rows(),train.m_cn_projection_matrix.cols(),train.m_cn_projection_matrix.data());
    Matlab::write_mat("/home/huajun/Documents/hj2/hog_pm",train.m_hog_projection_matrix.rows(),train.m_hog_projection_matrix.cols(),train.m_hog_projection_matrix.data());
   



    // Array4D<complex<float> > cn_hf_in1(41, 21, 3, 1);
    // Array4D<complex<float> > hog_hf_in1(27, 14, 10, 1);
    // Array4D<complex<float> > cn_init_samplef_proj(41, 21, 3, 1);
    // Array4D<complex<float> > hog_init_samplef_proj(27, 14, 10, 1);
    // Array4D<complex<float> > cn_init_samplef(41, 21, 10, 1);
    // Array4D<complex<float> > hog_init_samplef(27, 14, 31, 1);
    // Array4D<complex<float> > cn_init_hf(41, 21, 3, 1);
    // Array4D<complex<float> > hog_init_hf(27, 14, 10, 1);
    // train.m_reg_filter.resize(3,7);
    // MatrixXcf cn_init_samplef_H(10,861);
    // MatrixXcf hog_init_samplef_H(31,378);
    // MatrixXf cn_hf2(10,3);
    // MatrixXf hog_hf2(31,10);

    // Array4D<float> cn_diag_M1(41, 21, 3, 1);
    // Array4D<float> hog_diag_M1(27, 14, 10, 1);
    // MatrixXf cn_diag_M2(10,3);
    // MatrixXf hog_diag_M2(31,10);

    // Array4D<complex<float> > cn_rhs_samplef1(41, 21, 3, 1);
    // Array4D<complex<float> > hog_rhs_samplef1(27, 14, 10, 1);
    // MatrixXf cn_rhs_samplef2(10,3);
    // MatrixXf hog_rhs_samplef2(31,10);


    // Array4D<complex<float> > cn_hf_out1,hog_hf_out1;
    // MatrixXf cn_hf_out2,hog_hf_out2;

    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_hf1",train.m_cn_hf1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_hf1",train.m_hog_hf1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_hf2",cn_hf2.rows(),cn_hf2.cols(),cn_hf2.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_hf2",hog_hf2.rows(),hog_hf2.cols(),hog_hf2.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_dM1",cn_diag_M1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_dM1",hog_diag_M1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_dM2",cn_diag_M2.rows(),cn_diag_M2.cols(),cn_diag_M2.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_dM2",hog_diag_M2.rows(),hog_diag_M2.cols(),hog_diag_M2.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_rhs_samplef1",cn_rhs_samplef1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_rhs_samplef1",hog_rhs_samplef1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_rhs_samplef2",cn_rhs_samplef2.rows(),cn_rhs_samplef2.cols(),cn_rhs_samplef2.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_rhs_samplef2",hog_rhs_samplef2.rows(),hog_rhs_samplef2.cols(),hog_rhs_samplef2.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_init_samplef",cn_init_samplef);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_init_samplef",hog_init_samplef);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_init_samplef_proj",cn_init_samplef_proj);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_init_samplef_proj",hog_init_samplef_proj);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_XH",cn_init_samplef_H.rows(),cn_init_samplef_H.cols(),cn_init_samplef_H.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_XH",hog_init_samplef_H.rows(),hog_init_samplef_H.cols(),hog_init_samplef_H.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/reg_filter",train.m_reg_filter.rows(),train.m_reg_filter.cols(),train.m_reg_filter.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_init_hf",cn_init_hf);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_init_hf",hog_init_hf);  

    // train.m_CG_opts.maxit = config::init_CG_iter/config::init_GN_iter;
    // train.m_CG_opts.CG_use_FR = true;
    // train.m_CG_opts.CG_standard_alpha = true;

    // train.pcg_ccot_joint( cn_init_samplef_proj, hog_init_samplef_proj, cn_init_samplef, hog_init_samplef,
    //                     cn_init_samplef_H, hog_init_samplef_H, cn_init_hf, hog_init_hf,
    //                     cn_diag_M1, hog_diag_M1, cn_diag_M2, hog_diag_M2, 
    //                     cn_rhs_samplef1, hog_rhs_samplef1, cn_rhs_samplef2, hog_rhs_samplef2, 
    //                     cn_hf2, hog_hf2);  
    
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_hf1",train.m_cn_hf1);
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_hf1",train.m_hog_hf1);
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_hf2",cn_hf2.rows(),cn_hf2.cols(),cn_hf2.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_hf2",hog_hf2.rows(),hog_hf2.cols(),hog_hf2.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_dM1",cn_diag_M1);
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_dM1",hog_diag_M1);
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_dM2",cn_diag_M2.rows(),cn_diag_M2.cols(),cn_diag_M2.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_dM2",hog_diag_M2.rows(),hog_diag_M2.cols(),hog_diag_M2.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_rhs_samplef1",cn_rhs_samplef1);
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_rhs_samplef1",hog_rhs_samplef1);
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_rhs_samplef2",cn_rhs_samplef2.rows(),cn_rhs_samplef2.cols(),cn_rhs_samplef2.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_rhs_samplef2",hog_rhs_samplef2.rows(),hog_rhs_samplef2.cols(),hog_rhs_samplef2.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_init_samplef",cn_init_samplef);
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_init_samplef",hog_init_samplef);
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_init_samplef_proj",cn_init_samplef_proj);
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_init_samplef_proj",hog_init_samplef_proj);
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_XH",cn_init_samplef_H.rows(),cn_init_samplef_H.cols(),cn_init_samplef_H.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_XH",hog_init_samplef_H.rows(),hog_init_samplef_H.cols(),hog_init_samplef_H.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/reg_filter",train.m_reg_filter.rows(),train.m_reg_filter.cols(),train.m_reg_filter.data());
    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_init_hf",cn_init_hf);
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_init_hf",hog_init_hf);  




Track::Train train(41, 21, 3, 27, 14, 10);

    train.m_CG_opts.CG_use_FR = config::CG_use_FR;
    train.m_CG_opts.CG_standard_alpha = config::CG_standard_alpha;
    train.m_CG_opts.maxit = config::CG_iter;
    train.m_reg_filter.resize(3,7);
    train.state.cn_p.resize(41,21,3,1);
    train.state.hog_p.resize(27,14,10,1);
    train.state.cn_r_prev.resize(41,21,3,1);
    train.state.hog_r_prev.resize(27,14,10,1);
    train.m_cn_sample_energy.resize(41,21,3,1);
    train.m_hog_sample_energy.resize(27,14,10,1);
    train.m_reg_energy = 0.0763157;
    train.m_cn_yf.resize(41,21);
    train.m_hog_yf.resize(27,14);
    Array4D<complex<float> > cn_samplesf(41,21,3,30);
    Array4D<complex<float> > hog_samplesf(27,14,10,30);
    VectorXf sample_weights(30);

    Array4D<float > cn_diag_M(41,21,3,1);
    Array4D<float > hog_diag_M(27,14,10,1);
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_hf",train.m_cn_hf1);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_hf",train.m_hog_hf1);
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_samplesf",cn_samplesf);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_samplesf",hog_samplesf);
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_yf",train.m_cn_yf.rows(),train.m_cn_yf.cols(),train.m_cn_yf.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_yf",train.m_hog_yf.rows(),train.m_hog_yf.cols(),train.m_hog_yf.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/reg_filter",train.m_reg_filter.rows(),train.m_reg_filter.cols(),train.m_reg_filter.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/sample_weights",sample_weights.rows(),sample_weights.cols(),sample_weights.data());
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_sample_energy",train.m_cn_sample_energy);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_sample_energy",train.m_hog_sample_energy);
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_p",train.state.cn_p);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_p",train.state.hog_p);
    Matlab::read_mat("/home/huajun/Documents/hj1/cn_r",train.state.cn_r_prev);
    Matlab::read_mat("/home/huajun/Documents/hj1/hog_r",train.state.hog_r_prev);
    train.state.rho = 6.8220641e-07;

    Array4D<complex<float> > cn_hf_out,hog_hf_out;
    train.train_filter( cn_samplesf,  hog_samplesf, sample_weights);


    Matlab::write_mat("/home/huajun/Documents/hj2/cn_hf",train.m_cn_hf1);
    Matlab::write_mat("/home/huajun/Documents/hj2/hog_hf",train.m_hog_hf1);
    Matlab::write_mat("/home/huajun/Documents/hj2/cn_p",train.state.cn_p);
    Matlab::write_mat("/home/huajun/Documents/hj2/hog_p",train.state.hog_p);
    Matlab::write_mat("/home/huajun/Documents/hj2/cn_r",train.state.cn_r_prev);
    Matlab::write_mat("/home/huajun/Documents/hj2/hog_r",train.state.hog_r_prev);


    // Track::Train train(41, 21, 3, 27, 14, 10);

    // train.m_CG_opts.CG_use_FR = config::CG_use_FR;
    // train.m_CG_opts.CG_standard_alpha = config::CG_standard_alpha;
    // train.m_CG_opts.maxit = config::CG_iter;
    // train.m_reg_filter.resize(3,7);
    // train.state.cn_p.resize(41,21,3,1);
    // train.state.hog_p.resize(27,14,10,1);
    // train.state.cn_r_prev.resize(41,21,3,1);
    // train.state.hog_r_prev.resize(27,14,10,1);

    // Array4D<complex<float> > cn_hf(41,21,3,1);
    // Array4D<complex<float> > hog_hf(27,14,10,1);
    // Array4D<complex<float> > cn_samplesf(41,21,3,30);
    // Array4D<complex<float> > hog_samplesf(27,14,10,30);
    // VectorXf sample_weights(30);
    // Array4D<complex<float> > cn_rhs_samplef(41,21,3,1);
    // Array4D<complex<float> > hog_rhs_samplef(27,14,10,1);

    // Array4D<float > cn_diag_M(41,21,3,1);
    // Array4D<float > hog_diag_M(27,14,10,1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_samplesf",cn_samplesf);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_samplesf",hog_samplesf);
    // Matlab::read_mat("/home/huajun/Documents/hj1/reg_filter",train.m_reg_filter.rows(),train.m_reg_filter.cols(),train.m_reg_filter.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/sample_weights",sample_weights.rows(),sample_weights.cols(),sample_weights.data());
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_rhs_samplef",cn_rhs_samplef);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_rhs_samplef",hog_rhs_samplef);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_diag_M",cn_diag_M);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_diag_M",hog_diag_M);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_hf",train.m_cn_hf1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_hf",train.m_hog_hf1);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_p",train.state.cn_p);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_p",train.state.hog_p);
    // Matlab::read_mat("/home/huajun/Documents/hj1/cn_r",train.state.cn_r_prev);
    // Matlab::read_mat("/home/huajun/Documents/hj1/hog_r",train.state.hog_r_prev);
    // train.state.rho = 6.8220641e-07;


    // Array4D<complex<float> > cn_hf_out,hog_hf_out;
    // train.pcg_ccot( cn_samplesf,  hog_samplesf, sample_weights,  cn_diag_M, hog_diag_M,
    //                  cn_rhs_samplef, hog_rhs_samplef);


    // Matlab::write_mat("/home/huajun/Documents/hj2/cn_hf",train.m_cn_hf1);
    // Matlab::write_mat("/home/huajun/Documents/hj2/hog_hf",train.m_hog_hf1);
}