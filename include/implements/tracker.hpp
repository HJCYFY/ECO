#pragma once

#include "opencv2/core.hpp"
#include "Eigen/Core"
#include "fhog.hpp"
#include "cnf.hpp"
#include "train.hpp"
#include "scale_filter.hpp"
#include "sample_space.hpp"
#include <mutex>
#include <thread>
#include <condition_variable>

using namespace Feature;
namespace Track{
class Tracker{

public:
    Tracker(float tlx,float tly,float height,float width,cv::Mat im);
    ~Tracker();
    void track(cv::Mat img);

// private:
    void init();
    void get_interp_fourier(int hh,int hw,int ch,int cw);
    void get_reg_filter();

    void extract_features(cv::Mat img, cv::Point& pos, float scale,
                         shared_ptr<Feature::Feature>& cn_features,
                         shared_ptr<Feature::Feature>& hog_features);
    cv::Mat sample_patch(cv::Mat img, cv::Point pos, cv::Size2f sample_sz, cv::Size output_sz);

    void init_model(Mat img);
    void target_localization(Mat img);
    void model_update(Mat img);
    void thread_train_filter();
    void shift_sample(Array4D<complex<float> >& cn_xlf,Array4D<complex<float> >& hog_xlf, Point2d shift_samp);
    void init_projection_matrix(shared_ptr<Feature::Feature>& cn_f,shared_ptr<Feature::Feature>& hog_f);
    Point2f optimize_scores();

    cv::Point2f m_pos;
    cv::Point m_sample_pos;

    int m_im_height;
    int m_im_width;

    cv::Size2f m_init_sz;
    cv::Size2f m_target_sz;
    cv::Size m_support_sz;
    cv::Size m_output_sz;
    cv::Size m_cn_feature_sz,m_hog_feature_sz;
    cv::Size m_cn_filter_sz,m_hog_filter_sz;

    cv::Size m_cn_pad_sz,m_hog_pad_sz;

    float m_currentScaleFactor;
    float m_minScaleFactor;
    float m_maxScaleFactor;

    Eigen::MatrixXd m_cn_kx,m_cn_ky, m_hog_kx,m_hog_ky;                // shift_sample
    Eigen::MatrixXf m_cn_cos_win, m_hog_cos_win;
    MatrixXcf m_cn_interp, m_hog_interp;

    shared_ptr<thread> train_filter_thread;
    mutex samplesf_mutex;
    mutex wait_finish_mutex;
    mutex start,finish;
    bool waitting,finished,end_thread;
    std::condition_variable start_train_filter;
    std::condition_variable finish_train_filter;

    Array4D<complex<float> > m_cn_samplesf, m_hog_samplesf;
    Array4D<complex<float> > m_cn_xlf_proj,m_hog_xlf_proj;

    shared_ptr<Train> m_train;
    shared_ptr<ScaleFilter> m_scalefilter;
    shared_ptr<SampleSpace> m_samplespace;
    shared_ptr<CNf> m_cn_extra;
    shared_ptr<fHog> m_hog_extra;

    MatrixXcf m_scores_fs_feat1; // top half 
    MatrixXcf m_scores_fs_feat2; // top half 
    MatrixXcf m_scores_fs_sum;  // top half

    int m_frame_count;
    int m_frames_since_last_train;

    double time;
    bool m_train_thread;
};
}