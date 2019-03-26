#pragma once

#include <vector>
#include <config.hpp>
#include <type.hpp>
#include <matlab_func.hpp>
using namespace cv;
using namespace std;
namespace Track{

class SampleSpace{
public:
    SampleSpace();
    void update_sample_space_model(const Array4D<complex<float> >& cn_samplesf,const Array4D<complex<float> >& hog_samplesf,
                                const Array4D<complex<float> >& cn_train_sample,const Array4D<complex<float> >& hog_train_sample);

    int num_training_samples;
    VectorXf vec_zero;
    VectorXf prior_weights;
    MatrixXf gram_matrix;
    MatrixXf distance_matrix;
    float minimum_sample_weight;

    int merged_sample_id;
    int new_sample_id;
    Array4D<complex<float> > cn_merged_sample,hog_merged_sample;    // 用于 更新 samplesf
private:
    VectorXf find_gram_vector(const Array4D<complex<float> >& cn_samplesf,const Array4D<complex<float> >& hog_samplesf,
                            const Array4D<complex<float> >& cn_train_sample,const Array4D<complex<float> >& hog_train_sample);
    void update_distance_matrix(VectorXf& gram_vector,float sample_norm, int id1, int id2, float w1, float w2);

    void merge_samples(const Array4D<complex<float> >& cn_sample1,const Array4D<complex<float> >& hog_sample1,
                                        const Array4D<complex<float> >& cn_sample2,const Array4D<complex<float> >& hog_sample2,
                                        float w1, float w2);
};

}