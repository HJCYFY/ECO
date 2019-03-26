#pragma once

#include "config.hpp"
#include "type.hpp"
#include "matlab_func.hpp"
#include <mutex>
namespace Track{


struct Train_State{
    int flag;
    float rho;
    Array4D<complex<float> > cn_p;
    Array4D<complex<float> > hog_p;
    Array4D<complex<float> > cn_r_prev;
    Array4D<complex<float> > hog_r_prev;
};


class Train{
public:
    Train(int cn_hf_height,int cn_hf_width,int cn_hf_dim,int hog_hf_height,int hog_hf_width,int hog_hf_dim);
    ~Train(){};

    void train_joint(Array4D<complex<float> >& cn_xlf,Array4D<complex<float> >& hog_xlf);
    void train_filter(Array4D<complex<float> >& cn_samplesf, Array4D<complex<float> >& hog_samplesf, VectorXf &sample_weights);

    void pcg_ccot_joint(const Array4D<complex<float> >& cn_init_samplef_proj,    const Array4D<complex<float> >& hog_init_samplef_proj, 
                        const Array4D<complex<float> >& cn_init_samplef,            const Array4D<complex<float> >& hog_init_samplef,
                        const MatrixXcf& cn_init_samplef_H,                         const MatrixXcf& hog_init_samplef_H,
                        const Array4D<complex<float> >& cn_init_hf,                 const Array4D<complex<float> >& hog_init_hf,

                        const Array4D<float >& cn_diag_M1,                          const Array4D<float >& hog_diag_M1,
                        const MatrixXf& cn_diag_M2,                                 const MatrixXf& hog_diag_M2, 

                        const Array4D<complex<float> >& cn_rhs_samplef1,            const Array4D<complex<float> >& hog_rhs_samplef1,
                        const MatrixXf& cn_rhs_samplef2,                            const MatrixXf& hog_rhs_samplef2, 
                        MatrixXf& cn_hf2,                                           MatrixXf& hog_hf2);

    void pcg_ccot(const Array4D<complex<float> >& cn_samplesf,    const Array4D<complex<float> >& hog_samplesf, 
                        const VectorXf& sample_weights, 
                        const Array4D<float >& cn_diag_M,                const Array4D<float >& hog_diag_M,
                        const Array4D<complex<float> >& cn_rhs_samplef,  const Array4D<complex<float> >& hog_rhs_samplef);
    
    void lhs_operation(Array4D<complex<float> >& cn_hf, Array4D<complex<float> >& hog_hf,
                        const Array4D<complex<float> >& cn_samplesf, const Array4D<complex<float> >& hog_samplesf, const VectorXf& sample_weights,
                        Array4D<complex<float> >& cn_hf_out, Array4D<complex<float> >& hog_hf_out);

    void lhs_operation_joint(const Array4D<complex<float> >& cn_hf_in1,          const Array4D<complex<float> >& hog_hf_in1,
                        const MatrixXf& cn_hf_in2,                              const MatrixXf& hog_hf_in2,
                        const Array4D<complex<float> >& cn_init_samplef_proj,   const Array4D<complex<float> >& hog_init_samplef_proj, 
                        const Array4D<complex<float> >& cn_init_samplef,        const Array4D<complex<float> >& hog_init_samplef,
                        const MatrixXcf& cn_init_samplef_H,                     const MatrixXcf& hog_init_samplef_H,
                        const Array4D<complex<float> >& cn_init_hf,             const Array4D<complex<float> >& hog_init_hf,
                        Array4D<complex<float> >& cn_hf_out1,                   Array4D<complex<float> >& hog_hf_out1,
                        MatrixXf& cn_hf_out2,                                   MatrixXf& hog_hf_out2
                        );

    void diag_precond(const Array4D<complex<float> >& _cn_hf1,          const Array4D<complex<float> >& _hog_hf1, 
                        const MatrixXf& _cn_hf2,                        const MatrixXf& _hog_hf2,
                        const Array4D<float >& cn_diag_M1,              const Array4D<float >& hog_diag_M1,
                        const MatrixXf& cn_diag_M2,                     const MatrixXf& hog_diag_M2,
                        Array4D<complex<float> >& _cn_hf_out1,          Array4D<complex<float> >& _hog_hf_out1,
                        MatrixXf& _cn_hf_out2,                          MatrixXf& _hog_hf_out2);

    float inner_product_joint(const Array4D<complex<float> >& A1,    const Array4D<complex<float> >& A2, 
                        MatrixXf& m1,                             MatrixXf& m2,
                        const Array4D<complex<float> >& A3,             const Array4D<complex<float> >& A4, 
                        MatrixXf& m3,                             MatrixXf& m4);

    float inner_product_filter(const Array4D<complex<float> >& A1,    const Array4D<complex<float> >& A2, 
                        const Array4D<complex<float> >& A3,             const Array4D<complex<float> >& A4);

    Array4D<complex<float> > m_cn_hf1,m_hog_hf1;
    std::mutex hf_mutex;
    Array4D<complex<float> > m_cn_hf,m_hog_hf;

    MatrixXf m_cn_yf,m_hog_yf;
    MatrixXf m_cn_projection_matrix, m_hog_projection_matrix;
    MatrixXf m_cn_proj_energy, m_hog_proj_energy;
    Array4D<float > m_cn_new_sample_energy, m_hog_new_sample_energy;
    Array4D<float > m_cn_sample_energy, m_hog_sample_energy;
    MatrixXf m_reg_filter;
    float m_reg_energy;

    CG_OPTS m_CG_opts;
    Train_State state;
};
}