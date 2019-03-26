#include "train.hpp"

namespace Track{

Train::Train(int cn_hf_height,int cn_hf_width,int cn_hf_dim,int hog_hf_height,int hog_hf_width,int hog_hf_dim)
{
    complex<float> zero = complex<float>(0.f,0.f);
    m_cn_hf1.resize(cn_hf_height,cn_hf_width,cn_hf_dim,1,zero);
    m_hog_hf1.resize(hog_hf_height,hog_hf_width,hog_hf_dim,1,zero);

    m_CG_opts.tol = 1e-6;
    m_CG_opts.debug = config::debug;

    if(config::CG_forgetting_rate == INFINITY || config::learning_rate >= 1)
        m_CG_opts.init_forget_factor = 0;
    else
        m_CG_opts.init_forget_factor = pow(1-config::learning_rate,config::CG_forgetting_rate);
}

void Train::train_filter(Array4D<complex<float> >& cn_samplesf, Array4D<complex<float> >& hog_samplesf, VectorXf &sample_weights)
{    
    // 将 samplesf 改为正常序 这里 可以 简化
    Array4D<complex<float> > cn_rhs_samplef(cn_samplesf.dim1,cn_samplesf.dim2,cn_samplesf.dim3,1);    
    cn_rhs_samplef.getVector() = Map<MatrixXcf>(cn_samplesf.data,cn_samplesf.stride12*cn_samplesf.dim3,cn_samplesf.dim4) * sample_weights;

    Array4D<complex<float> > hog_rhs_samplef(hog_samplesf.dim1,hog_samplesf.dim2,hog_samplesf.dim3,1);    
    hog_rhs_samplef.getVector() = Map<MatrixXcf>(hog_samplesf.data,hog_samplesf.stride12*hog_samplesf.dim3,hog_samplesf.dim4) * sample_weights;
    
    for(int i=0;i<cn_rhs_samplef.dim3;++i)
        cn_rhs_samplef.getMatrix12(i) = cn_rhs_samplef.getMatrix12(i).conjugate().array() * m_cn_yf.array();
    for(int i=0;i<hog_rhs_samplef.dim3;++i)
        hog_rhs_samplef.getMatrix12(i) = hog_rhs_samplef.getMatrix12(i).conjugate().array() * m_hog_yf.array();

    Array4D<float > cn_diag_M(m_cn_sample_energy.dim1,m_cn_sample_energy.dim2,m_cn_sample_energy.dim3,1);
    Array4D<float > hog_diag_M(m_hog_sample_energy.dim1,m_hog_sample_energy.dim2,m_hog_sample_energy.dim3,1);

    MatrixXf cn_se_mean = m_cn_sample_energy.getMatrix12(0);
    for(int i=1;i<m_cn_sample_energy.dim3;++i)
        cn_se_mean += m_cn_sample_energy.getMatrix12(i);
    cn_se_mean = cn_se_mean / m_cn_sample_energy.dim3 * (1-config::precond_data_param);

    MatrixXf hog_se_mean = m_hog_sample_energy.getMatrix12(0);
    for(int i=1;i<m_hog_sample_energy.dim3;++i)
        hog_se_mean += m_hog_sample_energy.getMatrix12(i);
    hog_se_mean = hog_se_mean / m_hog_sample_energy.dim3 * (1-config::precond_data_param);

    for(int i=0; i<cn_diag_M.dim3; ++i)
        Map<MatrixXf>(cn_diag_M.data+i*cn_diag_M.stride12,cn_diag_M.dim1,cn_diag_M.dim2) = (1-config::precond_reg_param) * 
                                    (Map<MatrixXf>(m_cn_sample_energy.data+i*m_cn_sample_energy.stride12,m_cn_sample_energy.dim1,m_cn_sample_energy.dim2) * config::precond_data_param +
                                    cn_se_mean).array() + config::precond_reg_param*m_reg_energy;
    for(int i=0; i<hog_diag_M.dim3; ++i)
        Map<MatrixXf>(hog_diag_M.data+i*hog_diag_M.stride12,hog_diag_M.dim1,hog_diag_M.dim2) = (1-config::precond_reg_param) * 
                                    (Map<MatrixXf>(m_hog_sample_energy.data+i*m_hog_sample_energy.stride12,m_hog_sample_energy.dim1,m_hog_sample_energy.dim2) * config::precond_data_param +
                                    hog_se_mean).array() + config::precond_reg_param*m_reg_energy;
    pcg_ccot(cn_samplesf, hog_samplesf, sample_weights, cn_diag_M, hog_diag_M, cn_rhs_samplef, hog_rhs_samplef);
}

/***************************************************
 * 参与变量 hf projection_matrix xlf yf reg_filter sample_energy reg_energy proj_energy
 * 改变变量 hf projection_matrix
 * *************************************************/
void Train::train_joint(Array4D<complex<float> >& cn_init_samplef,Array4D<complex<float> >& hog_init_samplef)
{    
    MatrixXcf cn_init_samplef_H,hog_init_samplef_H;
    cn_init_samplef_H = Map<MatrixXcf>(cn_init_samplef.data,cn_init_samplef.stride12,cn_init_samplef.stride34).transpose().conjugate();
    hog_init_samplef_H = Map<MatrixXcf>(hog_init_samplef.data,hog_init_samplef.stride12,hog_init_samplef.stride34).transpose().conjugate();
 
    Array4D<float > cn_diag_M1(m_cn_sample_energy.dim1, m_cn_sample_energy.dim2, m_cn_sample_energy.dim3, 1);
    Array4D<float > hog_diag_M1(m_hog_sample_energy.dim1, m_hog_sample_energy.dim2, m_hog_sample_energy.dim3, 1);

    MatrixXf mean_cn_se = Map<MatrixXf>(m_cn_sample_energy.data,m_cn_sample_energy.dim1,m_cn_sample_energy.dim2);
    for(int i=1;i<m_cn_sample_energy.dim3;++i)
    {
        mean_cn_se += Map<MatrixXf>(m_cn_sample_energy.data+i*m_cn_sample_energy.stride12,m_cn_sample_energy.dim1,m_cn_sample_energy.dim2);
    }    
    mean_cn_se = mean_cn_se /m_cn_sample_energy.dim3 * (1-config::precond_data_param);

    MatrixXf mean_hog_se = Map<MatrixXf>(m_hog_sample_energy.data,m_hog_sample_energy.dim1,m_hog_sample_energy.dim2);
    for(int i=1;i<m_hog_sample_energy.dim3;++i)
    {
        mean_hog_se += Map<MatrixXf>(m_hog_sample_energy.data+i*m_hog_sample_energy.stride12,m_hog_sample_energy.dim1,m_hog_sample_energy.dim2);
    }
    mean_hog_se = mean_hog_se /m_hog_sample_energy.dim3 * (1-config::precond_data_param);

    for(int i=0; i<cn_diag_M1.dim3; ++i)
        Map<MatrixXf>(cn_diag_M1.data+i*cn_diag_M1.stride12,cn_diag_M1.dim1,cn_diag_M1.dim2) = (1-config::precond_reg_param) * 
                                    (Map<MatrixXf>(m_cn_sample_energy.data+i*m_cn_sample_energy.stride12,m_cn_sample_energy.dim1,m_cn_sample_energy.dim2) * config::precond_data_param +
                                    mean_cn_se).array() + config::precond_reg_param*m_reg_energy;
    for(int i=0; i<hog_diag_M1.dim3; ++i)
        Map<MatrixXf>(hog_diag_M1.data+i*hog_diag_M1.stride12,hog_diag_M1.dim1,hog_diag_M1.dim2) = (1-config::precond_reg_param) * 
                                    (Map<MatrixXf>(m_hog_sample_energy.data+i*m_hog_sample_energy.stride12,m_hog_sample_energy.dim1,m_hog_sample_energy.dim2) * config::precond_data_param +
                                    mean_hog_se).array() + config::precond_reg_param*m_reg_energy;

    MatrixXf cn_diag_M2 = (m_cn_proj_energy.array() + config::projection_reg) * config::precond_proj_param;;
    MatrixXf hog_diag_M2 =  (m_hog_proj_energy.array() + config::projection_reg) * config::precond_proj_param;

    Array4D<complex<float> > cn_fyf(m_cn_hf1.dim1,m_cn_hf1.dim2,m_cn_hf1.dim3,m_cn_hf1.dim4);
    Array4D<complex<float> > hog_fyf(m_hog_hf1.dim1,m_hog_hf1.dim2,m_hog_hf1.dim3,m_hog_hf1.dim4);

    Array4D<complex<float> > cn_init_samplef_proj(cn_init_samplef.dim1,cn_init_samplef.dim2,m_cn_projection_matrix.cols(),1);
    Array4D<complex<float> > hog_init_samplef_proj(hog_init_samplef.dim1,hog_init_samplef.dim2,m_hog_projection_matrix.cols(),1);

    Array4D<complex<float> > cn_rhs_samplef1(cn_init_samplef.dim1,cn_init_samplef.dim2,m_cn_projection_matrix.cols(),1);
    Array4D<complex<float> > hog_rhs_samplef1(hog_init_samplef.dim1,hog_init_samplef.dim2,m_hog_projection_matrix.cols(),1);
    MatrixXf cn_hf2,hog_hf2;
    Array4D<complex<float> > cn_init_hf,hog_init_hf;
    for(int iter=0;iter<config::init_GN_iter;++iter) //
    {
        Map<MatrixXcf>(cn_init_samplef_proj.data,cn_init_samplef_proj.stride12,cn_init_samplef_proj.dim3) = 
            Map<MatrixXcf>(cn_init_samplef.data,cn_init_samplef.stride12,cn_init_samplef.dim3) * m_cn_projection_matrix;
        Map<MatrixXcf>(hog_init_samplef_proj.data,hog_init_samplef_proj.stride12,hog_init_samplef_proj.dim3) = 
            Map<MatrixXcf>(hog_init_samplef.data,hog_init_samplef.stride12,hog_init_samplef.dim3) * m_hog_projection_matrix; 

        cn_init_hf = m_cn_hf1;
        hog_init_hf = m_hog_hf1;
        for(int i=0;i<cn_rhs_samplef1.dim3;++i)
            Map<MatrixXcf>(cn_rhs_samplef1.data+i*cn_rhs_samplef1.stride12,cn_rhs_samplef1.dim1,cn_rhs_samplef1.dim2) = 
                                        Map<MatrixXcf>(cn_init_samplef_proj.data+i*cn_init_samplef_proj.stride12,cn_init_samplef_proj.dim1,cn_init_samplef_proj.dim2).conjugate().array() * m_cn_yf.array();
        for(int i=0;i<hog_rhs_samplef1.dim3;++i)
            Map<MatrixXcf>(hog_rhs_samplef1.data+i*hog_rhs_samplef1.stride12,hog_rhs_samplef1.dim1,hog_rhs_samplef1.dim2) = 
                                        Map<MatrixXcf>(hog_init_samplef_proj.data+i*hog_init_samplef_proj.stride12,hog_init_samplef_proj.dim1,hog_init_samplef_proj.dim2).conjugate().array() * m_hog_yf.array();

        for(int i=0;i<cn_fyf.dim3;++i)
            cn_fyf.getMatrix12(i) = m_cn_hf1.getMatrix12(i).conjugate().array() * m_cn_yf.array();
        for(int i=0;i<hog_fyf.dim3;++i)
            hog_fyf.getMatrix12(i) = m_hog_hf1.getMatrix12(i).conjugate().array() * m_hog_yf.array();
    
        MatrixXf cn_rhs_samplef2 = 2*(cn_init_samplef_H * Map<MatrixXcf>(cn_fyf.data,cn_fyf.stride12,cn_fyf.dim3) - 
                                            cn_init_samplef_H.rightCols(m_cn_hf1.dim1)*Map<MatrixXcf>(cn_fyf.data,cn_fyf.stride12,cn_fyf.dim3).bottomRows(m_cn_hf1.dim1)).real() - 
                                            config::projection_reg * m_cn_projection_matrix;
        MatrixXf hog_rhs_samplef2 = 2*(hog_init_samplef_H * Map<MatrixXcf>(hog_fyf.data,hog_fyf.stride12,hog_fyf.dim3) - 
                                            hog_init_samplef_H.rightCols(m_hog_hf1.dim1)*Map<MatrixXcf>(hog_fyf.data,hog_fyf.stride12,hog_fyf.dim3).bottomRows(m_hog_hf1.dim1)).real() - 
                                            config::projection_reg * m_hog_projection_matrix;
        // Initialize the projection matrix increment to zero
        cn_hf2 = MatrixXf::Zero(m_cn_projection_matrix.rows(),m_cn_projection_matrix.cols());
        hog_hf2 = MatrixXf::Zero(m_hog_projection_matrix.rows(),m_hog_projection_matrix.cols());

        pcg_ccot_joint( cn_init_samplef_proj, hog_init_samplef_proj,  cn_init_samplef, hog_init_samplef,
                        cn_init_samplef_H, hog_init_samplef_H, cn_init_hf, hog_init_hf,
                        cn_diag_M1, hog_diag_M1, cn_diag_M2, hog_diag_M2, cn_rhs_samplef1, hog_rhs_samplef1,
                        cn_rhs_samplef2, hog_rhs_samplef2, cn_hf2, hog_hf2);

        for(int i=0;i<m_cn_hf1.dim3;++i)
            m_cn_hf1.getMatrix12(i).bottomRows(m_cn_hf1.dim1/2).rightCols(1) = 
                    m_cn_hf1.getMatrix12(i).topRows(m_cn_hf1.dim1/2).rightCols(1).colwise().reverse().conjugate().eval();
        for(int i=0;i<m_hog_hf1.dim3;++i)
            m_hog_hf1.getMatrix12(i).bottomRows(m_hog_hf1.dim1/2).rightCols(1) = 
                    m_hog_hf1.getMatrix12(i).topRows(m_hog_hf1.dim1/2).rightCols(1).colwise().reverse().conjugate().eval();

        m_cn_projection_matrix += cn_hf2;
        m_hog_projection_matrix += hog_hf2;
    }
}

void Train::pcg_ccot_joint(
                        // used for lhs_operation_joint
                        const Array4D<complex<float> >& cn_init_samplef_proj,    const Array4D<complex<float> >& hog_init_samplef_proj, 
                        const Array4D<complex<float> >& cn_init_samplef,            const Array4D<complex<float> >& hog_init_samplef,
                        const MatrixXcf& cn_init_samplef_H,                         const MatrixXcf& hog_init_samplef_H,
                        const Array4D<complex<float> >& cn_init_hf,                 const Array4D<complex<float> >& hog_init_hf,
                        // used for diag_precond
                        const Array4D<float >& cn_diag_M1,                          const Array4D<float >& hog_diag_M1,
                        const MatrixXf& cn_diag_M2,                                 const MatrixXf& hog_diag_M2, 
                        // 
                        const Array4D<complex<float> >& cn_rhs_samplef1,            const Array4D<complex<float> >& hog_rhs_samplef1,
                        const MatrixXf& cn_rhs_samplef2,                            const MatrixXf& hog_rhs_samplef2, 
                        MatrixXf& cn_hf2,                                           MatrixXf& hog_hf2)
{
    state.flag = 1;
    float rho1,rho=1;
    Array4D<complex<float> > cn_r1,hog_r1;
    MatrixXf cn_r2,hog_r2;
    lhs_operation_joint(m_cn_hf1, m_hog_hf1, cn_hf2, hog_hf2, cn_init_samplef_proj, hog_init_samplef_proj, 
                            cn_init_samplef, hog_init_samplef, cn_init_samplef_H, hog_init_samplef_H,
                            cn_init_hf, hog_init_hf,                    
                            cn_r1, hog_r1,
                            cn_r2, hog_r2);

    for(int i=0;i<cn_r1.dim3;++i)
        cn_r1.getMatrix12(i) = cn_rhs_samplef1.getMatrix12(i) - cn_r1.getMatrix12(i);

    for(int i=0;i<hog_r1.dim3;++i)
        hog_r1.getMatrix12(i) = hog_rhs_samplef1.getMatrix12(i) - hog_r1.getMatrix12(i); 

    cn_r2 = cn_rhs_samplef2-cn_r2;
    hog_r2 = hog_rhs_samplef2-hog_r2;

    Array4D<complex<float> > cn_r_prev1,hog_r_prev1;
        MatrixXf cn_r_prev2,hog_r_prev2;
    Array4D<complex<float> > cn_hf_y1,hog_hf_y1;
        MatrixXf cn_hf_y2,hog_hf_y2;
    Array4D<complex<float> > cn_hf_p1,hog_hf_p1;
        MatrixXf cn_hf_p2,hog_hf_p2;
    Array4D<complex<float> > cn_hf_q1,hog_hf_q1;
        MatrixXf cn_hf_q2,hog_hf_q2;

    for(int i=0;i<m_CG_opts.maxit;)    // m_CG_opts.maxit
    {
        diag_precond(cn_r1, hog_r1, cn_r2, hog_r2, 
                    cn_diag_M1, hog_diag_M1, cn_diag_M2, hog_diag_M2, 
                    cn_hf_y1,hog_hf_y1,cn_hf_y2,hog_hf_y2);

        rho1 = rho;
        rho = inner_product_joint(cn_r1,hog_r1,cn_r2,hog_r2,
                      cn_hf_y1,hog_hf_y1,cn_hf_y2,hog_hf_y2);
                      
        if(rho==0 || isinf(rho))
        {
            state.flag = 4;
            break;
        }
        if(i==0)
        {
            cn_hf_p1 = cn_hf_y1;
            cn_hf_p2 = cn_hf_y2;
            hog_hf_p1 = hog_hf_y1;
            hog_hf_p2 = hog_hf_y2;
        }
        else
        {
            float beta;
            if(m_CG_opts.CG_use_FR)
                beta = rho /rho1;
            else
            {
                float rho2 = inner_product_joint(cn_r_prev1,hog_r_prev1,cn_r_prev2,hog_r_prev2,
                            cn_hf_y1,hog_hf_y1,cn_hf_y2,hog_hf_y2);
                beta = (rho - rho2) / rho1;
            }
            if(beta == 0 || isinf(beta))
            {
                state.flag = 4;
                break;
            }
            beta = max(0.f, beta);
            cn_hf_p1.getVector() = cn_hf_y1.getVector() + cn_hf_p1.getVector()*beta;
            hog_hf_p1.getVector() = hog_hf_y1.getVector() + hog_hf_p1.getVector()*beta;
            cn_hf_p2 = cn_hf_y2 + cn_hf_p2 * beta;
            hog_hf_p2 = hog_hf_y2 + hog_hf_p2 * beta;
        }
        lhs_operation_joint(cn_hf_p1, hog_hf_p1, cn_hf_p2, hog_hf_p2, cn_init_samplef_proj, hog_init_samplef_proj, 
                            cn_init_samplef, hog_init_samplef, cn_init_samplef_H, hog_init_samplef_H, 
                            cn_init_hf, hog_init_hf,
                            cn_hf_q1, hog_hf_q1, cn_hf_q2, hog_hf_q2);
        float pq = inner_product_joint(cn_hf_p1,hog_hf_p1,cn_hf_p2,hog_hf_p2,cn_hf_q1,hog_hf_q1,cn_hf_q2,hog_hf_q2);
        float alpha;
        
        if(pq <= 0 || isinf(pq))
        {
            state.flag =4;
            break;
        }
        else
        {
            if(m_CG_opts.CG_standard_alpha)
            {
                alpha = rho/pq;
            }
            else
                alpha = inner_product_joint(cn_hf_p1,hog_hf_p1,cn_hf_p2,hog_hf_p2,cn_r1, hog_r1, cn_r2, hog_r2) / pq;
        }
        if(isinf(alpha))
        {
            state.flag =4;
            break;
        }
        
        if(!m_CG_opts.CG_use_FR)
        {
            cn_r_prev1 = cn_hf_y1;
            cn_r_prev2 = cn_hf_y2;
            hog_r_prev1 = hog_hf_y1;
            hog_r_prev2 = hog_hf_y2;
        }
        m_cn_hf1.getVector() = m_cn_hf1.getVector() + cn_hf_p1.getVector()*alpha;
        m_hog_hf1.getVector() = m_hog_hf1.getVector() + hog_hf_p1.getVector()*alpha;
        cn_hf2 = cn_hf2 + cn_hf_p2*alpha;
        hog_hf2 = hog_hf2 + hog_hf_p2*alpha;

        ++i;
        if(i<m_CG_opts.maxit)
        {
            cn_r1.getVector() = cn_r1.getVector() - cn_hf_q1.getVector() * alpha;
            hog_r1.getVector() = hog_r1.getVector() - hog_hf_q1.getVector() * alpha;
            cn_r2 = cn_r2 - cn_hf_q2 * alpha;
            hog_r2 = hog_r2 - hog_hf_q2 * alpha;
        }        
    }
}

void Train::pcg_ccot(
                    // used for lhs_operation
                    const Array4D<complex<float> >& cn_samplesf,    const Array4D<complex<float> >& hog_samplesf, 
                    const VectorXf& sample_weights, 
                    // used for diag_precond
                    const Array4D<float >& cn_diag_M,                          const Array4D<float >& hog_diag_M,
                    // 
                    const Array4D<complex<float> >& cn_rhs_samplef,            const Array4D<complex<float> >& hog_rhs_samplef
                    )
{
    int maxit = 5;
    Array4D<complex<float> > cn_p,hog_p;
    Array4D<complex<float> > cn_r_prev,hog_r_prev;
    float rho = 1;
    if(m_CG_opts.init_forget_factor && (state.cn_p.data != NULL))
    {
        cn_p = state.cn_p;
        hog_p = state.hog_p;
        rho = state.rho / m_CG_opts.init_forget_factor;
        cn_r_prev = state.cn_r_prev;
        hog_r_prev = state.hog_r_prev;
    }
    state.flag = 1;
    Array4D<complex<float> > cn_r,hog_r;
    m_cn_hf = m_cn_hf1;
    m_hog_hf = m_hog_hf1;
    lhs_operation(m_cn_hf, m_hog_hf, cn_samplesf, hog_samplesf, sample_weights, cn_r, hog_r);

    for(int i=0;i<cn_r.dim3;++i)
        cn_r.getMatrix12(i) = cn_rhs_samplef.getMatrix12(i) - cn_r.getMatrix12(i);
    for(int i=0;i<hog_r.dim3;++i)
        hog_r.getMatrix12(i) = hog_rhs_samplef.getMatrix12(i) - hog_r.getMatrix12(i);

    Array4D<complex<float> > cn_z(cn_r.dim1,cn_r.dim2,cn_r.dim3,cn_r.dim4);
    Array4D<complex<float> > hog_z(hog_r.dim1,hog_r.dim2,hog_r.dim3,hog_r.dim4);

    Array4D<complex<float> > cn_q,hog_q;
    for(int ii=0;ii<maxit;)
    {
        cn_z.getVector() = cn_r.getVector().array() / cn_diag_M.getVector().array();
        hog_z.getVector() = hog_r.getVector().array() / hog_diag_M.getVector().array();

        float rho1 = rho;
        rho = inner_product_filter(cn_r,hog_r,cn_z,hog_z);
        if((rho == 0) || isinf(rho))
        {
            state.flag = 4;
            break;
        }
        
        if (ii == 0 && (state.cn_p.data==NULL))
        {
            cn_p = cn_z;
            hog_p = hog_z;
        }
        else
        {
            float beta;
            if(m_CG_opts.CG_use_FR)
                beta = rho/rho1;
            else
            {
                float rho2 = inner_product_filter(cn_r_prev,hog_r_prev,cn_z,hog_z);
                beta = (rho - rho2) / rho1;
            }        
            if ((beta == 0) || isinf(beta))
            {
                state.flag = 4;
                break;
            }
            beta = max(0.f, beta);
            cn_p.getVector() = cn_z.getVector() + beta * cn_p.getVector();
            hog_p.getVector() = hog_z.getVector() + beta * hog_p.getVector();
        }
        lhs_operation(cn_p, hog_p, cn_samplesf, hog_samplesf, sample_weights, cn_q, hog_q);

        float pq = inner_product_filter(cn_p,hog_p,cn_q,hog_q);
        float alpha;
        if ((pq <= 0) || isinf(pq))
        {
            state.flag = 4;
            break;
        }
        else
        {
            if(m_CG_opts.CG_standard_alpha)
                alpha = rho / pq;
            else
                alpha = inner_product_filter(cn_p,hog_p,cn_r,hog_r) / pq;
        }
        if(isinf(alpha))
        {
            state.flag = 4;
            break;
        }
        if(!m_CG_opts.CG_use_FR)
        {
            cn_r_prev = cn_r;
            hog_r_prev = hog_r;
        }
        m_cn_hf.getVector() += alpha * cn_p.getVector();
        m_hog_hf.getVector() += alpha * hog_p.getVector();
        ++ii;
        if(ii<maxit)
        {
            cn_r.getVector() -= alpha*cn_q.getVector();
            hog_r.getVector() -= alpha*hog_q.getVector();
        }
    }


    state.cn_p = cn_p;
    state.hog_p = hog_p;
    state.rho = rho;
    if(!m_CG_opts.CG_use_FR)
    {
        state.cn_r_prev = cn_r_prev;
        state.hog_r_prev = hog_r_prev;
    }
    std::unique_lock<std::mutex> lock(hf_mutex);
    m_cn_hf1 = m_cn_hf;
    m_hog_hf1 = m_hog_hf;
}

void Train::lhs_operation(Array4D<complex<float> >& cn_hf, Array4D<complex<float> >& hog_hf,
                        const Array4D<complex<float> >& cn_samplesf, const Array4D<complex<float> >& hog_samplesf, const VectorXf& sample_weights,
                        Array4D<complex<float> >& cn_hf_out, Array4D<complex<float> >& hog_hf_out)
{
    int cn_filter_h = cn_hf.dim1;
    int cn_filter_w = cn_hf.dim2;
    int hog_filter_h = hog_hf.dim1;
    int hog_filter_w = hog_hf.dim2;

    Array4D<complex<float> > cn_sh(cn_samplesf.dim1,cn_samplesf.dim2,1,cn_samplesf.dim4);
    Array4D<complex<float> > hog_sh(hog_samplesf.dim1,hog_samplesf.dim2,1,hog_samplesf.dim4);

    for(int i=0;i<cn_sh.dim4;++i)
        Map<MatrixXcf>(cn_sh.data+i*cn_sh.stride12,cn_sh.stride12,1) = (Map<MatrixXcf>(cn_samplesf.data+i*cn_samplesf.stride12*cn_samplesf.dim3,cn_samplesf.stride12,cn_samplesf.dim3).array() * 
                Map<MatrixXcf>(cn_hf.data,cn_hf.stride12,cn_hf.dim3).array()).rowwise().sum();
    for(int i=0;i<hog_sh.dim4;++i)
        Map<MatrixXcf>(hog_sh.data+i*hog_sh.stride12,hog_sh.stride12,1) = (Map<MatrixXcf>(hog_samplesf.data+i*hog_samplesf.stride12*hog_samplesf.dim3,hog_samplesf.stride12,hog_samplesf.dim3).array() * 
                Map<MatrixXcf>(hog_hf.data,hog_hf.stride12,hog_hf.dim3).array()).rowwise().sum();

    if(cn_filter_h > hog_filter_h)
    {
        int pad_h = (cn_filter_h - hog_filter_h)/2;
        int pad_w = cn_filter_w - hog_filter_w;
        for(int i=0;i<cn_sh.dim4;++i)
            cn_sh.getMatrix12(i).middleRows(pad_h,hog_filter_h).rightCols(hog_filter_w) += hog_sh.getMatrix12(i);

        for(int i=0;i<cn_sh.dim4;++i)
            cn_sh.getMatrix12(i) *= sample_weights(i);

        cn_hf_out.resize(cn_samplesf.dim1,cn_samplesf.dim2,cn_samplesf.dim3,1,complex<float>(0,0));
        for(int i=0;i<cn_samplesf.dim3;++i)
            for(int j=0;j<cn_samplesf.dim4;++j)
                cn_hf_out.getMatrix12(i).array() += (cn_sh.getMatrix12(j).array().conjugate()*cn_samplesf.getMatrix12(j*cn_samplesf.dim3+i).array()).conjugate();

        hog_hf_out.resize(hog_samplesf.dim1,hog_samplesf.dim2,hog_samplesf.dim3,1,complex<float>(0,0));
        for(int i=0;i<hog_samplesf.dim3;++i)
            for(int j=0;j<hog_samplesf.dim4;++j)
                hog_hf_out.getMatrix12(i).array() += (cn_sh.getMatrix12(j).middleRows(pad_h,hog_filter_h).middleCols(pad_w,hog_filter_w).array().conjugate() *
                                                hog_samplesf.getMatrix12(j*hog_samplesf.dim3+i).array()).conjugate();

        {   // cn
            int reg_pad = min((int)m_reg_filter.cols()-1,cn_hf.dim2-1);
            Array4D<complex<float> > cn_hf_conv(cn_hf.dim1,cn_hf.dim2+reg_pad,cn_hf.dim3,cn_hf.dim4);
            for(int i=0;i<cn_hf_conv.dim3;++i)
            {
                cn_hf_conv.getMatrix12(i).leftCols(cn_hf.dim2) = cn_hf.getMatrix12(i);
                cn_hf_conv.getMatrix12(i).rightCols(reg_pad) = cn_hf.getMatrix12(i).middleCols(cn_hf.dim2-reg_pad-1,reg_pad).conjugate().colwise().reverse().rowwise().reverse();
            }
            Array4D<complex<float> > cn_hf_conv2;
            Matlab::conv3(cn_hf_conv, m_reg_filter, Matlab::convn_full, cn_hf_conv2);

            Array4D<complex<float> > cn_hf_conv3(cn_hf_conv2.dim1,cn_hf_conv2.dim2-reg_pad,cn_hf_conv2.dim3,cn_hf_conv2.dim4);
            for(int i=0;i<cn_hf_conv2.dim3;++i)
                cn_hf_conv3.getMatrix12(i) = cn_hf_conv2.getMatrix12(i).leftCols(cn_hf_conv2.dim2-reg_pad);
            Array4D<complex<float> > cn_hf_conv4;
            Matlab::conv3(cn_hf_conv3,m_reg_filter,Matlab::convn_valid,cn_hf_conv4);
            for(int i=0;i<cn_hf_out.dim3;++i)
                cn_hf_out.getMatrix12(i) += cn_hf_conv4.getMatrix12(i);
        }
        {   // hog
            int reg_pad = min((int)m_reg_filter.cols()-1,hog_hf.dim2-1);
            Array4D<complex<float> > hog_hf_conv(hog_hf.dim1,hog_hf.dim2+reg_pad,hog_hf.dim3,hog_hf.dim4);
            for(int i=0;i<hog_hf_conv.dim3;++i)
            {
                hog_hf_conv.getMatrix12(i).leftCols(hog_hf.dim2) = hog_hf.getMatrix12(i);
                hog_hf_conv.getMatrix12(i).rightCols(reg_pad) = hog_hf.getMatrix12(i).middleCols(hog_hf.dim2-reg_pad-1,reg_pad).conjugate().colwise().reverse().rowwise().reverse();
            }
            Array4D<complex<float> > hog_hf_conv2;
            Matlab::conv3(hog_hf_conv, m_reg_filter, Matlab::convn_full, hog_hf_conv2);

            Array4D<complex<float> > hog_hf_conv3(hog_hf_conv2.dim1,hog_hf_conv2.dim2-reg_pad,hog_hf_conv2.dim3,hog_hf_conv2.dim4);
            for(int i=0;i<hog_hf_conv2.dim3;++i)
                hog_hf_conv3.getMatrix12(i) = hog_hf_conv2.getMatrix12(i).leftCols(hog_hf_conv2.dim2-reg_pad);
            Array4D<complex<float> > hog_hf_conv4;
            Matlab::conv3(hog_hf_conv3,m_reg_filter,Matlab::convn_valid,hog_hf_conv4);
            for(int i=0;i<hog_hf_out.dim3;++i)
                hog_hf_out.getMatrix12(i) += hog_hf_conv4.getMatrix12(i);
        }
    }
    else
    {

    }
}

void Train::lhs_operation_joint(const Array4D<complex<float> >& cn_hf_in1,          const Array4D<complex<float> >& hog_hf_in1,
                            const MatrixXf& cn_hf_in2,                              const MatrixXf& hog_hf_in2,
                            const Array4D<complex<float> >& cn_init_samplef_proj,   const Array4D<complex<float> >& hog_init_samplef_proj, 
                            const Array4D<complex<float> >& cn_init_samplef,        const Array4D<complex<float> >& hog_init_samplef,
                            const MatrixXcf& cn_init_samplef_H,                     const MatrixXcf& hog_init_samplef_H,
                            const Array4D<complex<float> >& cn_init_hf,             const Array4D<complex<float> >& hog_init_hf,
                            Array4D<complex<float> >& cn_hf_out1,                   Array4D<complex<float> >& hog_hf_out1,
                            MatrixXf& cn_hf_out2,                                   MatrixXf& hog_hf_out2
                            )
{    
    int cn_filter_h = cn_hf_in1.dim1;
    int cn_filter_w = cn_hf_in1.dim2;
    int hog_filter_h = hog_hf_in1.dim1;
    int hog_filter_w = hog_hf_in1.dim2;
    Array4D<complex<float> > cn_sh,hog_sh;
    cn_sh.resize(cn_init_samplef_proj.dim1, cn_init_samplef_proj.dim2, 1,1);
    cn_sh.getVector() = (Map<MatrixXcf>(cn_init_samplef_proj.data,cn_init_samplef_proj.stride12,cn_init_samplef_proj.dim3).array() * 
            Map<MatrixXcf>(cn_hf_in1.data,cn_hf_in1.stride12,cn_hf_in1.dim3).array()).rowwise().sum();
 
    hog_sh.resize(hog_init_samplef_proj.dim1, hog_init_samplef_proj.dim2, 1,1);
    hog_sh.getVector() = (Map<MatrixXcf>(hog_init_samplef_proj.data,hog_init_samplef_proj.stride12,hog_init_samplef_proj.dim3).array() * 
            Map<MatrixXcf>(hog_hf_in1.data,hog_hf_in1.stride12,hog_hf_in1.dim3).array()).rowwise().sum();

    if(cn_filter_h>=hog_filter_h)
    {
        int hog_pad_h = (cn_filter_h - hog_filter_h)/2;
        int hog_pad_w = cn_filter_w - hog_filter_w;
        cn_sh.getMatrix12(0).middleRows(hog_pad_h, hog_filter_h).middleCols(hog_pad_w,hog_filter_w)
                += hog_sh.getMatrix12(0);

        Array4D<complex<float> > temp_cn_hf_out1,temp_hog_hf_out1;
        temp_cn_hf_out1.resize(cn_init_samplef_proj.dim1,cn_init_samplef_proj.dim2,cn_init_samplef_proj.dim3,1);
        for(int i=0;i<temp_cn_hf_out1.dim3;++i)
            temp_cn_hf_out1.getMatrix12(i) = 
                    (cn_init_samplef_proj.getMatrix12(i).array() * cn_sh.getMatrix12(0).array().conjugate()).conjugate();
            
        // 注意下面 乘以 cn_sh
        temp_hog_hf_out1.resize(hog_init_samplef_proj.dim1,hog_init_samplef_proj.dim2,hog_init_samplef_proj.dim3,1);   
        for(int i=0;i<temp_hog_hf_out1.dim3;++i)
            temp_hog_hf_out1.getMatrix12(i) = (hog_init_samplef_proj.getMatrix12(i).array() * 
                    cn_sh.getMatrix12(0).middleRows(hog_pad_h, hog_filter_h).middleCols(hog_pad_w,hog_filter_w).array().conjugate()).conjugate();

        {   // cn
            int reg_pad = min((int)m_reg_filter.cols()-1,cn_hf_in1.dim1-1);
            Array4D<complex<float> > cn_hf_conv(cn_hf_in1.dim1,cn_hf_in1.dim2+reg_pad,cn_hf_in1.dim3,1);
            for(int i=0;i<cn_hf_in1.dim3;++i)   
            {
                Map<MatrixXcf>(cn_hf_conv.data+i*cn_hf_conv.stride12,cn_hf_conv.dim1,cn_hf_conv.dim2).leftCols(cn_hf_in1.dim2) = 
                        Map<MatrixXcf>(cn_hf_in1.data+i*cn_hf_in1.stride12,cn_hf_in1.dim1,cn_hf_in1.dim2);
                Map<MatrixXcf>(cn_hf_conv.data+i*cn_hf_conv.stride12,cn_hf_conv.dim1,cn_hf_conv.dim2).rightCols(reg_pad) = 
                        Map<MatrixXcf>(cn_hf_in1.data+i*cn_hf_in1.stride12,cn_hf_in1.dim1,cn_hf_in1.dim2).middleCols(cn_hf_in1.dim2-reg_pad-1,reg_pad).colwise().reverse().rowwise().reverse().conjugate();
            }           
            Array4D<complex<float> > cn_hf_conv2;
            Matlab::conv3(cn_hf_conv,m_reg_filter,Matlab::convn_full,cn_hf_conv2);    
            Array4D<complex<float> > cn_hf_conv3(cn_hf_conv2.dim1,cn_hf_conv2.dim2-reg_pad,cn_hf_conv2.dim3,1);
            for(int i=0;i<cn_hf_conv2.dim3;++i)
                Map<MatrixXcf>(cn_hf_conv3.data+i*cn_hf_conv3.stride12,cn_hf_conv3.dim1,cn_hf_conv3.dim2) = 
                    Map<MatrixXcf>(cn_hf_conv2.data+i*cn_hf_conv2.stride12,cn_hf_conv2.dim1,cn_hf_conv2.dim2).leftCols(cn_hf_conv2.dim2-reg_pad);
            Array4D<complex<float> > temp_cn_hf_out2;
            Matlab::conv3(cn_hf_conv3,m_reg_filter,Matlab::convn_valid,temp_cn_hf_out2);       
            for(int i=0;i<temp_cn_hf_out1.dim3;++i)
                Map<MatrixXcf>(temp_cn_hf_out1.data+i*temp_cn_hf_out1.stride12,temp_cn_hf_out1.dim1,temp_cn_hf_out1.dim2) += 
                    Map<MatrixXcf>(temp_cn_hf_out2.data+i*temp_cn_hf_out2.stride12,temp_cn_hf_out2.dim1,temp_cn_hf_out2.dim2);
        }              
        {   // hog
            int reg_pad = min((int)m_reg_filter.cols()-1,hog_hf_in1.dim1-1);
            Array4D<complex<float> > hog_hf_conv(hog_hf_in1.dim1,hog_hf_in1.dim2+reg_pad,hog_hf_in1.dim3,1);
            for(int i=0;i<hog_hf_in1.dim3;++i)   
            {
                Map<MatrixXcf>(hog_hf_conv.data+i*hog_hf_conv.stride12,hog_hf_conv.dim1,hog_hf_conv.dim2).leftCols(hog_hf_in1.dim2) = 
                        Map<MatrixXcf>(hog_hf_in1.data+i*hog_hf_in1.stride12,hog_hf_in1.dim1,hog_hf_in1.dim2);
                Map<MatrixXcf>(hog_hf_conv.data+i*hog_hf_conv.stride12,hog_hf_conv.dim1,hog_hf_conv.dim2).rightCols(reg_pad) = 
                        Map<MatrixXcf>(hog_hf_in1.data+i*hog_hf_in1.stride12,hog_hf_in1.dim1,hog_hf_in1.dim2).middleCols(hog_hf_in1.dim2-reg_pad-1,reg_pad).colwise().reverse().rowwise().reverse().conjugate();
            }           
            Array4D<complex<float> > hog_hf_conv2;
            Matlab::conv3(hog_hf_conv,m_reg_filter,Matlab::convn_full,hog_hf_conv2);    
            Array4D<complex<float> > hog_hf_conv3(hog_hf_conv2.dim1,hog_hf_conv2.dim2-reg_pad,hog_hf_conv2.dim3,1);
            for(int i=0;i<hog_hf_conv2.dim3;++i)
                Map<MatrixXcf>(hog_hf_conv3.data+i*hog_hf_conv3.stride12,hog_hf_conv3.dim1,hog_hf_conv3.dim2) = 
                    Map<MatrixXcf>(hog_hf_conv2.data+i*hog_hf_conv2.stride12,hog_hf_conv2.dim1,hog_hf_conv2.dim2).leftCols(hog_hf_conv2.dim2-reg_pad);
            Array4D<complex<float> > temp_hog_hf_out2;
            Matlab::conv3(hog_hf_conv3,m_reg_filter,Matlab::convn_valid,temp_hog_hf_out2);       
            for(int i=0;i<temp_hog_hf_out1.dim3;++i)
                Map<MatrixXcf>(temp_hog_hf_out1.data+i*temp_hog_hf_out1.stride12,temp_hog_hf_out1.dim1,temp_hog_hf_out1.dim2) += 
                    Map<MatrixXcf>(temp_hog_hf_out2.data+i*temp_hog_hf_out2.stride12,temp_hog_hf_out2.dim1,temp_hog_hf_out2.dim2);
        }
        
        Array4D<complex<float> > cn_BP,hog_BP;
        cn_BP.resize(cn_init_samplef.dim1,cn_init_samplef.dim2,1,1);
        MatrixXcf a = (Map<MatrixXcf>(cn_init_samplef.data,cn_init_samplef.stride12,cn_init_samplef.dim3)*cn_hf_in2);
        cn_BP.getVector() = ((Map<MatrixXcf>(cn_init_samplef.data,cn_init_samplef.stride12,cn_init_samplef.dim3)*cn_hf_in2).array()*
                    Map<MatrixXcf>(cn_init_hf.data,cn_init_hf.stride12,cn_init_hf.dim3).array()).rowwise().sum();
        hog_BP.resize(hog_init_samplef.dim1,hog_init_samplef.dim2,1,1);
        hog_BP.getVector() = ((Map<MatrixXcf>(hog_init_samplef.data,hog_init_samplef.stride12,hog_init_samplef.dim3)*hog_hf_in2).array()*
                    Map<MatrixXcf>(hog_init_hf.data,hog_init_hf.stride12,hog_init_hf.dim3).array()).rowwise().sum();
        
        cn_BP.getMatrix12(0).middleRows(hog_pad_h,hog_filter_h).rightCols(hog_filter_w) += hog_BP.getMatrix12(0);  
        cn_hf_out1.resize(cn_init_samplef_proj.dim1,cn_init_samplef_proj.dim2,cn_init_samplef_proj.dim3,1);

        for(int i=0;i<cn_hf_out1.dim3;++i)
            cn_hf_out1.getMatrix12(i) = temp_cn_hf_out1.getMatrix12(i).array() + cn_init_samplef_proj.getMatrix12(i).conjugate().array()*cn_BP.getMatrix12(0).array();

        // Matlab::write_mat("/home/huajun/Documents/hj2/cn_hf_out1",cn_hf_out1);
        // Matlab::write_mat("/home/huajun/Documents/hj2/cn_BP",cn_BP); 

        Array4D<complex<float> > cn_fBP(cn_init_hf.dim1,cn_init_hf.dim2,cn_init_hf.dim3,1);
        for(int i=0;i<cn_fBP.dim3;++i)
            cn_fBP.getMatrix12(i) = cn_init_hf.getMatrix12(i).array().conjugate() * cn_BP.getMatrix12(0).array();
             
        Array4D<complex<float> > cn_shBP(cn_init_hf.dim1,cn_init_hf.dim2,cn_init_hf.dim3,1);
        for(int i=0;i<cn_shBP.dim3;++i)
            cn_shBP.getMatrix12(i) = cn_init_hf.getMatrix12(i).array().conjugate() * cn_sh.getMatrix12(0).array();
       
        hog_hf_out1.resize(hog_init_samplef_proj.dim1,hog_init_samplef_proj.dim2,hog_init_samplef_proj.dim3,1);
        for(int i=0;i<hog_hf_out1.dim3;++i)
            hog_hf_out1.getMatrix12(i) = temp_hog_hf_out1.getMatrix12(i).array() + 
                                         hog_init_samplef_proj.getMatrix12(i).array().conjugate()* 
                                         cn_BP.getMatrix12(0).middleRows(hog_pad_h,hog_filter_h).rightCols(hog_filter_w).array();
                        
        Array4D<complex<float> > hog_fBP(hog_init_hf.dim1,hog_init_hf.dim2,hog_init_hf.dim3,1);
        for(int i=0;i<hog_init_hf.dim3;++i)
            hog_fBP.getMatrix12(i) = hog_init_hf.getMatrix12(i).array().conjugate() * cn_BP.getMatrix12(0).middleRows(hog_pad_h,hog_filter_h).rightCols(hog_filter_w).array();
       
        Array4D<complex<float> > hog_shBP(hog_init_hf.dim1,hog_init_hf.dim2,hog_init_hf.dim3,hog_init_hf.dim4);
        for(int i=0;i<hog_init_hf.dim3;++i)
            hog_shBP.getMatrix12(i) = hog_init_hf.getMatrix12(i).array().conjugate() * cn_sh.getMatrix12(0).middleRows(hog_pad_h,hog_filter_h).rightCols(hog_filter_w).array();
        
        // cn
        MatrixXf cn_hf_out3 = 2*(cn_init_samplef_H * Map<MatrixXcf>(cn_fBP.data,cn_fBP.stride12,cn_fBP.dim3)
                                    - cn_init_samplef_H.rightCols(m_cn_hf1.dim1) * Map<MatrixXcf>(cn_fBP.data,cn_fBP.stride12,cn_fBP.dim3).bottomRows(m_cn_hf1.dim1)
                                    ).real()
                                    + config::projection_reg * cn_hf_in2;
                                    
        MatrixXf cn_hf_out4 = 2*(cn_init_samplef_H * Map<MatrixXcf>(cn_shBP.data,cn_shBP.stride12,cn_shBP.dim3)
                                    - cn_init_samplef_H.rightCols(m_cn_hf1.dim1) * Map<MatrixXcf>(cn_shBP.data,cn_shBP.stride12,cn_shBP.dim3).bottomRows(m_cn_hf1.dim1)).real();
        
        cn_hf_out2 = cn_hf_out3 + cn_hf_out4;

        // hog
        MatrixXf hog_hf_out3 = 2*(hog_init_samplef_H * Map<MatrixXcf>(hog_fBP.data,hog_fBP.stride12,hog_fBP.dim3)
                                    - hog_init_samplef_H.rightCols(m_hog_hf1.dim1) * Map<MatrixXcf>(hog_fBP.data,hog_fBP.stride12,hog_fBP.dim3).bottomRows(m_hog_hf1.dim1)
                                    ).real()
                                    + config::projection_reg * hog_hf_in2;
                                    
        MatrixXf hog_hf_out4 = 2*(hog_init_samplef_H * Map<MatrixXcf>(hog_shBP.data,hog_shBP.stride12,hog_shBP.dim3)
                                    - hog_init_samplef_H.rightCols(m_hog_hf1.dim1) * Map<MatrixXcf>(hog_shBP.data,hog_shBP.stride12,hog_shBP.dim3).bottomRows(m_hog_hf1.dim1)).real();
        
        hog_hf_out2 = hog_hf_out3 + hog_hf_out4;
    }
    else
    {
        printf(" wait to do \n");
        exit(0);
    }
}

void Train::diag_precond(const Array4D<complex<float> >& _cn_hf1,      const Array4D<complex<float> >& _hog_hf1, 
                        const MatrixXf& _cn_hf2,                        const MatrixXf& _hog_hf2,
                        const Array4D<float >& cn_diag_M1,              const Array4D<float >& hog_diag_M1,
                        const MatrixXf& cn_diag_M2,                     const MatrixXf& hog_diag_M2,
                        Array4D<complex<float> >& _cn_hf_out1,          Array4D<complex<float> >& _hog_hf_out1,
                        MatrixXf& _cn_hf_out2,                          MatrixXf& _hog_hf_out2)
{
    _cn_hf_out1.resize(_cn_hf1.dim1,_cn_hf1.dim2,_cn_hf1.dim3,1);
    _hog_hf_out1.resize(_hog_hf1.dim1,_hog_hf1.dim2,_hog_hf1.dim3,1);
    
    _cn_hf_out1.getVector() = _cn_hf1.getVector().array() / cn_diag_M1.getVector().array();
    _hog_hf_out1.getVector() = _hog_hf1.getVector().array() / hog_diag_M1.getVector().array();

    _cn_hf_out2.array() = _cn_hf2.array() / cn_diag_M2.array();
    _hog_hf_out2.array() = _hog_hf2.array() / hog_diag_M2.array();
}

float Train::inner_product_joint(const Array4D<complex<float> >& A1,    const Array4D<complex<float> >& A2, 
                        MatrixXf& m1,                             MatrixXf& m2,
                        const Array4D<complex<float> >& A3,             const Array4D<complex<float> >& A4, 
                        MatrixXf& m3,                             MatrixXf& m4)
{    
    complex<float> ip;
    complex<float> two(2.f,0);
    ip = (A1.getVector_T().conjugate() * A3.getVector())(0,0)*two;
    for(int i=0;i<A1.dim3;++i)
    {
        ip -= (A1.getMatrix12(i).rightCols(1).transpose().conjugate() * A3.getMatrix12(i).rightCols(1))(0,0);
    }
    ip += (Map<MatrixXf>(m1.data(),1,m1.size()).conjugate() * Map<MatrixXf>(m3.data(),m3.size(),1))(0,0);
    ip += (A2.getVector_T().conjugate() * A4.getVector())(0,0)*two;
    for(int i=0;i<A2.dim3;++i)
    {
        ip -= (A2.getMatrix12(i).rightCols(1).transpose().conjugate() * A4.getMatrix12(i).rightCols(1))(0,0);
    }
    ip += (Map<MatrixXf>(m2.data(),1,m2.size()).conjugate() * Map<MatrixXf>(m4.data(),m4.size(),1))(0,0);
    return ip.real();
}

float Train::inner_product_filter(const Array4D<complex<float> >& A1,    const Array4D<complex<float> >& A2, 
                    const Array4D<complex<float> >& A3,             const Array4D<complex<float> >& A4)
{
    complex<float> ip;
    complex<float> two(2.f,0);
    ip = (A1.getVector_T().conjugate() * A3.getVector())(0,0)*two;
    for(int i=0;i<A1.dim3;++i)
    {
        ip -= (A1.getMatrix12(i).rightCols(1).transpose().conjugate() * A3.getMatrix12(i).rightCols(1))(0,0);
    }
    ip += (A2.getVector_T().conjugate() * A4.getVector())(0,0)*two;
    for(int i=0;i<A2.dim3;++i)
    {
        ip -= (A2.getMatrix12(i).rightCols(1).transpose().conjugate() * A4.getMatrix12(i).rightCols(1))(0,0);
    }
    return ip.real();
}
}