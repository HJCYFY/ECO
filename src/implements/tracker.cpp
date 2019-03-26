#include "tracker.hpp"
#include "matlab_func.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/eigen.hpp"
#include "fftw3.h"
#include <ctime>
#include <chrono>
namespace Track
{
Tracker::Tracker(float tlx,float tly,float height,float width, Mat im)
{
    config::ConfigParser("./resource/config.cfg");

    m_im_height = im.rows;
    m_im_width = im.cols;

    m_pos.x = tlx+(width-1)/2.0f;
    m_pos.y = tly+(height-1)/2.0f;
    m_init_sz.height = height;
    m_init_sz.width = width;

    float search_area = height * width * config::search_area_scale * config::search_area_scale;
    if(search_area > config::max_image_sample_size)
    {
        m_currentScaleFactor = sqrt(search_area / config::max_image_sample_size);
    }
    else if(search_area < config::min_image_sample_size)
    {
        m_currentScaleFactor = sqrt(search_area / config::min_image_sample_size);
    }
    else
        m_currentScaleFactor = 1.0;
    m_target_sz.height = m_init_sz.height / m_currentScaleFactor;
    m_target_sz.width = m_init_sz.width / m_currentScaleFactor;
    init();
    init_model(im);
}

Tracker::~Tracker()
{
    while(!finished)
        std::this_thread::sleep_for(std::chrono::seconds(1));
    end_thread = true;
    start_train_filter.notify_one();
    train_filter_thread->join();
}

void Tracker::init()
{
    time = 0;
    m_frame_count = 1;
    // 采样区域的大小
    int sample_h,sample_w;
    switch(config::search_area_shape)
    {
        case proportional:
        {
            sample_h = floor(m_target_sz.height*config::search_area_scale);
            sample_w = floor(m_target_sz.width*config::search_area_scale);
            break;
        }
        case square:
        {
            sample_h = sqrt(m_target_sz.height*m_target_sz.width*config::search_area_scale*config::search_area_scale);
            sample_w = sample_h;
            break;
        }
        case fix_padding:
        {
            float pad = sqrt(m_target_sz.height*m_target_sz.width*config::search_area_scale*config::search_area_scale + (m_target_sz.height+m_target_sz.width)/4.0f) + (m_target_sz.height+m_target_sz.width)/2.0f;
            sample_h =  m_target_sz.height + pad;
            sample_w =  m_target_sz.width + pad;
            break;
        }
        case custom:
        {
            sample_h = m_target_sz.height*2;
            sample_w = m_target_sz.width*2;
            break;
        }
        default:
        {
            printf("Wrong search area shape!\nuse square shape.\n");
            sample_h = floor(m_target_sz.height*config::search_area_scale);
            sample_w = floor(m_target_sz.width*config::search_area_scale);
            break;
        }
    }
    // 初始化特征提取
    m_cn_extra = make_shared<CNf>(config::cn_cell_size);
    m_hog_extra = make_shared<fHog>(config::hog_cell_size,config::hog_orient_num);

    // 真正的采样区域大小 
    float max_cell_size = max(config::hog_cell_size,config::cn_cell_size);
    float min_cell_size = min(config::hog_cell_size,config::cn_cell_size);
    switch(config::size_mode)
    {
        case same:
        {
            m_support_sz.height = sample_h;   
            m_support_sz.width = sample_w;
            break;
        }
        case exact:
        {
            m_support_sz.height = round(sample_h/max_cell_size) * max_cell_size;
            m_support_sz.width = round(sample_w/max_cell_size) * max_cell_size;
            break;
        }
        case odd_cells:
        {
            // 采样区域为 奇数个cell_size
            m_support_sz.height = (1+2*round(sample_h/(2*max_cell_size))) * max_cell_size;
            m_support_sz.width = (1+2*round(sample_w/(2*max_cell_size))) * max_cell_size;

            int best_choice = 0;
            int val,max =0;
            for(int i=0;i<max_cell_size;++i)
            {
                val = int((m_support_sz.height+i)/min_cell_size)%2
                        + int((m_support_sz.width+i)/min_cell_size)%2
                        + int((m_support_sz.height+i)/max_cell_size)%2
                        + int((m_support_sz.width+i)/max_cell_size)%2;
                if(max < val)
                {
                    max = val;
                    best_choice = i;
                }
            }
            m_support_sz.height += best_choice;
            m_support_sz.width += best_choice;
            break;
        }
        default:
        {
            printf("Wrong SIZE_MODE\n");
            exit(-1);
            break;
        }
    }
    // 特征 cell数量
    m_cn_feature_sz.height = m_support_sz.height/config::cn_cell_size;
    m_cn_feature_sz.width = m_support_sz.width/config::cn_cell_size;
    m_hog_feature_sz.height = m_support_sz.height/config::hog_cell_size;
    m_hog_feature_sz.width = m_support_sz.width/config::hog_cell_size;

    // 采样结果的维度 (压缩或不压缩)
    int hog_sample_dim,cn_sample_dim;
    if(config::use_projection_matrix)
    {
        cn_sample_dim = config::cn_compressed_dim;
        hog_sample_dim = config::hog_compressed_dim;
    }
    else
    {
        cn_sample_dim = 10;         // cn 不压缩特征维度
        hog_sample_dim = 31;        // hog 不压缩特征维度
    }

    // 保持 filter_sz 为奇数
    m_cn_filter_sz.height = m_cn_feature_sz.height + (m_cn_feature_sz.height+1)%2;
    m_cn_filter_sz.width = m_cn_feature_sz.width + (m_cn_feature_sz.width+1)%2;
    m_hog_filter_sz.height = m_hog_feature_sz.height + (m_hog_feature_sz.height+1)%2;
    m_hog_filter_sz.width = m_hog_feature_sz.width + (m_hog_feature_sz.width+1)%2;

    // 初始化 Train
    m_train = make_shared<Train>(m_cn_filter_sz.height,m_cn_filter_sz.width/2+1,cn_sample_dim,
                                    m_hog_filter_sz.height,m_hog_filter_sz.width/2+1,hog_sample_dim);

    m_output_sz.height = max(m_cn_filter_sz.height,m_hog_filter_sz.height);
    m_output_sz.width = max(m_cn_filter_sz.width,m_hog_filter_sz.width);

    // 根据谁的特征多 谁就是主特征 另一个特征需要填充到主特征大小
    // 需要填充的边 
    m_cn_pad_sz.height = (m_output_sz.height - m_cn_filter_sz.height)/2;
    m_cn_pad_sz.width = (m_output_sz.height - m_cn_filter_sz.width)/2;
    m_hog_pad_sz.height = (m_output_sz.height - m_hog_filter_sz.height)/2;
    m_hog_pad_sz.width = (m_output_sz.width - m_hog_filter_sz.width)/2;


    double osf = sqrt(floor(m_target_sz.height)*floor(m_target_sz.width)) * config::output_sigma_factor;
    double sig_y_height = osf * m_output_sz.height/ m_support_sz.height;
    double sig_y_width = osf * m_output_sz.width / m_support_sz.width;

    double s2pih = sqrt(2*PI) * sig_y_height / m_output_sz.height;
    double s2piw = sqrt(2*PI) * sig_y_width / m_output_sz.width;
    
    // cn ky kx yf 初始化
    m_cn_ky.resize(m_cn_filter_sz.height,1);
    for(int i=0;i<m_cn_filter_sz.height;++i)
    {
        m_cn_ky(i,0) = i- (m_cn_filter_sz.height/2);
    }
    MatrixXd cn_yf_y = s2pih * (-2*(m_cn_ky*PI*sig_y_height/m_output_sz.height).array().pow(2)).exp();
    m_cn_kx.resize(1,m_cn_filter_sz.width/2+1);
    for(int i=0;i<m_cn_filter_sz.width/2+1;++i)
    {
        m_cn_kx(0,i) = i- (m_cn_filter_sz.width/2);
    }
    MatrixXd cn_yf_x = s2piw * (-2 * (m_cn_kx*PI*sig_y_width/m_output_sz.width).array().pow(2)).exp();
    m_train->m_cn_yf = (cn_yf_y * cn_yf_x).cast<float>();
    // Matlab::write_mat("/home/huajun/Documents/hj1/cn_yf",m_train->m_cn_yf.rows(),m_train->m_cn_yf.cols(),m_train->m_cn_yf.data());

    // cn ky kx yf 初始化
    m_hog_ky.resize(m_hog_filter_sz.height,1);
    for(int i=0;i<m_hog_filter_sz.height;++i)
    {
        m_hog_ky(i,0) = i- (m_hog_filter_sz.height/2);
    }
    MatrixXd hog_yf_y = s2pih * (-2*(m_hog_ky*PI*sig_y_height/m_output_sz.height).array().pow(2)).exp();

    m_hog_kx.resize(1,m_hog_filter_sz.width/2+1);
    for(int i=0;i<m_hog_filter_sz.width/2+1;++i)
    {
        m_hog_kx(0,i) = i- (m_hog_filter_sz.width/2);
    }
    MatrixXd hog_yf_x = s2piw * (-2 * (m_hog_kx*PI*sig_y_width/m_output_sz.width).array().pow(2)).exp();
    m_train->m_hog_yf = (hog_yf_y * hog_yf_x).cast<float>();
    
    // Matlab::write_mat("/home/huajun/Documents/hj1/hog_yf",m_train->m_hog_yf.rows(),m_train->m_hog_yf.cols(),m_train->m_hog_yf.data());
    // construct cosine window
    m_cn_cos_win = (Matlab::hannWin(m_cn_feature_sz.height+2) * (Matlab::hannWin(m_cn_feature_sz.width+2).transpose())).middleRows(1,m_cn_feature_sz.height).middleCols(1,m_cn_feature_sz.width).cast<float>();
    m_hog_cos_win = (Matlab::hannWin(m_hog_feature_sz.height+2) * (Matlab::hannWin(m_hog_feature_sz.width+2).transpose())).middleRows(1,m_hog_feature_sz.height).middleCols(1,m_hog_feature_sz.width).cast<float>();

    // Matlab::write_mat("/home/huajun/Documents/hj1/cn_cos_win",m_cn_cos_win.rows(),m_cn_cos_win.cols(),m_cn_cos_win.data());
    // Matlab::write_mat("/home/huajun/Documents/hj1/hog_cos_win",m_hog_cos_win.rows(),m_hog_cos_win.cols(),m_hog_cos_win.data());
    // Compute Fourier series of interpolation function
    get_interp_fourier(m_hog_filter_sz.height,m_hog_filter_sz.width,m_cn_filter_sz.height,m_cn_filter_sz.width);
    // Matlab::write_mat("/home/huajun/Documents/hj1/cn_interp",m_cn_interp.rows(),m_cn_interp.cols(),m_cn_interp.data());
    // Matlab::write_mat("/home/huajun/Documents/hj1/hog_interp",m_hog_interp.rows(),m_hog_interp.cols(),m_hog_interp.data());
    // Construct spatial regularization filter
    get_reg_filter();

    // Matlab::write_mat("/home/huajun/Documents/hj1/reg_filter",m_train->m_reg_filter.rows(),m_train->m_reg_filter.cols(),m_train->m_reg_filter.data());
    // 初始化 ScaleFilter
    m_scalefilter = make_shared<ScaleFilter>(m_init_sz);
    if(m_scalefilter->nScales > 0)
    {
        m_minScaleFactor = pow(m_scalefilter->fScaleStep, ceil(log(max(5.0f/ m_support_sz.height,5.0f/m_support_sz.width)) / log(m_scalefilter->fScaleStep)));
        m_maxScaleFactor = pow(m_scalefilter->fScaleStep, ceil(log(min(m_im_height/(float)m_target_sz.height,m_im_width/(float)m_target_sz.width)) / log(m_scalefilter->fScaleStep)));
    }

    // 初始化 samplesf
    complex<float> zero(0,0);
    m_cn_samplesf.resize(m_cn_filter_sz.height,(m_cn_filter_sz.width+1)/2,cn_sample_dim,config::nSamples,zero);
    m_hog_samplesf.resize(m_hog_filter_sz.height,(m_hog_filter_sz.width+1)/2,hog_sample_dim,config::nSamples,zero);
        
    m_samplespace = make_shared<SampleSpace>();
    m_samplespace->minimum_sample_weight = config::learning_rate*pow((1-config::learning_rate),(2*config::nSamples));    
    m_scores_fs_sum.resize(m_output_sz.height/2+1,m_output_sz.width);

    waitting = false;
    finished = true;
    end_thread = false;
    train_filter_thread = make_shared<thread>(&Tracker::thread_train_filter,this);
}

void Tracker::get_interp_fourier(int hh,int hw,int ch,int cw)
{
    MatrixXcd hog_col,hog_row,cn_col,cn_row;
    switch(config::interpolation_method)
    {
        case none:
        {
            hog_col.resize(hh,1); 
            hog_row.resize(1,hw);
            cn_col.resize(ch,1);
            cn_row.resize(1,cw);
            complex<double> one(1,0);
            hog_col.fill(one);
            hog_row.fill(one);
            cn_col.fill(one);
            cn_row.fill(one);
            break;
        }
        case ideal:
        {
            hog_col.resize(hh,1); 
            hog_row.resize(1,hw);
            cn_col.resize(ch,1);
            cn_row.resize(1,cw);
            complex<double> _hh(1.0/hh,0);
            complex<double> _hw(1.0/hw,0);
            complex<double> _ch(1.0/ch,0);
            complex<double> _cw(1.0/cw,0);
            hog_col.fill(_hh);
            hog_row.fill(_hw);
            cn_col.fill(_ch);
            cn_row.fill(_cw);
            break;
        }
        case bicubic:
        {
            double a = config::interpolation_bicubic_a;
            MatrixXd c_col(ch,1);
            MatrixXd c_row(1,cw);
            MatrixXd h_col(hh,1);
            MatrixXd h_row(1,hw);
            for(int i=0;i<ch;++i)
                c_col(i) = i-ch/2;
            c_col /= ch;
            for(int i=0;i<cw;++i)
                c_row(i) = i-cw/2;
            c_row /= cw;
            for(int i=0;i<hh;++i)
                h_col(i) = i-hh/2;
            h_col /= hh;
            for(int i=0;i<hw;++i)
                h_row(i) = i-hw/2;
            h_row /= hw;

            complex<double> _i2PI(0,-2*PI);
            complex<double> i2PI(0,2*PI);
            complex<double> _i4PI(0,-4*PI);
            complex<double> i4PI(0,4*PI);
            complex<double> i12PI(0,12*PI);
            complex<double> i16PI(0,16*PI);

            cn_col = -(12*((c_col.array()*_i2PI).exp()+(c_col.array()*i2PI).exp()-a) +
                        6*a*((c_col.array()*_i4PI).exp() + (c_col.array()*i4PI).exp()) +
                        c_col.array()*i12PI*((c_col.array()*_i2PI).exp()-(c_col.array()*i2PI).exp()) +
                        a*c_col.array()*i16PI*((c_col.array()*_i2PI).exp()-(c_col.array()*i2PI).exp()) +
                        a*c_col.array()*i4PI*((c_col.array()*_i4PI).exp()-(c_col.array()*i4PI).exp()) -24) / 
                        (16*c_col.array().pow(4)*pow(PI,4)*ch);
            cn_col(ch/2,0) = complex<double>(1.0/ch,0);
            hog_col = -(12*((h_col.array()*_i2PI).exp()+(h_col.array()*i2PI).exp()-a) +
                        6*a*((h_col.array()*_i4PI).exp() + (h_col.array()*i4PI).exp()) +
                        h_col.array()*i12PI*((h_col.array()*_i2PI).exp()-(h_col.array()*i2PI).exp()) +
                        a*h_col.array()*i16PI*((h_col.array()*_i2PI).exp()-(h_col.array()*i2PI).exp()) +
                        a*h_col.array()*i4PI*((h_col.array()*_i4PI).exp()-(h_col.array()*i4PI).exp()) -24) / 
                        (16*h_col.array().pow(4)*pow(PI,4)*hh);
            hog_col(hh/2) = complex<double>(1.0/hh,0);
            cn_row = -(12*((c_row.array()*_i2PI).exp()+(c_row.array()*i2PI).exp()-a) +
                        6*a*((c_row.array()*_i4PI).exp() + (c_row.array()*i4PI).exp()) +
                        c_row.array()*i12PI*((c_row.array()*_i2PI).exp()-(c_row.array()*i2PI).exp()) +
                        a*c_row.array()*i16PI*((c_row.array()*_i2PI).exp()-(c_row.array()*i2PI).exp()) +
                        a*c_row.array()*i4PI*((c_row.array()*_i4PI).exp()-(c_row.array()*i4PI).exp()) -24) / 
                        (16*c_row.array().pow(4)*pow(PI,4)*cw);
            cn_row(cw/2) = complex<double>(1.0/cw,0);

            hog_row = -(12*((h_row.array()*_i2PI).exp()+(h_row.array()*i2PI).exp()-a) +
                        6*a*((h_row.array()*_i4PI).exp() + (h_row.array()*i4PI).exp()) +
                        h_row.array()*i12PI*((h_row.array()*_i2PI).exp()-(h_row.array()*i2PI).exp()) +
                        a*h_row.array()*i16PI*((h_row.array()*_i2PI).exp()-(h_row.array()*i2PI).exp()) +
                        a*h_row.array()*i4PI*((h_row.array()*_i4PI).exp()-(h_row.array()*i4PI).exp()) -24) / 
                        (16*h_row.array().pow(4)*pow(PI,4)*hw);
            hog_row(hw/2) = complex<double>(1.0/hw,0);
            break;
        }
        default:
        {
            printf("Wrong interpolation method!\n");
            exit(-1);
        }
    }

    if(config::interpolation_centering)
    {
        int start;
        start = (hh-1)/2;
        for(int i=0;i<hog_col.rows();++i)
        {
            hog_col(i,0) *= exp(complex<double>(0,-PI/hh*(i-start)));
        }
        start = (ch-1)/2;
        for(int i=0;i<cn_col.rows();++i)
        {
            cn_col(i,0) *= exp(complex<double>(0,-PI/ch*(i-start)));
        }
        start = (hw-1)/2;
        for(int i=0;i<hog_row.cols();++i)
        {
            hog_row(0,i) *= exp(complex<double>(0,-PI/hw*(i-start)));
        }
        start = (cw-1)/2;
        for(int i=0;i<cn_row.cols();++i)
        {
            cn_row(0,i) *= exp(complex<double>(0,-PI/cw*(i-start)));
        }
    }

    if(config::interpolation_windowing)
    {
        MatrixXd win_hog_h = Matlab::hannWin(hh+2);
        MatrixXd win_hog_w = Matlab::hannWin(hw+2);
        MatrixXd win_cn_h = Matlab::hannWin(ch+2);
        win_cn_h = win_cn_h.middleRows(1,ch);
        MatrixXd win_cn_w = Matlab::hannWin(cw+2);
        win_cn_w = win_cn_w.middleRows(1,cw);

        hog_col.array() *= win_hog_h.middleRows(1,hh).array();
        cn_col.array() *= win_cn_h.middleRows(1,ch).array();
        hog_row.array() *= win_hog_w.middleRows(1,hw).array();
        cn_row.array() *= win_cn_w.middleRows(1,cw).array();
    }
    m_cn_interp = (cn_col*cn_row).leftCols(cw/2+1).cast<complex<float> >();
    m_hog_interp = (hog_col*hog_row).leftCols(hw/2+1).cast<complex<float> >();
}

void Tracker::get_reg_filter()
{
    // 初始化 reg_filter 用于 train 里面做卷机运算
    if(config::use_reg_window)
    {
        double reg_w = config::reg_window_edge - config::reg_window_min;
        Size2d reg_scale;
        reg_scale.height = m_target_sz.height *0.5;
        reg_scale.width = m_target_sz.width *0.5;

        int h = m_support_sz.height;
        int w = m_support_sz.width ;
        int half_h = h/2 + 1;
        // 使用 FFTW 计算 dft
        double *in = (double*) fftw_malloc(sizeof(double) * h * w);
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * half_h * w);
        // 用于计算 fft
        fftw_plan p1 = fftw_plan_dft_r2c_2d(w, h, in, out, FFTW_ESTIMATE);
        // 用于计算 ifft
        fftw_plan p2 = fftw_plan_dft_c2r_2d(w, h, out, in, FFTW_ESTIMATE);

        // 数据初始化
        MatrixXd wrs(h,w);
        MatrixXd wcs(w,h);
        double* wg_data = wrs.data();
        float start = (m_support_sz.height-1)/2.0f;
        for(int i=0; i< h;++i)
            wg_data[i] = i - start;
        for(int i=1; i< w;++i)
            memcpy(wg_data+i*h,wg_data,h*sizeof(double));

        wg_data = wcs.data();
        start = (m_support_sz.width-1)/2.0f;
        for(int i=0; i< w;++i)
            wg_data[i] = i - start;
        for(int i=1; i< h;++i)
            memcpy(wg_data+i*w,wg_data,w*sizeof(double));

        Map<MatrixXd>(in,h,w) = reg_w *((wrs.array()/reg_scale.height).abs().pow(config::reg_window_power) + 
                    (wcs.transpose().array()/reg_scale.width).abs().pow(config::reg_window_power)) + config::reg_window_min;

        fftw_execute(p1); /* repeat as needed */
        Map<MatrixXcd>((complex<double>*)out,half_h,w) /= (h*w);
        MatrixXd dft_abs= Map<MatrixXcd>((complex<double>*)out,half_h,w).array().abs();
        double max_rwd = dft_abs.maxCoeff() * config::reg_sparsity_threshold;
        
        complex<double> complex_zero(0,0);
        complex<double> complex_mzero(0,-0);
        complex<double>* cd_ptr = (complex<double>*)out;
        double* d_ptr = dft_abs.data();
        for(int i=0;i<dft_abs.size();++i)
        {
            if(d_ptr[i]<max_rwd)
                cd_ptr[i] = complex_zero;
        }
        MatrixXcd reg_window_dft(h,w);
        
        reg_window_dft.topRows(half_h) = Map<MatrixXcd>((complex<double>*)out,half_h,w);
        reg_window_dft.bottomLeftCorner(h-half_h,1) = Map<MatrixXcd>((complex<double>*)out,half_h,w).middleRows(1,h-half_h).leftCols(1).colwise().reverse().conjugate();
        reg_window_dft.bottomRightCorner(h-half_h,w-1) = Map<MatrixXcd>((complex<double>*)out,half_h,w).middleRows(1,h-half_h).rightCols(w-1).colwise().reverse().rowwise().reverse().conjugate();
        fftw_execute(p2);
        double min_rws = Map<MatrixXd>(in,h,w).array().minCoeff();
        reg_window_dft(0,0) = reg_window_dft(0,0) - (min_rws - config::reg_window_min);
        fftw_destroy_plan(p1);
        fftw_destroy_plan(p2);
        fftw_free(in); fftw_free(out);
        // shift 
        MatrixXcd reg_window_dft_shift(h,w);
        int hh = (h+1)/2; int wh = (w+1)/2;
        reg_window_dft_shift.topLeftCorner(h-hh, w-wh) = reg_window_dft.bottomRightCorner(h-hh, w-wh);
        reg_window_dft_shift.topRightCorner(h-hh, wh) = reg_window_dft.bottomLeftCorner(h-hh, wh);
        reg_window_dft_shift.bottomLeftCorner(hh, w-wh) = reg_window_dft.topRightCorner(hh, w-wh);
        reg_window_dft_shift.bottomRightCorner(hh, wh) = reg_window_dft.topLeftCorner(hh, wh);
        MatrixXi z_Mat;
        z_Mat = MatrixXi::Zero(h,w);
        cd_ptr = reg_window_dft_shift.data();
        int *i_ptr = z_Mat.data();
        for(int i=0;i<h*w;++i)
            if((cd_ptr[i] != complex_zero) && (cd_ptr[i] !=complex_mzero))
            {
                i_ptr[i] = 1;
            }
        Matrix<int,1,Dynamic> rowV = z_Mat.colwise().any();
        Matrix<int,1,Dynamic> colV = z_Mat.rowwise().any();
        Matrix<int,1,Dynamic> rowVD(1,rowV.sum());
        i_ptr = rowVD.data();
        for(int i=0;i<rowV.size();++i)
            if(rowV(i)!=0)
                *(i_ptr++) = i;
        Matrix<int,1,Dynamic> colVD(1,colV.sum());
        i_ptr = colVD.data();
        for(int i=0;i<colV.size();++i)
            if(colV(i)!=0)
                *(i_ptr++) = i;
        m_train->m_reg_filter.resize(colVD.size(),rowVD.size());
        float* f_ptr = m_train->m_reg_filter.data();
        for(int i=0;i<rowVD.size();++i)
            for(int j=0;j<colVD.size();++j)
                *(f_ptr++) = reg_window_dft_shift(colVD(j),rowVD(i)).real();
    }
    else
    {
        m_train->m_reg_filter.resize(1,1);
        m_train->m_reg_filter(0,0) = config::reg_window_min;
    }
    m_train->m_reg_energy = m_train->m_reg_filter.array().abs2().sum();
}

void Tracker::track(Mat img)
{
    m_scalefilter->sScale.interpScaleFactors.resize(33);

    clock_t start, end;
    start = clock();
    target_localization(img);
    model_update(img);
    end = clock();
    cout << " track cost : "<<(double)(end - start) / CLOCKS_PER_SEC <<" s "<< endl;
    time +=( (double)(end - start) / CLOCKS_PER_SEC);
    // Visualizing
    Size target_sz = m_target_sz * m_currentScaleFactor;
    cv::Rect rect_position_vis(m_pos.x - target_sz.width/2,m_pos.y-target_sz.height/2,target_sz.width,target_sz.height);

    // // visualize
    // MatrixXcf extend = MatrixXcf::Zero(409/2+1,409);
    // extend.topLeftCorner(m_scores_fs_sum.rows(),m_scores_fs_sum.cols()/2+1) = m_scores_fs_sum.leftCols(m_scores_fs_sum.cols()/2+1);
    // extend.topRightCorner(m_scores_fs_sum.rows(),m_scores_fs_sum.cols() - m_scores_fs_sum.cols()/2-1) = 
    //                                                     m_scores_fs_sum.rightCols(m_scores_fs_sum.cols()-m_scores_fs_sum.cols()/2-1);
    // MatrixXf score;
    // Matlab::ifft2(extend,score,409,409);
    // MatrixXf score_shift(409,409);
    // score_shift.topLeftCorner(204,204) = score.bottomRightCorner(204,204);
    // score_shift.topRightCorner(204,205) = score.bottomLeftCorner(204,205);
    // score_shift.bottomLeftCorner(205,204) = score.topRightCorner(205,204);
    // score_shift.bottomRightCorner(205,205) = score.topLeftCorner(205,205);
    // float min = score_shift.minCoeff();
    // float max = score_shift.maxCoeff();
    // MatrixXi i_im = ((score_shift.array() - min) * 255 /(max-min)).cast<int>();

    // Mat weight;
    // eigen2cv(i_im,weight);
    // weight.convertTo(weight, CV_8UC1);
    // Mat im_color;
    // applyColorMap(weight, im_color, COLORMAP_JET);

    // imshow("color",im_color);

    // Size resp_sz;
    // resp_sz.height = round(m_support_sz.height*m_currentScaleFactor);
    // resp_sz.width = round(m_support_sz.width*m_currentScaleFactor);

    // resize(im_color,im_color,resp_sz);
    // int left=0,right=0,top=0,bottom=0;
    // if( m_sample_pos.x-resp_sz.width/2 < 0)
    // {
    //     left = resp_sz.width/2 - m_sample_pos.x;
    // }
    // if(m_sample_pos.x-resp_sz.width/2 + resp_sz.width > img.cols -1)
    // {
    //     right = (m_sample_pos.x-resp_sz.width/2 + resp_sz.width) - img.cols + 1;
    // }
    // if( m_sample_pos.y-resp_sz.height/2 < 0)
    // {
    //     top = resp_sz.height/2 - m_sample_pos.y;
    // }
    // if(m_sample_pos.y-resp_sz.height/2 + resp_sz.height > img.rows -1)
    // {
    //     bottom = (m_sample_pos.y-resp_sz.height/2 + resp_sz.height)-img.rows +1;
    // }
    // Mat ROI1 = img(Rect(m_sample_pos.x-resp_sz.width/2 + left ,m_sample_pos.y-resp_sz.height/2 + top,resp_sz.width - left - right,resp_sz.height - top - bottom));//(im,rect_position_vis);
    // Mat ROI2 = im_color(Rect(left,top,resp_sz.width - left - right,resp_sz.height - top - bottom));
    // addWeighted(ROI1, 0.5, ROI2, 0.5, 0.0, ROI1);
    rectangle(img, rect_position_vis, Scalar(0,255,0), 2);
    resize(img,img,Size(),0.5,0.5);
    imshow("ret",img);
    waitKey(1);     
    ++m_frame_count;
}

void Tracker::extract_features(cv::Mat img, cv::Point& pos, float scale,
                         shared_ptr<Feature::Feature>& cn_features,
                         shared_ptr<Feature::Feature>& hog_features)
{
    cv::Mat img_sample;
    cv::Size2f img_scale_sz;
    img_scale_sz.height = m_support_sz.height * scale;
    img_scale_sz.width = m_support_sz.width * scale;
    img_sample = sample_patch(img,pos,img_scale_sz,m_support_sz);
    cn_features = m_cn_extra->extract(img_sample);
    hog_features = m_hog_extra->extract(img_sample); 
    if(config::normalize_power==2)
    {
        cn_features->norm(config::normalize_power,config::normalize_size,config::normalize_dim);
        hog_features->norm(config::normalize_power,config::normalize_size,config::normalize_dim);
    }
}

Mat Tracker::sample_patch(cv::Mat img, cv::Point pos, cv::Size2f sample_sz, cv::Size output_sz)
{
    assert(img.channels()==3);
    float resize_factor = min(sample_sz.height/output_sz.height,sample_sz.width/output_sz.width);
    int df = max(int(resize_factor - 0.1),1);
    if(df>1)
    {
        if(pos.y<0)
            pos.y = 0;
        if(pos.x<0)
            pos.x = 0;
        if(pos.y>=m_im_height)
            pos.y = m_im_height-1;
        if(pos.x>=m_im_width)
            pos.x = m_im_width-1;

        int osh = pos.y%df;
        int osw = pos.x%df;
        pos.y = (pos.y-osh)/df;
        pos.x = (pos.x-osw)/df;

        sample_sz.height = sample_sz.height / df;
        sample_sz.width = sample_sz.width / df;

        Mat temp(ceil((img.rows-osh)/(float)df),ceil((img.cols-osw)/(float)df),CV_8UC3);
        uchar* t_data,* im_data = img.data;
        int r=0;
        for(int row=osh;row<img.rows;row+=df)
        {
            t_data = temp.ptr<uchar>(r);
            im_data = img.ptr<uchar>(row);
            r++;
            int c=0;
            for(int col=osw;col<img.cols;col+=df)
            {
                t_data[3*c] = im_data[3*col];
                t_data[3*c+1] = im_data[3*col+1];
                t_data[3*c+2] = im_data[3*col+2];
                ++c;
            }
        }
        img = temp;
    }
    cv::Size sample_size;
    sample_size.height = max(int(sample_sz.height+0.5),2);
    sample_size.width = max(int(sample_sz.width+0.5),2);
    int half_w = floor(sample_sz.width/2);
    int half_h = floor(sample_sz.height/2);
    
    cv::Mat im_patch(sample_size,CV_8UC3,Scalar(0,0,0));
    int minx = pos.x - half_w;//包含
    int maxx = minx + sample_size.width; //不包含
    int miny = pos.y - half_h;//包含
    int maxy = miny + sample_size.height;//不包含

    if(minx<img.cols && miny<img.rows && maxx>=0 && maxy>=0)
    {
        int im_x=0,im_y=0,im_w=sample_size.width,im_h=sample_size.height;
        if(minx<0)
        {
            im_x = -minx;
            im_w = sample_size.width + minx;
            minx = 0;
        }
        if(maxx>img.cols)
        {
            im_w = im_w - maxx + img.cols;
        }
        if(miny<0)
        {
            im_y = -miny;
            im_h = sample_size.height + miny;
            miny = 0;
        }
        if(maxy>img.rows)
        {
            im_h = im_h - maxy + img.rows ;
        }
        uchar *im_data , *im_patch_data;

        for(int row=0;row<im_h;++row)
        {
            im_data = img.ptr<uchar>(miny+row) + 3*minx;
            im_patch_data = im_patch.ptr<uchar>(im_y+row)+ 3*im_x;
            memcpy(im_patch_data,im_data,sizeof(uchar)*im_w*3);
        }
    }

    if(output_sz == sample_size)
        return im_patch;
    cv::Mat ret;
    cv::resize(im_patch,ret,output_sz);
    return ret;
}

void Tracker::target_localization(Mat img)
{
    Point2f old_pos(Infinity,Infinity);
    for(int iter=0;iter<config::refinement_iterations && m_pos!=old_pos;++iter)
    {
        m_sample_pos.x = round(m_pos.x);
        m_sample_pos.y = round(m_pos.y);
        float sample_scale = m_currentScaleFactor;
        shared_ptr<Feature::Feature> cn_xt, hog_xt;
        extract_features(img, m_sample_pos, sample_scale, cn_xt, hog_xt);
        Array4D<float> cn_xt_proj(cn_xt->dim1,cn_xt->dim2,m_train->m_cn_projection_matrix.cols(),1);
        Array4D<float> hog_xt_proj(hog_xt->dim1,hog_xt->dim2,m_train->m_hog_projection_matrix.cols(),1);

        Map<MatrixXf>(cn_xt_proj.data,cn_xt_proj.stride12,cn_xt_proj.dim3) = 
                Map<MatrixXf>(cn_xt->data,cn_xt->dim1*cn_xt->dim2,cn_xt->dim3) * m_train->m_cn_projection_matrix;
        Map<MatrixXf>(hog_xt_proj.data,hog_xt_proj.stride12,hog_xt_proj.dim3) = 
                Map<MatrixXf>(hog_xt->data,hog_xt->dim1*hog_xt->dim2,hog_xt->dim3) * m_train->m_hog_projection_matrix;

        for(int i=0;i<cn_xt_proj.dim3;++i)
            Map<MatrixXf>(cn_xt_proj.data + i*cn_xt_proj.stride12,cn_xt_proj.dim1,cn_xt_proj.dim2) = 
                    Map<MatrixXf>(cn_xt_proj.data + i*cn_xt_proj.stride12,cn_xt_proj.dim1,cn_xt_proj.dim2).array() * m_cn_cos_win.array();
        for(int i=0;i<hog_xt_proj.dim3;++i)
            Map<MatrixXf>(hog_xt_proj.data + i*hog_xt_proj.stride12,hog_xt_proj.dim1,hog_xt_proj.dim2) = 
                    Map<MatrixXf>(hog_xt_proj.data + i*hog_xt_proj.stride12,hog_xt_proj.dim1,hog_xt_proj.dim2).array() * m_hog_cos_win.array();

        Matlab::fft2(cn_xt_proj,m_cn_xlf_proj);
        Matlab::fft2(hog_xt_proj,m_hog_xlf_proj);
        // Interpolate features to the continuous domain
        for(int i=0;i<m_cn_xlf_proj.dim3;++i)
            m_cn_xlf_proj.getMatrix12(i).array() *= m_cn_interp.array();
        for(int i=0;i<m_hog_xlf_proj.dim3;++i)
            m_hog_xlf_proj.getMatrix12(i).array() *= m_hog_interp.array();

        m_scores_fs_feat1.setZero(m_cn_xlf_proj.dim1,m_cn_xlf_proj.dim2);
        m_scores_fs_feat2.setZero(m_hog_xlf_proj.dim1,m_hog_xlf_proj.dim2);
        if(m_cn_pad_sz.height == 0 && m_cn_pad_sz.width == 0)
        {
            std::unique_lock<std::mutex> lock(m_train->hf_mutex);
            {
                for(int i=0;i<m_cn_xlf_proj.dim3;++i)
                {
                    m_scores_fs_feat1.array() += m_train->m_cn_hf1.getMatrix12(i).array() * m_cn_xlf_proj.getMatrix12(i).array();
                }
                for(int i=0;i<m_hog_xlf_proj.dim3;++i)
                {
                    m_scores_fs_feat2.array() += m_train->m_hog_hf1.getMatrix12(i).array() * m_hog_xlf_proj.getMatrix12(i).array();
                }
            }
            m_scores_fs_feat1.middleRows(m_hog_pad_sz.height, m_scores_fs_feat2.rows()).rightCols(m_scores_fs_feat2.cols()) += m_scores_fs_feat2;

            m_scores_fs_sum.leftCols(1) = m_scores_fs_feat1.bottomRightCorner(m_output_sz.height/2+1,1);
            m_scores_fs_sum.middleCols(1,m_output_sz.width/2).topRows(1) = m_scores_fs_feat1.middleRows(m_output_sz.height/2+1,1).leftCols(m_output_sz.width/2).rowwise().reverse().conjugate();
            m_scores_fs_sum.middleCols(1,m_output_sz.width/2).bottomRows(m_output_sz.height/2) = m_scores_fs_feat1.topLeftCorner(m_output_sz.height/2,m_output_sz.width/2).colwise().reverse().rowwise().reverse().conjugate();
            m_scores_fs_sum.rightCols(m_output_sz.width/2) = m_scores_fs_feat1.bottomLeftCorner(m_output_sz.height/2+1,m_output_sz.width/2);
            
            Point2f trans = optimize_scores();

            float translation_vec_y = trans.y * m_support_sz.height / m_output_sz.height * m_currentScaleFactor; 
            float translation_vec_x = trans.x * m_support_sz.width / m_output_sz.width * m_currentScaleFactor; 
            
            old_pos = m_pos;
            m_pos.x = m_pos.x + translation_vec_x;
            m_pos.y = m_pos.y + translation_vec_y;
            if(config::clamp_position)
            {
                m_pos.x = max(0,min(m_im_width-1,(int)m_pos.x));
                m_pos.y = max(0,min(m_im_height-1,(int)m_pos.y));
            }
            float scale_change_factor = m_scalefilter->Track(img,m_pos,m_target_sz,m_currentScaleFactor);
            m_currentScaleFactor *= scale_change_factor;
        }
        else
        { 
            // wait to do;
            printf("wait to to!\n");
            exit(-1);
        }
    }
}

void Tracker::init_model(Mat img)
{    
    // Extract image region for training sample
    m_sample_pos.x = round(m_pos.x);
    m_sample_pos.y = round(m_pos.y);
    shared_ptr<Feature::Feature> cn_xl,hog_xl;
    extract_features(img,m_sample_pos,m_currentScaleFactor,cn_xl,hog_xl);
    // Do windowing of features
    Array4D<float> cn_xlw,hog_xlw;
    cn_xlw.resize(cn_xl->dim1,cn_xl->dim2,cn_xl->dim3,1);
    hog_xlw.resize(hog_xl->dim1,hog_xl->dim2,hog_xl->dim3,1);
    
    for(int i=0;i<cn_xl->dim3;++i)
    {
        Map<MatrixXf>(cn_xlw.data + i*cn_xlw.dim1*cn_xlw.dim2,cn_xlw.dim2,cn_xlw.dim1).array() = 
                    Map<MatrixXf>(cn_xl->data+ i*cn_xl->dim1*cn_xl->dim2,cn_xl->dim2,cn_xl->dim1).array() * m_cn_cos_win.array();
    }
    for(int i=0;i<hog_xl->dim3;++i)
    {
        Map<MatrixXf>(hog_xlw.data+ i*hog_xlw.dim1*hog_xlw.dim2,hog_xlw.dim2,hog_xlw.dim1).array() = 
                    Map<MatrixXf>(hog_xl->data+i*hog_xl->dim1*hog_xl->dim2,hog_xl->dim2,hog_xl->dim1).array() * m_hog_cos_win.array();
    }
    // Compute the fourier series
    Array4D<complex<float> > cn_xlf,hog_xlf;
    Matlab::fft2(cn_xlw,cn_xlf);
    Matlab::fft2(hog_xlw,hog_xlf);
    // Interpolate features to the continuous domain
    for(int i=0;i<cn_xlf.dim3;++i)
        cn_xlf.getMatrix12(i).array() *= m_cn_interp.array();
    for(int i=0;i<hog_xlf.dim3;++i)
        hog_xlf.getMatrix12(i).array() *= m_hog_interp.array();
        
    // Shift sample
    Point2d shift_samp;
    shift_samp.y = 2*PI*(m_pos.y - m_sample_pos.y)/(m_currentScaleFactor*m_support_sz.height);
    shift_samp.x = 2*PI*(m_pos.x - m_sample_pos.x)/(m_currentScaleFactor*m_support_sz.width);
    shift_sample(cn_xlf, hog_xlf, shift_samp);

    // Init the projection matrix
    init_projection_matrix(cn_xl,hog_xl);

    // Project sample
    m_cn_xlf_proj.resize(cn_xlf.dim1,cn_xlf.dim2,m_train->m_cn_projection_matrix.cols(),1);    
    m_hog_xlf_proj.resize(hog_xlf.dim1,hog_xlf.dim2,m_train->m_hog_projection_matrix.cols(),1);  

    Map<MatrixXcf>(m_cn_xlf_proj.data,m_cn_xlf_proj.dim1*m_cn_xlf_proj.dim2,m_cn_xlf_proj.dim3) = 
                        Map<MatrixXcf>(cn_xlf.data,cn_xlf.stride12,cn_xlf.dim3) * m_train->m_cn_projection_matrix;
    Map<MatrixXcf>(m_hog_xlf_proj.data,m_hog_xlf_proj.dim1*m_hog_xlf_proj.dim2,m_hog_xlf_proj.dim3) = 
                        Map<MatrixXcf>(hog_xlf.data,hog_xlf.stride12,hog_xlf.dim3) * m_train->m_hog_projection_matrix;

    if(config::use_sample_merge)
        m_samplespace->update_sample_space_model(m_cn_samplesf,m_hog_samplesf,m_cn_xlf_proj,m_hog_xlf_proj);


    m_train->m_cn_sample_energy.resize(m_cn_xlf_proj.dim1,m_cn_xlf_proj.dim2,m_cn_xlf_proj.dim3,m_cn_xlf_proj.dim4);
    m_train->m_hog_sample_energy.resize(m_hog_xlf_proj.dim1,m_hog_xlf_proj.dim2,m_hog_xlf_proj.dim3,m_hog_xlf_proj.dim4);
    Map<MatrixXf>(m_train->m_cn_sample_energy.data,1,m_train->m_cn_sample_energy.size) = Map<MatrixXcf>(m_cn_xlf_proj.data,1,m_cn_xlf_proj.size).array().abs2();
    Map<MatrixXf>(m_train->m_hog_sample_energy.data,1,m_train->m_hog_sample_energy.size) = Map<MatrixXcf>(m_hog_xlf_proj.data,1,m_hog_xlf_proj.size).array().abs2();

    m_train->m_CG_opts.maxit = config::init_CG_iter/config::init_GN_iter;
    m_train->m_CG_opts.CG_use_FR = true;
    m_train->m_CG_opts.CG_standard_alpha = true;

    // 仅在train_joint 中使用 ，可以移到其他地方
    m_train->m_cn_proj_energy = 2*MatrixXf::Ones(m_train->m_cn_projection_matrix.rows(),m_train->m_cn_projection_matrix.cols()) * 
                                        m_train->m_cn_yf.array().abs2().sum() / 41;
    m_train->m_hog_proj_energy = 2*MatrixXf::Ones(m_train->m_hog_projection_matrix.rows(),m_train->m_hog_projection_matrix.cols()) * 
                                        m_train->m_hog_yf.array().abs2().sum() / 41;
    m_train->train_joint(cn_xlf,hog_xlf);

    // Re-project and insert training sample
    Map<MatrixXcf>(m_cn_xlf_proj.data,m_cn_xlf_proj.stride12,m_cn_xlf_proj.dim3) = 
                Map<MatrixXcf>(cn_xlf.data,cn_xlf.stride12,cn_xlf.dim3) * m_train->m_cn_projection_matrix;
    Map<MatrixXcf>(m_hog_xlf_proj.data,m_hog_xlf_proj.stride12,m_hog_xlf_proj.dim3) = 
                        Map<MatrixXcf>(hog_xlf.data,hog_xlf.stride12,hog_xlf.dim3) * m_train->m_hog_projection_matrix;
                
    Map<MatrixXcf>(m_cn_samplesf.data,m_cn_samplesf.stride12*m_cn_samplesf.dim3,m_cn_samplesf.dim4).leftCols(1) = m_cn_xlf_proj.getVector();
    Map<MatrixXcf>(m_hog_samplesf.data,m_hog_samplesf.stride12*m_hog_samplesf.dim3,m_hog_samplesf.dim4).leftCols(1) = m_hog_xlf_proj.getVector();

    m_samplespace->gram_matrix(0,0) = 2*(m_cn_xlf_proj.getVector().array().abs2().sum()+m_hog_xlf_proj.getVector().array().abs2().sum());
    
    m_train->m_CG_opts.CG_use_FR = config::CG_use_FR;
    m_train->m_CG_opts.CG_standard_alpha = config::CG_standard_alpha;
    m_train->m_CG_opts.maxit = config::CG_iter;

    m_scalefilter->Update(img, m_pos, m_target_sz, m_currentScaleFactor);
    ++m_frame_count;
}

void Tracker::model_update(Mat img)
{
    if(config::learning_rate>0)
    {
        Point2d shift_samp;
        shift_samp.y = 2*PI*(m_pos.y - m_sample_pos.y)/(m_currentScaleFactor*m_support_sz.height);
        shift_samp.x = 2*PI*(m_pos.x - m_sample_pos.x)/(m_currentScaleFactor*m_support_sz.width);
        shift_sample(m_cn_xlf_proj, m_hog_xlf_proj, shift_samp);
    }
    if(config::use_sample_merge)
        m_samplespace->update_sample_space_model(m_cn_samplesf,m_hog_samplesf,m_cn_xlf_proj,m_hog_xlf_proj);
    else
    {
        printf("track 7 To Do\n");
        exit(0);
    }
    if(config::learning_rate>0)
    {
        if(m_samplespace->merged_sample_id >= 0)
        {
            Map<MatrixXcf>(m_cn_samplesf.data,m_cn_samplesf.stride12*m_cn_samplesf.dim3,m_cn_samplesf.dim4).middleCols(m_samplespace->merged_sample_id,1) = 
                m_samplespace->cn_merged_sample.getVector(); 
            Map<MatrixXcf>(m_hog_samplesf.data,m_hog_samplesf.stride12*m_hog_samplesf.dim3,m_hog_samplesf.dim4).middleCols(m_samplespace->merged_sample_id,1) = 
                m_samplespace->hog_merged_sample.getVector(); 
        }
        if(m_samplespace->new_sample_id >= 0)
        {
            Map<MatrixXcf>(m_cn_samplesf.data,m_cn_samplesf.stride12*m_cn_samplesf.dim3,m_cn_samplesf.dim4).middleCols(m_samplespace->new_sample_id,1) = 
                m_cn_xlf_proj.getVector(); 
            Map<MatrixXcf>(m_hog_samplesf.data,m_hog_samplesf.stride12*m_hog_samplesf.dim3,m_hog_samplesf.dim4).middleCols(m_samplespace->new_sample_id,1) = 
                m_hog_xlf_proj.getVector();
        }
    }
    if(m_frame_count<config::skip_after_frame)
    {    
        m_train->m_cn_sample_energy.getVector() = (1-config::learning_rate)*m_train->m_cn_sample_energy.getVector().array() * config::learning_rate * m_cn_xlf_proj.getVector().array().abs2();
        m_train->m_hog_sample_energy.getVector() = (1-config::learning_rate)*m_train->m_hog_sample_energy.getVector().array() * config::learning_rate * m_hog_xlf_proj.getVector().array().abs2();
        
        m_train->train_filter(m_cn_samplesf,m_hog_samplesf, m_samplespace->prior_weights);        
        m_frames_since_last_train = 0;
    }
    else if(m_frames_since_last_train >= config::train_gap)
    {
        // 如果已经结束更新 开始下一轮更新 否则 等待更新结束
        if(finished)
        {
            {
                unique_lock<mutex> lock_start(wait_finish_mutex);
                waitting = false;
                finished = false;
            }
            start_train_filter.notify_one();
        }
        else
        {
            unique_lock<mutex> lock_start(wait_finish_mutex);
            unique_lock<mutex> lock(finish);
            finish_train_filter.wait(lock);
            waitting = true;
        }
        m_frames_since_last_train = 0;
    }
    else
        ++m_frames_since_last_train;

    m_scalefilter->Update(img, m_pos, m_target_sz, m_currentScaleFactor);
}

void Tracker::thread_train_filter()
{          
    while(true)
    {
        unique_lock<mutex> lock(start);
        start_train_filter.wait(lock);
        if(end_thread)
            break;
        Array4D<complex<float> > cn_samplesf,hog_samplesf;
        VectorXf weights;
        {
            unique_lock<mutex> lock(samplesf_mutex);
            cn_samplesf = m_cn_samplesf;
            hog_samplesf = m_hog_samplesf;
            weights = m_samplespace->prior_weights;

            m_train->m_cn_sample_energy.getVector() = (1-config::learning_rate)*m_train->m_cn_sample_energy.getVector().array() * config::learning_rate * m_cn_xlf_proj.getVector().array().abs2();
            m_train->m_hog_sample_energy.getVector() = (1-config::learning_rate)*m_train->m_hog_sample_energy.getVector().array() * config::learning_rate * m_hog_xlf_proj.getVector().array().abs2();
        }
        m_train->train_filter(cn_samplesf,hog_samplesf, weights);
        // 如果 等待更新中 提示开始更新
        {
            unique_lock<mutex> lock_start(wait_finish_mutex);
            cout<<" train_filter "<<endl;
            finished = true;
            if(waitting)
                finish_train_filter.notify_one();
        }
    }        
}

void Tracker::shift_sample(Array4D<complex<float> >& cn_xlf,Array4D<complex<float> >& hog_xlf, Point2d shift)
{
    complex<double> shift_x(0,shift.x);
    complex<double> shift_y(0,shift.y);

    MatrixXcd cn_shift_exp_y(m_cn_ky.rows(),1);
    complex<double>* cn_shift_exp_y_data = cn_shift_exp_y.data();
    double* ky_data = m_cn_ky.data();
    for(int i=0;i<cn_shift_exp_y.rows();++i)
        cn_shift_exp_y_data[i] = exp(shift_y*complex<double>(ky_data[i],0));
    
    MatrixXcd hog_shift_exp_y(m_hog_ky.rows(),1);
    complex<double>* hog_shift_exp_y_data = hog_shift_exp_y.data();
    ky_data = m_hog_ky.data();
    for(int i=0;i<hog_shift_exp_y.rows();++i)
        hog_shift_exp_y_data[i] = exp(shift_y*complex<double>(ky_data[i],0));

    MatrixXcd cn_shift_exp_x(1,m_cn_kx.cols());
    complex<double>* cn_shift_exp_x_data = cn_shift_exp_x.data();
    double* kx_data = m_cn_kx.data();
    for(int i=0;i<cn_shift_exp_x.cols();++i)
        cn_shift_exp_x_data[i] = exp(shift_x*complex<double>(kx_data[i],0));

    MatrixXcd hog_shift_exp_x(1,m_hog_kx.cols());
    complex<double>* hog_shift_exp_x_data = hog_shift_exp_x.data();
    kx_data = m_hog_kx.data();
    for(int i=0;i<hog_shift_exp_x.cols();++i)
        hog_shift_exp_x_data[i] = exp(shift_x*complex<double>(kx_data[i],0));
    MatrixXcf cn_shift_exp = (cn_shift_exp_y*cn_shift_exp_x).cast<complex<float> >();
    MatrixXcf hog_shift_exp = (hog_shift_exp_y*hog_shift_exp_x).cast<complex<float> >();

    for(int i=0;i<cn_xlf.dim3;++i)
    {
        Map<MatrixXcf>(cn_xlf.data+i*cn_xlf.stride12,cn_xlf.dim1,cn_xlf.dim2).array() *= cn_shift_exp.array();
    }
    for(int i=0;i<hog_xlf.dim3;++i)
    {
        Map<MatrixXcf>(hog_xlf.data+i*hog_xlf.stride12,hog_xlf.dim1,hog_xlf.dim2).array() *= hog_shift_exp.array();
    }
}

void Tracker::init_projection_matrix(shared_ptr<Feature::Feature>& cn_f,shared_ptr<Feature::Feature>& hog_f)
{
    int cn_rows = cn_f->dim1;
    int cn_cols = cn_f->dim2;
    int cn_dims = cn_f->dim3;
    int cn_stride = cn_cols*cn_rows;

    int hog_rows = hog_f->dim1;
    int hog_cols = hog_f->dim2;
    int hog_dims = hog_f->dim3;
    int hog_stride = hog_cols*hog_rows;

    // 计算均值 再做减法
    float *data,*dst;
    float* __attribute__((aligned(16))) cn_data = new float[cn_cols*cn_rows*cn_dims];
    for(int dim=0;dim<cn_dims;++dim)
    {
        #ifdef __SSE2__
            data = cn_f->data+dim*cn_stride;
            dst = cn_data+dim*cn_stride;
            float sum_4[4];
            const int aliq = cn_stride-4;
            __m128 _128_sum = _mm_set1_ps(0.0f);
            int index=0;
            for(;index<=aliq;index+=4)
            {
                _128_sum = _mm_add_ps(_128_sum,_mm_loadu_ps(data+index));
            }
            _mm_storeu_ps(sum_4,_128_sum);
            float sum=0;
            sum = sum_4[0]+sum_4[1]+sum_4[2]+sum_4[3];
            for(;index<cn_stride;++index)
            {
                sum += data[index];
            }
            float mean = sum/cn_stride;
            __m128 _128_mean = _mm_set1_ps(mean);
            index=0;
            for(;index<=aliq;index+=4)
            {
                _mm_storeu_ps(dst+index,_mm_sub_ps(_mm_loadu_ps(data+index),_128_mean));
            }
            for(;index<cn_stride;++index)
            {
                dst[index] = data[index]-mean;
            }
        #else
            data = cn_f->data+dim*cn_stride;
            dst = cn_data+dim*cn_stride;
            int index=0;
            float sum=0;
            for(;index<cn_stride;++index)
            {
                sum += data[index];
            }
            float mean = sum/cn_stride;                        
            for(index=0;index<cn_stride;++index)
            {
                dst[index] = data[index]-mean;
            }
        #endif
    }
    float* __attribute__((aligned(16))) hog_data = new float[hog_cols*hog_rows*hog_dims];
    for(int dim=0;dim<hog_dims;++dim)
    {
        data = hog_f->data+dim*hog_stride;
        dst = hog_data+dim*hog_stride;
        #if __SSE2__
            float sum_4[4];
            const int aliq = hog_stride-4;
            __m128 _128_sum = _mm_set1_ps(0.0f);
            int index=0;
            for(;index<=aliq;index+=4)
            {
                _128_sum = _mm_add_ps(_128_sum,_mm_loadu_ps(data+index));
            }
            _mm_storeu_ps(sum_4,_128_sum);
            float sum=0;
            sum = sum_4[0]+sum_4[1]+sum_4[2]+sum_4[3];
            for(;index<hog_stride;++index)
            {
                sum += data[index];
            }
            float mean = sum/hog_stride;   
            
            __m128 _128_mean = _mm_set1_ps(mean);
            index=0;
            for(;index<=aliq;index+=4)
            {
                _mm_storeu_ps(dst+index,_mm_sub_ps(_mm_loadu_ps(data+index),_128_mean));
            }
            for(;index<hog_stride;++index)
            {
                dst[index] = data[index]-mean;
            }
        #else
            int index=0;
            float sum=0;
            for(;index<hog_stride;++index)
            {
                sum += data[index];
            }
            float mean = sum/hog_stride;               
            for(index=0;index<hog_stride;++index)
            {
                dst[index] = data[index]-mean;
            }
        #endif
    }

    switch(config::proj_init_method)
    {
        case pca:
        {
            Eigen::JacobiSVD<MatrixXf> svd_cn(Eigen::Map<MatrixXf,Eigen::Aligned16>(cn_data,cn_rows*cn_cols,cn_dims).transpose()*
                                                Eigen::Map<MatrixXf,Eigen::Aligned16>(cn_data,cn_rows*cn_cols,cn_dims), Eigen::ComputeThinU);
            m_train->m_cn_projection_matrix = svd_cn.matrixU().leftCols(config::cn_compressed_dim);  

            Eigen::JacobiSVD<MatrixXf> svd_hog(Eigen::Map<MatrixXf,Eigen::Aligned16>(hog_data,hog_rows*hog_cols,hog_dims).transpose()*
                                                Eigen::Map<MatrixXf,Eigen::Aligned16>(hog_data,hog_rows*hog_cols,hog_dims), 
                                                Eigen::ComputeThinU);
            m_train->m_hog_projection_matrix = svd_hog.matrixU().leftCols(config::hog_compressed_dim);        
            break;
        }
        case rand_uni:
        {
            std::random_device rd{};
            std::mt19937 gen{rd()};
            std::normal_distribution<float> d{0,1};
            m_train->m_cn_projection_matrix.resize(10,3);
            m_train->m_hog_projection_matrix.resize(31,10);
            
            for(int row=0;row<10;++row)
                for(int col=0;col<3;++col)
                    m_train->m_cn_projection_matrix(row,col) = d(gen);
            for(int row=0;row<31;++row)
                for(int col=0;col<10;++col)
                    m_train->m_hog_projection_matrix(row,col) = d(gen);
            break;
        }
        default:
        {
            printf("wrong PROJ_INIT_METHOD !\n");
            exit(-1);
            break; 
        }
    }
    delete[] cn_data; 
    delete[] hog_data;
}

Point2f Tracker::optimize_scores()
{
    // sample_fs 这里要改 暂时 这样代替
    MatrixXcf scores_fs(m_output_sz.height,m_output_sz.width);
    MatrixXcf temp(m_output_sz.height,m_output_sz.width);
    int h1=(m_output_sz.height+1)/2, h2=m_output_sz.height-h1, w1=(m_output_sz.width+1)/2, w2=m_output_sz.width-w1;

    temp.topRows(m_output_sz.height/2+1) = m_scores_fs_sum;
    temp.bottomLeftCorner(m_output_sz.height-m_output_sz.height/2-1,1) = 
            m_scores_fs_sum.middleRows(1,m_output_sz.height-m_output_sz.height/2-1).leftCols(1).colwise().reverse().conjugate();
    temp.bottomRightCorner(m_output_sz.height-m_output_sz.height/2-1,m_output_sz.width-1) = 
            m_scores_fs_sum.middleRows(1,m_output_sz.height-m_output_sz.height/2-1).rightCols(m_output_sz.width-1).colwise().reverse().rowwise().reverse().conjugate();
    
    scores_fs.topLeftCorner(h2,w2) = temp.bottomRightCorner(h2,w2);
    scores_fs.topRightCorner(h2,w1) = temp.bottomLeftCorner(h2,w1);
    scores_fs.bottomLeftCorner(h1,w2) = temp.topRightCorner(h1,w2);
    scores_fs.bottomRightCorner(h1,w1) = temp.topLeftCorner(h1,w1);


    // 暂时 假定 rows cols 相同 都等于 m_scores_fs_sum.cols()
    MatrixXf sampled_scores(m_scores_fs_sum.cols(),m_scores_fs_sum.cols());
    
    Matlab::ifft2(m_scores_fs_sum,sampled_scores,m_scores_fs_sum.cols(),m_scores_fs_sum.cols());
   
    int col,row;
    float init_max_score = sampled_scores.maxCoeff(&row, &col);
    
    int trans_row = ((row + (m_output_sz.height-1)/2)% m_output_sz.height) - (m_output_sz.height-1)/2;
    int trans_col = ((col + (m_output_sz.width-1)/2)% m_output_sz.width) - (m_output_sz.width-1)/2;
    
    float init_pos_y = 2*PI* trans_row/m_output_sz.height;
    float init_pos_x = 2*PI* trans_col/m_output_sz.width;

    float max_pos_y = init_pos_y;
    float max_pos_x = init_pos_x;

    int start = -ceil((m_output_sz.height-1)/2);
    int end = floor((m_output_sz.height-1)/2);
    int num = end-start+1;
    MatrixXf ky(1,end-start+1);
    float *ky_data = ky.data();
    for(int i=0;i<num;++i)
    {
        ky_data[i] = start+i;
    }
    complex<float> imag1(0,max_pos_y);
    MatrixXcf exp_iky = (ky * imag1).array().exp();

    start = -ceil((m_output_sz.width-1)/2);
    end = floor((m_output_sz.width-1)/2);
    num = end-start+1;
    MatrixXf kx(num,1);
    float *kx_data = kx.data();
    for(int i=0;i<num;++i)
    {
        kx_data[i] = start+i;
    }
    complex<float> imag2(0,max_pos_x);
    MatrixXcf exp_ikx = (kx * imag2).array().exp();
    
    MatrixXf ky2 = ky.array().abs2();
    MatrixXf kx2 = kx.array().abs2();

    MatrixXcf ky_exp_ky,kx_exp_kx,y_resp,resp_x,k2_exp;
    complex<float> ival;
    float grad_y,grad_x,H_yy,H_xx,H_xy,det_H;
    complex<float> imag(0,1);
    
    for(int iter=0;iter<config::newton_iterations;++iter)
    {
        // Compute gradient
        ky_exp_ky = ky.array() * exp_iky.array();
        kx_exp_kx = kx.array() * exp_ikx.array();
        y_resp = exp_iky * scores_fs;
        resp_x = scores_fs * exp_ikx;
        grad_y = -(ky_exp_ky*resp_x)(0).imag();
        grad_x = -(y_resp*kx_exp_kx)(0).imag();

        // Compute Hessian
        ival = ((exp_iky*resp_x)*imag)(0);
        k2_exp = ky2.array() * exp_iky.array();
        H_yy = (-(k2_exp * resp_x)(0) + ival).real();
        k2_exp = kx2.array() * exp_ikx.array();
        H_xx = (-(y_resp * k2_exp)(0) + ival).real();
        H_xy = (-(ky_exp_ky * (scores_fs * kx_exp_kx))(0) + ival).real();
        det_H = H_yy * H_xx - H_xy * H_xy;
        
        max_pos_y = max_pos_y - (H_xx * grad_y - H_xy * grad_x) / det_H;
        max_pos_x = max_pos_x - (H_yy * grad_x - H_xy * grad_y) / det_H;

        exp_iky = (ky * (imag * max_pos_y)).array().exp();
        exp_ikx = (kx * (imag * max_pos_x)).array().exp();
    }

    float max_score = (exp_iky *scores_fs * exp_ikx)(0,0).real();

    if(max_score < init_max_score)
    {
        max_score = init_max_score;
        max_pos_y = init_pos_y;
        max_pos_x = init_pos_x;
    }
    max_pos_y += PI;
    max_pos_x += PI;
    while(max_pos_y > 2*PI) 
    {
        max_pos_y -= (2*PI);
    }
    while(max_pos_y < 0)
    {
        max_pos_y += (2*PI);
    }
    while(max_pos_x > 2*PI) 
    {
        max_pos_x -= (2*PI);
    }
    while(max_pos_x < 0)
    {
        max_pos_x += (2*PI);
    }

    float disp_row = (max_pos_y - PI) / (2*PI) * m_output_sz.height;
    float disp_col = (max_pos_x - PI) / (2*PI) * m_output_sz.width;

    return Point2f(disp_col,disp_row);
}
}