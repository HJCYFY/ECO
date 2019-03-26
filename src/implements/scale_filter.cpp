#include "scale_filter.hpp"
#include "config.hpp"
#include "opencv2/imgproc.hpp"
#include <Eigen/Dense>
#include "fftw3.h"

#include "opencv2/highgui.hpp"
#include "matlab_func.hpp"
namespace Track
{

ScaleFilter::ScaleFilter(Size2f init_sz):bFirstFrame(true)
{
    fhog = make_shared<fHog>(4,9);
    nScales = config::number_of_scales_filter;
    fScaleStep = config::scale_step;
    double scale_step = config::scale_step_filter;
    double sacle_sigma = config::number_of_interp_scales * config::scale_sigma_factor;
    int start = -floor((config::number_of_scales_filter-1)/2);
    int end = ceil((config::number_of_scales_filter-1)/2);
    int length_se = end-start+1;

    double* scale_exp = new double[length_se];
    double tmp = config::number_of_interp_scales/(double)nScales;
    for(int i=0;i<length_se;++i)
    {
        scale_exp[i] = (start+i)*tmp;
    }
    double* scale_exp_shift = new double[length_se];

    // shift w/2
    memcpy(scale_exp_shift,scale_exp-start,(length_se+start)*sizeof(double));
    memcpy(scale_exp_shift+(length_se+start),scale_exp,end*sizeof(double));

    start = -floor((config::number_of_interp_scales-1)/2);
    end = ceil((config::number_of_interp_scales-1)/2);
    int length_ise = end-start+1;
    double* interp_scale_exp = new double[length_ise];
    for(int i=0;i<length_ise;++i)
    {
        interp_scale_exp[i] = (start+i);
    }
    double* interp_scale_exp_shift = new double[length_ise];
    
    // shift w/2
    memcpy(interp_scale_exp_shift,interp_scale_exp-start,(length_ise+start)*sizeof(double));
    memcpy(interp_scale_exp_shift+(length_ise+start),interp_scale_exp,end*sizeof(double));

    sScale.scaleSizeFactors.resize(length_se);
    double* scaleSizeFactors_data = sScale.scaleSizeFactors.data();
    for(int i=0;i<length_se;++i)
        scaleSizeFactors_data[i] = pow(scale_step,scale_exp[i]);

    sScale.interpScaleFactors.resize(length_ise);    
    double* interpScaleFactors_data = sScale.interpScaleFactors.data();
    for(int i=0;i<length_ise;++i)
        interpScaleFactors_data[i] = pow(scale_step,interp_scale_exp_shift[i]);
    
    double *ys = (double*) fftw_malloc(sizeof(double) * length_se);
    fftw_complex *yf = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (length_se/2+1));
    // 计算 fft
    fftw_plan p = fftw_plan_dft_r2c_1d(length_se, ys, yf, FFTW_ESTIMATE);
    for(int i=0;i<length_se;++i)
    {
        ys[i] = exp(-0.5*pow(scale_exp_shift[i],2)/pow(sacle_sigma,2));
    }
    fftw_execute(p); 
    sScale.yf.resize(length_se,length_se/2+1);
    for(int i=0;i<length_se;++i)
        sScale.yf.middleRows(i,1) = Map<MatrixXcd>((complex<double>*)yf,1,length_se/2+1).cast<complex<float> >();

    delete[] scale_exp;
    delete[] scale_exp_shift;
    delete[] interp_scale_exp;
    delete[] interp_scale_exp_shift;
    fftw_destroy_plan(p);
    fftw_free(ys); fftw_free(yf);

    MatrixXf win =  Matlab::hannWin(length_se).cast<float>();
    sScale.window.resize(length_se,length_se);
    for(int i=0;i<win.rows();++i)
        sScale.window.middleCols(i,1).fill(win(i));

    float scale_model_area = config::scale_model_factor *config::scale_model_factor * init_sz.height * init_sz.width;

    if(scale_model_area > config::scale_model_max_area)
        config::scale_model_factor = sqrt(config::scale_model_max_area/(init_sz.height * init_sz.width));
    ScaleModel_sz.height = max(int(init_sz.height*config::scale_model_factor),8);
    ScaleModel_sz.width = max(int(init_sz.width*config::scale_model_factor),8);
    sScale.max_scale_dim = (config::s_num_compressed_dim == -1);
    
    if(sScale.max_scale_dim)
        config::s_num_compressed_dim = length_se;

    fScaleFactors = 1.0f;
}

float ScaleFilter::Track(Mat im, Point2f pos, Size2f target_sz, float currentScaleFactor)
{
    VectorXd scales = sScale.scaleSizeFactors*currentScaleFactor;
    
    MatrixXf xs = Sample(im, pos, target_sz, scales);  
    xs = feature_projection_scale(xs, sScale.basis);
    
    int N = xs.cols();
    MatrixXcf xsf(xs.rows(),N/2+1);
    double  *ptr_in = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex *ptr_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1));
    fftw_plan p = fftw_plan_dft_r2c_1d( N, ptr_in, ptr_out,FFTW_ESTIMATE);
    for(int i=0;i<xs.rows();++i)
    {
        Map<MatrixXd>(ptr_in,1,N) = xs.middleRows(i,1).cast<double>();
        fftw_execute(p);
        xsf.middleRows(i,1) = Map<MatrixXcd>((complex<double>*)ptr_out,1,N/2+1).cast< complex<float> >();
    }
    fftw_destroy_plan(p);
    fftw_free(ptr_in); fftw_free(ptr_out);

    MatrixXcf scale_responsef = (sScale.sf_num.array() * xsf.array()).colwise().sum() / (sScale.sf_den.array() + config::lambda);

    MatrixXf interp_scale_response;
    Matlab::ifft(resizeDFT(scale_responsef, N, config::number_of_interp_scales),interp_scale_response,config::number_of_interp_scales);
    int recovered_scale_index,v;
    interp_scale_response.maxCoeff(&v, &recovered_scale_index);

    float scale_change_factor;
    if(config::do_poly_interp)
    {
        int id1, id2;
        if(recovered_scale_index == 0)
            id1 = config::number_of_interp_scales - 1;
        else
            id1 = recovered_scale_index - 1;
        if(recovered_scale_index == config::number_of_interp_scales-1)
            id2 = 0;
        else
            id2 = recovered_scale_index + 1;

        MatrixXf poly_x(1,3);
        float* poly_x_data = poly_x.data();
        poly_x_data[0] = sScale.interpScaleFactors(id1);
        poly_x_data[1] = sScale.interpScaleFactors(recovered_scale_index);
        poly_x_data[2] = sScale.interpScaleFactors(id2);

        MatrixXf poly_y(3,1);
        float* poly_y_data = poly_y.data();
        poly_y_data[0] = interp_scale_response(id1) / config::number_of_interp_scales;
        poly_y_data[1] = interp_scale_response(recovered_scale_index) / config::number_of_interp_scales;
        poly_y_data[2] = interp_scale_response(id2) / config::number_of_interp_scales;

        MatrixXf poly_A_mat(3,3); 
        float* poly_A_mat_data = poly_A_mat.data();
        poly_A_mat_data[0] = poly_x_data[0]*poly_x_data[0];
        poly_A_mat_data[1] = poly_x_data[1]*poly_x_data[1];
        poly_A_mat_data[2] = poly_x_data[2]*poly_x_data[2];        
        poly_A_mat_data[3] = poly_x_data[0];
        poly_A_mat_data[4] = poly_x_data[1];
        poly_A_mat_data[5] = poly_x_data[2];
        poly_A_mat_data[6] = 1;
        poly_A_mat_data[7] = 1;
        poly_A_mat_data[8] = 1;

        MatrixXf poly = poly_A_mat.inverse() * poly_y;
        scale_change_factor = -poly(1)/(2*poly(0)); 
    }
    else
        scale_change_factor = sScale.interpScaleFactors(recovered_scale_index);
    return scale_change_factor;
}

void ScaleFilter::Update(Mat im, Point2f pos, Size2f target_sz, float currentScaleFactor)
{
    VectorXd scales = sScale.scaleSizeFactors*currentScaleFactor;
    MatrixXf xs = Sample(im, pos, target_sz, scales);
    
    if(bFirstFrame)
    {
        sScale.s_num = xs;
    }
    else
        sScale.s_num =  sScale.s_num * (1-config::scale_learning_rate) + xs * config::scale_learning_rate;

    MatrixXf scale_basis_den;
    if(sScale.max_scale_dim)
    {
        HouseholderQR<MatrixXf> qr;
        qr.compute(sScale.s_num);
        sScale.basis = (qr.householderQ() * MatrixXf::Identity(sScale.s_num.rows(),sScale.s_num.cols())).transpose();
        qr.compute(xs);
        scale_basis_den = (qr.householderQ() * MatrixXf::Identity(xs.rows(),xs.cols())).transpose();
    }
    else
    {
        BDCSVD<MatrixXf> svd;
        svd.compute(sScale.s_num,ComputeThinU);
        sScale.basis = svd.matrixU().leftCols(config::s_num_compressed_dim).adjoint();
    }
    MatrixXf tep = feature_projection_scale(sScale.s_num, sScale.basis);
    int N = tep.cols();
    MatrixXcf sf_proj(tep.rows(),N/2+1);
    double  *ptr_in;
    fftw_complex *ptr_out;
    fftw_plan p;
    ptr_in = (double*) fftw_malloc(sizeof(double) * N);
    ptr_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1));
    p = fftw_plan_dft_r2c_1d( N, ptr_in, ptr_out,FFTW_ESTIMATE);
    for(int i=0;i<tep.rows();++i)
    {
        Map<MatrixXd>(ptr_in,1,N) = tep.middleRows(i,1).cast<double>();
        fftw_execute(p);
        sf_proj.middleRows(i,1) = Map<MatrixXcd>((complex<double>*)ptr_out,1,N/2+1).cast< complex<float> >();
    }
    sScale.sf_num = sScale.yf.array() * sf_proj.conjugate().array();
    
    xs = feature_projection_scale(xs,scale_basis_den);
    MatrixXcf xsf(xs.rows(),N/2+1);
    for(int i=0;i<xs.rows();++i)
    {
        Map<MatrixXd>(ptr_in,1,N) = xs.middleRows(i,1).cast<double>();
        fftw_execute(p);
        xsf.middleRows(i,1) = Map<MatrixXcd>((complex<double>*)ptr_out,1,N/2+1).cast< complex<float> >();
    }
    fftw_destroy_plan(p);
    fftw_free(ptr_in); fftw_free(ptr_out);

    MatrixXf new_sf_den = xsf.array().abs2();

    if(bFirstFrame)
    {
        sScale.sf_den = new_sf_den.colwise().sum();
        bFirstFrame = false;
    }
    else
        sScale.sf_den = (1-config::scale_learning_rate) * sScale.sf_den + config::scale_learning_rate * new_sf_den.colwise().sum();
}

MatrixXf ScaleFilter::Sample(Mat im, Point2f pos, Size2f target_sz, VectorXd& scaleFactor)
{
    int num_scales = scaleFactor.rows();
    int df = floor(scaleFactor.minCoeff());    
    if(df > 1)
    {
        int height = ceil(float(im.rows)/df);
        int width = ceil(float(im.cols)/df);
        int stride = df*3;
        Mat tmp(height,width,CV_8UC3);
        uchar *tmp_data,*im_data;        
        for(int row=0;row<height;++row)
        {
            tmp_data = tmp.ptr<uchar>(row);
            im_data = im.ptr<uchar>(row*df);
            for(int col=0;col<width;++col)
            {
                tmp_data[0] = im_data[0];
                tmp_data[1] = im_data[1];
                tmp_data[2] = im_data[2];
                im_data += stride;
                tmp_data += 3;
            }
        }
        im = tmp;
        pos.x = (pos.x) / df;
        pos.y = (pos.y) / df;
        scaleFactor = scaleFactor / df;
    }
    MatrixXf scale_sample;
    int h, w, half_h, half_w;
    for(int i=0;i<num_scales;++i)
    {
        h = floor(target_sz.height * scaleFactor(i));
        w = floor(target_sz.width * scaleFactor(i));
        half_h = floor(h/2);
        half_w = floor(w/2);

        int minx = floor(pos.x - half_w);//包含
        int maxx = minx + w; //不包含
        int miny = floor(pos.y - half_h);//包含
        int maxy = miny + h;//不包含
        Mat im_patch(h,w,CV_8UC3);
        uchar* im_patch_data = im_patch.data;
        uchar* im_data = im.data;
        // 待优化
        for(int r=miny;r<maxy;++r)
        {
            if(r<0)
                im_data = im.ptr<uchar>(0);
            else if(r>=im.rows)
                im_data = im.ptr<uchar>(im.rows-1);
            else
                im_data = im.ptr<uchar>(r);
            for(int c=minx;c<maxx;++c)
            {
                if(c<0)
                {
                    im_patch_data[0] = im_data[0];
                    im_patch_data[1] = im_data[1];
                    im_patch_data[2] = im_data[2];                    
                }
                else if(c>im.cols-1)
                {
                    im_patch_data[0] = im_data[(im.cols-1)*3];
                    im_patch_data[1] = im_data[(im.cols-1)*3+1];
                    im_patch_data[2] = im_data[(im.cols-1)*3+2];
                }
                else
                {
                    im_patch_data[0] = im_data[c*3];
                    im_patch_data[1] = im_data[c*3+1];
                    im_patch_data[2] = im_data[c*3+2];
                }
                im_patch_data+=3;
            }
        }
        if(h == 0 || w ==0)
        {
            printf("target_sz too small!\n ");
            exit(0);
        }
        cv::resize(im_patch,im_patch,ScaleModel_sz);

        cout<<" ScaleModel_sz "<<ScaleModel_sz<<endl;
        cout<<" target_sz "<<target_sz<<endl;
        cout<<" h "<<h<<" w "<<w<<endl;
        shared_ptr<Feature::Feature> temp_hog;
        temp_hog = fhog->extract(im_patch);
        if(i==0)
            scale_sample.resize(temp_hog->dim1*temp_hog->dim2*temp_hog->dim3,num_scales);
        int N = temp_hog->dim1*temp_hog->dim2*temp_hog->dim3;
        memcpy(scale_sample.data()+i*N,temp_hog->data,sizeof(float)*N );
    }
    return scale_sample;
}

MatrixXf ScaleFilter::feature_projection_scale(const MatrixXf& x, const MatrixXf& projection_matrix)
{
    // cos_win 转 正方形 1×17 变 17×17

    MatrixXf ret = (projection_matrix*x).array() *sScale.window.array();
    return ret;
}

MatrixXcf ScaleFilter::resizeDFT(MatrixXcf& inputdft,int len, int desiredLen)
{
    int minsz = min(len,desiredLen);
    float scaling = (float)desiredLen/len;
    MatrixXcf resizeddft = MatrixXcf::Zero(1,desiredLen/2+1);

    int mids = ceil((float)minsz/2);

    resizeddft.leftCols(mids) = scaling * inputdft.leftCols(mids);
    return resizeddft;
}
}