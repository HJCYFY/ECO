#pragma once
#include "matlab_func.hpp"
#include "fhog.hpp"
#include "type.hpp"
using namespace cv;
using namespace Feature;
using namespace Eigen;

namespace Track
{
struct Scale{
    VectorXd scaleSizeFactors;
    VectorXd interpScaleFactors; 
    MatrixXcf yf;       // half
    bool max_scale_dim;
    MatrixXf window;
    MatrixXf basis;
    MatrixXf s_num;
    MatrixXcf sf_num; // half
    MatrixXf sf_den; // half
};

class ScaleFilter{
public:
    ScaleFilter(Size2f init_sz);
    float Track(Mat im, Point2f pos, Size2f target_sz, float currentScaleFactor);
    void Update(Mat im, Point2f pos, Size2f target_sz, float currentScaleFactor);
    
    bool bFirstFrame;
    int nScales;
    float fScaleStep;
    float fScaleFactors;
    Scale sScale;
    Size ScaleModel_sz;
private:
    MatrixXf Sample(Mat im, Point2f pos, Size2f target_sz, VectorXd& scales);
    MatrixXf feature_projection_scale(const MatrixXf& x,const MatrixXf& projection_matrix);
    MatrixXcf resizeDFT(MatrixXcf& inputdft, int len, int desiredLen);
    shared_ptr<fHog> fhog;    
};

}