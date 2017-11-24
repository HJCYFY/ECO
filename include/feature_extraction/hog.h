#pragma once
/**********************************************************************************
 * @ref hog.h implements the Histogram of Oriented Gradients (HOG) features
 * in the variants of Dalal Triggs @cite{dalal05histograms} and of UOCTTI
 * @cite{felzenszwalb09object}. Applications include object detection
 * and deformable object detection.
 * ********************************************************************************/
#include <opencv2/core.hpp>

#define Debug_Out
// #define SSE_ACC

#define PI 3.141592654f

namespace Feature{

    enum HogVariant { HogVariantDalalTriggs, HogVariantUoctti } ;

    class Hog{
    public:
        Hog(HogVariant _variant, unsigned int _numOrient,unsigned int _cellSize);
        ~Hog();
        void set_use_bilinear_orientation_assignments(bool x);
        void set_use_mem_pool(bool _x);
        void mem_pool_reset(unsigned int _width, unsigned int _height);
        void put_image ( cv::Mat _grayImg);
        void channels(float *H,float* R,int nOrients,float clip,int type);
        // void estract(float * _features);

    protected:
        void prepare_buffers();
        void free_buffers();
        void cal_grad(cv::Mat& img);
        void orient2Bin();
        void trilinear_interpolation();
        void norm_matrix();

        #ifdef Debug_Out
            void show_grad(); 
        #endif
        
        HogVariant variant ;
        unsigned int numOrient ;
        unsigned int cellSize ;
        unsigned int height ;
        unsigned int width ;
        unsigned int imgSize ;
        unsigned int dimension ;
        unsigned int hogWidth ;
        unsigned int hogHeight ;
        unsigned int hogStride ;
        unsigned int maxHeight ;
        unsigned int maxWidth ;

        bool useBilinearOrientationAssigment ;
        bool useMemoryPool ;
        bool full;
        
        float *acos_table ;
        float *acos;
        
        /* buffers */
        float* __attribute__((aligned(16))) hog; 
        float* __attribute__((aligned(16))) hog2 ;
        float* __attribute__((aligned(16))) hogNorm ;
        int16_t* __attribute__((aligned(16))) Img;
        float* __attribute__((aligned(16))) Grad; 
        float* __attribute__((aligned(16))) Orient ;
        float* __attribute__((aligned(16))) Grad0; 
        float* __attribute__((aligned(16))) Grad1 ;
        int32_t* __attribute__((aligned(16))) Orient0; 
        int32_t* __attribute__((aligned(16))) Orient1 ;

        int16_t* __attribute__((aligned(16))) Gradx; 
        int16_t* __attribute__((aligned(16))) Grady; 

        unsigned int col,row;
        unsigned int index;
    };
}