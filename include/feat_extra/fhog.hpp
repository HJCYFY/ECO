#pragma once
#include "feature.hpp"
#include <string>
#include <memory>
#include "opencv2/core.hpp"

// #define DEBUG
namespace Feature{

    class fHog{
        public:
            fHog(int _binSize,int _nOrients);
            ~fHog();
            
            shared_ptr<Feature> extract(cv::Mat img);
        private:

            void grad1( float *I, float *Gx, float *Gy, int h, int w, int x );
            void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full );
            void hogChannels( float *H, const float *R, const float *N,
                            int hb, int wb, int nOrients, float clip, int type );
            float* hogNormMatrix( float *H, int nOrients, int hb, int wb, int bin );
            void gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
                            int nb, int n, float norm, int nOrients, bool full, bool interpolate );
            void gradHist( float *M, float *O, float *H, int h, int w,
                            int bin, int nOrients, int softBin, bool full );
            void fhog( float *M, float *O, float *H, int h, int w, int binSize,
                            int nOrients, int softBin, float clip );
            int binSize;
            int nOrients;

            float *acos_table;
            float *acost;
    };
}