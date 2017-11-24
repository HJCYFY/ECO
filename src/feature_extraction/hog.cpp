#include "feature_extraction/hog.h"
#include <cassert>
#include "eco_assert.h"
#include <math.h>

#ifdef Debug_Out
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#endif

namespace Feature{

Hog::Hog(HogVariant _variant, unsigned int _nOrient,unsigned int _cellSize):
            variant(_variant),numOrient(_nOrient),cellSize(_cellSize),
            maxHeight(0),maxWidth(0),useBilinearOrientationAssigment(false),
            useMemoryPool(false),full(true),acos_table(NULL),acos(NULL),hog(NULL),
            hog2(NULL),hogNorm(NULL),Img(NULL),Grad(NULL),Orient(NULL),Grad0(NULL),
            Grad1(NULL),Orient0(NULL),Orient1(NULL),Gradx(NULL),Grady(NULL)
{
    const long int n=1000;
    const long int b=1;
    acos_table = new float[2*(n+b)];
    acos = acos_table+n+b;
    int i;
    for( i=-n-b; i<-n; i++ )   acos[i]=PI;
    for( i=-n; i<n; i++ )      acos[i]=std::acos(i/float(n));
    for( i=n; i<n+b; i++ )     acos[i]=0;

    switch (variant) {
        case HogVariantUoctti:
            dimension = 3*numOrient + 4 ;
            break ;
        case HogVariantDalalTriggs:
            dimension = 4*numOrient ;
            break ;
        default:
            ENSURE(0) ;
            break;
    }    
}

Hog::~Hog()
{
    delete[] acos_table;
    acos_table = NULL;

    if(hog)
    free_buffers();
}


void Hog::set_use_bilinear_orientation_assignments(bool _x)
{
    useBilinearOrientationAssigment = _x;
}

void Hog::set_use_mem_pool(bool _x)
{
    useMemoryPool = _x;
}

void Hog::mem_pool_reset(unsigned int _width, unsigned int _height)
{
    maxHeight = _height;
    maxWidth = _width;
    unsigned int maxImgSize = maxHeight*maxWidth;

    unsigned int maxHogWidth = (width + cellSize/2) / cellSize ;
    unsigned int maxHogHeight = (height + cellSize/2) / cellSize ;
    unsigned int maxHogSize = maxHogWidth * maxHogHeight ;
    ENSURE(maxImgSize>0);
    if(hog)
        free_buffers();

    hog = new float[maxHogSize * numOrient * 2]; 
    hog2 = new float[maxHogSize* numOrient];
    hogNorm = new float[maxHogSize];
    Img = new int16_t[maxImgSize];
    Grad = new float[maxImgSize]; 
    Orient = new float[maxImgSize];
    Grad0 = new float[maxHogSize]; 
    Grad1 = new float[maxHogSize];
    Orient0 = new int32_t[maxHogSize]; 
    Orient1 = new int32_t[maxHogSize];

    Gradx = new int16_t[maxImgSize];
    Grady = new int16_t[maxImgSize];

}

void Hog::prepare_buffers()
{    
    hog = new float[hogStride * numOrient * 2]; 
    hog2 = new float[hogStride * numOrient];
    hogNorm = new float[hogStride];

    memset(hog,0,sizeof(float)*hogStride * numOrient * 2);
    memset(hogNorm,0,sizeof(float)*hogStride);

    Img = new int16_t[imgSize];
    Grad = new float[imgSize]; 
    Orient = new float[imgSize];
    Grad0 = new float[imgSize]; 
    Grad1 = new float[imgSize];
    Orient0 = new int32_t[imgSize]; 
    Orient1 = new int32_t[imgSize];

    Gradx = new int16_t[imgSize];
    Grady = new int16_t[imgSize];
}

void Hog::free_buffers()
{
    delete[]  hog; 
    hog = NULL;
    delete[] hog2;
    hog2 = NULL;
    delete[] hogNorm;
    hogNorm = NULL;
    delete[] Img;
    Img = NULL;
    delete[] Grad; 
    Grad = NULL;
    delete[] Orient;
    Orient = NULL;
    delete[] Grad0; 
    Grad0 = NULL;
    delete[] Grad1;
    Grad1 = NULL;
    delete[] Orient0; 
    Orient0 = NULL;
    delete[] Orient1;
    Orient1 = NULL;
    delete[] Gradx;
    Gradx = NULL;
    delete[] Grady;
    Grady = NULL;
}

void Hog::cal_grad(cv::Mat& img)
{
    #ifdef SSE_ACC
        uchar* data = img.data;
        const unsigned int aliquot = imgSize/16;  
        const unsigned int mod = imgSize%16;
        const unsigned int aliquot8 = imgSize/8;  
        const unsigned int mod8 = imgSize%8;
        const unsigned int aliqW16 = width/16;
        const unsigned int modW16 = width%16;
        const unsigned int aliqW8 = width/8;
        const unsigned int modW8 = width%8;
        __m128i ucharData;
        __m128i zeros = _mm_setzero_si128();    

        // 将图像转为 int16类型
        if(img.isContinuous())
        {
            int16_t* out = Img;
            for(index=0; index<aliquot; ++index)
            {
                ucharData =  _mm_load_si128((__m128i *)data); 
                _mm_store_si128((__m128i*)out,_mm_unpacklo_epi8(ucharData, zeros));
                out+=8;
                _mm_store_si128((__m128i*)out,_mm_unpackhi_epi8(ucharData, zeros));
                out+=8;
                data+=16;
            }
            for(index=0; index<mod ;++index)
            {
                *out = *data;
                ++data;   ++out;
            }        
        }
        else
        {
            int16_t* out = Img;
            for(row=0; row<height; ++row)
            {
                data = img.ptr<uchar>(row);
                for(col =0;col<aliqW16;++col)
                {
                    ucharData = _mm_loadu_si128((__m128i *)data); 
                    _mm_storeu_si128((__m128i*)out,_mm_unpacklo_epi8(ucharData, zeros));
                    out+=8;
                    _mm_storeu_si128((__m128i*)out,_mm_unpackhi_epi8(ucharData, zeros));
                    out+=8;
                    data+=16;
                }
                for(col=0;col<modW16;++col)
                {
                    *out = *data;
                    ++data;   ++out;
                }
            }
        }        
        __m128i Two = _mm_set1_epi16(2);
        {   /*   y 方向梯度   */
            int16_t* row0Data ;
            int16_t* row1Data ;
            int16_t* out; 
            {   /*  row 0  */
                row0Data = Img;
                row1Data = Img+width;
                out = Grady;
                for(col=0;col<aliqW8;++col)
                {
                    _mm_storeu_si128((__m128i*)out, _mm_mullo_epi16(Two,_mm_subs_epi16(_mm_loadu_si128((__m128i*)row1Data),_mm_loadu_si128((__m128i*)row0Data))));
                    row0Data+=8;
                    row1Data+=8;
                    out+=8;
                }
                for(col=0;col<modW8;++col)
                {
                    *out = (*row1Data-*row0Data)*2;
                    ++out;  ++row1Data;   ++row0Data;
                }
            }
            {   /*  row h-1  */
                row0Data = Img+(height-2)*width;
                row1Data = Img+(height-1)*width;
                out = Grady+(height-1)*width;

                for(col=0;col<aliqW8;++col)
                {
                    _mm_storeu_si128((__m128i*)out, _mm_mullo_epi16(Two,_mm_subs_epi16(_mm_loadu_si128((__m128i*)row1Data),_mm_loadu_si128((__m128i*)row0Data))));
                    row0Data+=8;
                    row1Data+=8;
                    out+=8;
                }
                for(col=0;col<modW8;++col)
                {
                    *out = (*row1Data-*row0Data)*2;
                    ++out;  ++row1Data;   ++row0Data;
                }
            }
            {   /* row 1 to row h-1  */
                row0Data = Img;
                row1Data = Img+2*width;
                out = Grady+width;
                const long unsigned int aliqM = width*(height-2)/8;
                const long unsigned int modM = width*(height-2)%8;
                for(row=0;row<aliqM;++row)
                {
                    _mm_storeu_si128((__m128i*)out, _mm_subs_epi16(_mm_loadu_si128((__m128i*)row1Data),_mm_loadu_si128((__m128i*)row0Data)));
                    row0Data+=8;
                    row1Data+=8;
                    out+=8;
                }
                for(col=0;col<modM;++col)
                {
                    *out = (*row1Data-*row0Data);
                    ++out;  ++row1Data;   ++row0Data;
                }
            }
        }

        {   /*  x 方向梯度  */  
            int16_t* col0Data ;
            int16_t* col1Data ;
            int16_t* out; 
            {   //  col 0   
                for(row=0;row<height;++row)
                {
                    col0Data = Img+row*width;
                    col1Data = col0Data+1;
                    out = Gradx+row*width;
                    *out = (*col1Data - *col0Data)*2;                
                }
            }
            {   //  col w-1   
                for(row=0;row<height;++row)
                {
                    col0Data = Img+(row+1)*width-2;
                    col1Data = col0Data+1;
                    out = Gradx+(row+1)*width-1;
                    *out = (*col1Data - *col0Data)*2;                
                }
            }
            {   //  col 1 to  w-1   
                const long unsigned int aliqM = (width-2)/8;
                const long unsigned int modM = (width-2)%8;
                for(row=0;row<height;++row)
                {                
                    col0Data = Img+row*width;
                    col1Data = col0Data+2; 
                    out = Gradx+row*width+1;
                    for(col=0;col<aliqM;++col)    
                    {                    
                        _mm_storeu_si128((__m128i*)out,_mm_subs_epi16(_mm_loadu_si128((__m128i*)col1Data),_mm_loadu_si128((__m128i*)col0Data)));    
                        col0Data+=8;
                        col1Data+=8;
                        out+=8;
                    }                     
                    for(col=0;col<modM;++col)
                    {
                        *out = (*col1Data-*col0Data);
                        ++out;  ++col1Data;   ++col0Data;
                    }   
                }
            }             
        }
        {   /*  计算 Grad 和  Orient   */
            
            int16_t* gx =  Gradx;
            int16_t* gy =  Grady;
            float* grad = Grad;
            float* orient = Orient;
            __m128i GxData,GyData,GxInt32,GyInt32;
            __m128 GxFloat,GyFloat,Gradient;
            __m128 half = _mm_set1_ps(0.5);
            __m128 max = _mm_set1_ps(1e10f);
            __m128 num = _mm_set1_ps(1e3f);
            float ori[4];
    
            for(index=0;index<aliquot8;++index)
            {
                GxData = _mm_load_si128((__m128i*)gx); 
                GyData = _mm_load_si128((__m128i*)gy); 
                // 低四位 转 int32
                GxInt32 = _mm_srai_epi32(_mm_unpacklo_epi16(zeros, GxData), 16);
                GyInt32 = _mm_srai_epi32(_mm_unpacklo_epi16(zeros, GyData), 16);
                // 转 float            
                GxFloat = _mm_mul_ps(_mm_cvtepi32_ps(GxInt32),half);  
                GyFloat = _mm_mul_ps(_mm_cvtepi32_ps(GyInt32),half);
    
                // 计算 幅度 平方和再开方
                Gradient=_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(GxFloat,GxFloat),_mm_mul_ps(GyFloat,GyFloat)));
                // 存储
                _mm_store_ps(grad,Gradient);
                grad+=4;
                // 梯度求倒数  方向归一化到 -1000 ～ 1000 存储到ori
                _mm_storeu_ps(ori,_mm_mul_ps( _mm_mul_ps(GxFloat,_mm_min_ps(_mm_rcp_ps(Gradient),max)), num ));
    
                for(int i=0;i<4;++i)
                {
                    *orient = acos[(int)ori[i]];
                    ++orient;                
                }
                        
                // 高四位 转 int32
                GxInt32 = _mm_srai_epi32(_mm_unpackhi_epi16(zeros, GxData), 16);
                GyInt32 = _mm_srai_epi32(_mm_unpackhi_epi16(zeros, GyData), 16);
                // 转 float            
                GxFloat = _mm_mul_ps(_mm_cvtepi32_ps(GxInt32),half);  
                GyFloat = _mm_mul_ps(_mm_cvtepi32_ps(GyInt32),half);  
        
                // 计算 幅度 平方和再开方
                Gradient=_mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(GxFloat,GxFloat),_mm_mul_ps(GyFloat,GyFloat)));
                _mm_store_ps(grad,Gradient);
                grad+=4;
                // 梯度求倒数  方向归一化到 -1000 ～ 1000 存储到ori
                _mm_storeu_ps(ori,_mm_mul_ps( _mm_mul_ps(GxFloat,_mm_min_ps(_mm_rcp_ps(Gradient),max)), num ));
    
                for(int i=0;i<4;++i)
                {
                    *orient = acos[(int)ori[i]];
                    ++orient;
                }
                gx+=8;
                gy+=8;
            }
            for(index=0;index<mod8;++index)
            {
                *grad = sqrt((*gx)*(*gx)+(*gy)*(*gy))*0.5;    
                *orient = acos[(int)(((*gx)*std::min(1.0f/(*grad),1e10f))*1000.f)];
                ++gx;  ++gy;  ++grad;   ++orient;
            }   
        } 
        if(full)
        {
            int16_t* gy =  Grady;
            float* orinet = Orient;
            __m128i GyData;    
            __m128i _m128i_Zero = _mm_set1_epi32(0);
            __m128 _m128_PI = _mm_set1_ps(PI);
    
            for(index=0;index<aliquot8;++index)
            {
                GyData = _mm_load_si128((__m128i*)gy); 
                _mm_store_ps(orinet,_mm_add_ps( _mm_load_ps(orinet), _mm_and_ps( _mm_castsi128_ps(_mm_cmplt_epi32(_mm_srai_epi32(_mm_unpacklo_epi16(zeros, GyData), 16),_m128i_Zero)) ,_m128_PI) ));    
                orinet+=4;
                _mm_store_ps(orinet,_mm_add_ps( _mm_load_ps(orinet), _mm_and_ps( _mm_castsi128_ps(_mm_cmplt_epi32(_mm_srai_epi32(_mm_unpackhi_epi16(zeros, GyData), 16),_m128i_Zero)) ,_m128_PI) ));
                orinet+=4;
                gy+=8;
            }
            for(index=0;index<mod8;++index)
            {
                *orinet += ((*gy) <0)*PI;
                ++orinet;
                ++gy;
            }
        }

    #else
        // std::cout<<"HHH"<<std::endl;
        uchar* imgData = img.data;
        int16_t* data = Img;
        if(img.isContinuous())
        {
            for(index=0; index<imgSize; ++index)
            {
                *data = *imgData;
                ++imgData;   ++data;
            } 
        }
        else
        {
            for(row=0; row<height; ++row)
            {
                imgData = img.ptr<uchar>(row);
                for(col=0;col<width;++col)
                {
                    *data = *imgData;
                    ++imgData;   ++data;
                }
            }
        }  
        data = Img;
        int16_t *gx = Gradx;
        int16_t *gy = Grady;
        float* grad = Grad;
        float* orient = Orient;
        unsigned int rw;
        // 第一行 第一列
        gx[0] = (data[1] - data[0])*2;
        gy[0] = (data[width] - data[0])*2;
        grad[0] = sqrt(gx[0]*gx[0]+gy[0]*gy[0])/2;
        orient[0] = acos[(int)((gx[0]*0.5f*std::min(1.0f/grad[0],1e10f))*1000.f)];

        // 第一行 中间
        for (col = 1; col<width-1; ++col) 
        {
            gx[col] = data[col+1] - data[col-1];
            gy[col] = (data[col+width] - data[col])*2;
            grad[col] = sqrt(gx[col]*gx[col]+gy[col]*gy[col])/2;
            orient[col] = acos[(int)((gx[col]*0.5f*std::min(1.0f/grad[col],1e10f))*1000.f)];
        }
        // 第一行 最后
        gx[width-1] =  (data[width-1] - data[width-2])*2;
        gy[width-1] =  (data[width-1+width] - data[width-1])*2;
        grad[width-1] = sqrt(gx[width-1]*gx[width-1]+gy[width-1]*gy[width-1])/2;
        orient[width-1] = acos[(int)((gx[width-1]*0.5f*std::min(1.0f/grad[width-1],1e10f))*1000.f)];


        for (row = 1 ; row < height - 1 ; ++row) 
        {
            // 第一列
            rw = row*width;
            gx[rw] =  (data[rw+1] - data[rw])*2;
            gy[rw] =  (data[rw+width] - data[rw-width]);
            grad[rw] = sqrt(gx[rw]*gx[rw]+gy[rw]*gy[rw])/2;
            orient[rw] = acos[(int)((gx[rw]*0.5f*std::min(1.0f/grad[rw],1e10f))*1000.f)];
            for (col = 1 ; col < width - 1 ; ++col) 
            {
        
                gx[rw+col] = data[rw+col+1] - data[rw+col-1];
                gy[rw+col] = data[rw+col+width] - data[rw+col-width];   
                grad[rw+col] = sqrt(gx[rw+col]*gx[rw+col]+gy[rw+col]*gy[rw+col])/2;
                orient[rw+col] = acos[(int)((gx[rw+col]*0.5f*std::min(1.0f/grad[rw+col],1e10f))*1000.f)];         
            }
            // 最后一列
            rw = (row+1)*width-1;
            gx[rw] = (data[rw] - data[rw-1])*2;
            gy[rw] = data[rw+width] - data[rw-width];  
            grad[rw] = sqrt(gx[rw]*gx[rw]+gy[rw]*gy[rw])/2; 
            orient[rw] = acos[(int)((gx[rw]*0.5f*std::min(1.0f/grad[rw],1e10f))*1000.f)]; 
        }
        //最后一行 第一列
        rw = (height-1)*width;
        gx[rw] = (data[rw+1] - data[rw])*2;
        gy[rw] = (data[rw] - data[rw-width])*2; 
        grad[rw] = sqrt(gx[rw]*gx[rw]+gy[rw]*gy[rw])/2;
        orient[rw] = acos[(int)((gx[rw]*0.5f*std::min(1.0f/grad[rw],1e10f))*1000.f)];
        // 最后一行 中间
        for (col = 1; col<width-1; ++col) 
        {
            gx[rw+col] = data[rw+col+1] - data[rw+col-1];
            gy[rw+col] = (data[rw+col] - data[rw+col-width])*2;
            grad[rw+col] = sqrt(gx[rw+col]*gx[rw+col]+gy[rw+col]*gy[rw+col])/2;
            orient[rw+col] = acos[(int)((gx[rw+col]*0.5f*std::min(1.0f/grad[rw+col],1e10f))*1000.f)];
        } 
        // 最后一行 最后 
        rw = height*width-1;
        gx[rw] = (data[rw] - data[rw-1])*2;
        gy[rw] = (data[rw] - data[rw-width])*2;
        grad[rw] = sqrt(gx[rw]*gx[rw]+gy[rw]*gy[rw])/2;
        orient[rw] = acos[(int)((gx[rw]*0.5f*std::min(1.0f/grad[rw],1e10f))*1000.f)];

        if(full)
        {
            int16_t* gy =  Grady;
            float* orinet = Orient;
            for(index=0;index<imgSize;++index)
            {
                *orinet += ((*gy) <0)*PI;
                ++orinet;
                ++gy;
            }
        }
    #endif
}

void Hog::orient2Bin()
{
    int nOrients;
    if(variant == HogVariantDalalTriggs)
        nOrients = numOrient;
    else 
        nOrients = numOrient*2;
    const float _float_norm = 1.0f/cellSize/cellSize;    
    const float oMult=(float)nOrients/(full ? 2*PI:PI); 

    int _int_o0,_int_o1;
    float _float_o, _float_od, _float_m;

    float *p_Grad = Grad,*p_Orient = Orient,
            *p_M0 = Grad0,*p_M1 = Grad1;
    int32_t *p_o0 = Orient0,*p_o1 = Orient1;

    #ifdef SSE_ACC
        const unsigned int aliquot = imgSize/4;
        const __m128 _m128_oMult=_mm_set1_ps(oMult);
        const __m128i _m128i_oMax=_mm_set1_epi32(nOrients);
        const __m128i _m128i_One=_mm_set1_epi32(1);
        const __m128i _m128i_Zero=_mm_set1_epi32(0);
        const __m128 _m128_Zero=_mm_set1_ps(0.f);
        const __m128 half=_mm_set1_ps(0.5f);
        const __m128 _m128_norm=_mm_set1_ps(_float_norm);

        __m128i _128i_o0, _128i_o1;
        __m128 _128_o, _128_od, _128_m, _128_md;

        if(useBilinearOrientationAssigment)
        {
            for( index=0;index<aliquot;++index) 
            {
                _128_o=_mm_mul_ps(_mm_load_ps(p_Orient),_m128_oMult);
                // 取整数部分
                _128i_o0=_mm_cvttps_epi32(_128_o); 
                // 小数部分
                _128_od=_mm_sub_ps(_128_o,_mm_cvtepi32_ps(_128i_o0));
                // 给O0赋值 
                _mm_store_si128((__m128i*)p_o0,_128i_o0);
                _128i_o1=_mm_add_epi32(_128i_o0,_m128i_One);
                _128i_o1=_mm_and_si128(_mm_cmplt_epi32(_128i_o1,_m128i_oMax),_128i_o1);
                // 给O1赋值 
                _mm_store_si128((__m128i*)p_o1,_128i_o1);
                 // 幅度除以Cell面积
                _128_m =_mm_mul_ps(_mm_load_ps(p_Grad),_m128_norm); 
                _128_md =_mm_mul_ps(_128_od,_128_m); 
                _mm_store_ps(p_M1,_128_md);
                _mm_store_ps(p_M0,_mm_sub_ps(_128_m,_128_md));

                p_Orient+=4;
                p_Grad+=4;
                p_o0+=4;
                p_o1+=4;
                p_M0+=4;
                p_M1+=4;
            }
            for(index = index*4;index<imgSize;++index)
            {
                _float_o = (*p_Orient)*oMult;
                _int_o0=(int) _float_o;  
                _float_od=_float_o-_int_o0;
                *p_o0 = _int_o0;
                _int_o1=_int_o0+1; 
                if(_int_o1==nOrients) 
                    _int_o1=0; 
                *p_o1=_int_o1;
                _float_m = (*p_Grad) * _float_norm; 
                *p_M1=_float_od*_float_m; 
                *p_M0=_float_m - *p_M1;

                p_Orient+=1;
                p_Grad+=1;
                p_o0+=1;
                p_o1+=1;
                p_M0+=1;
                p_M1+=1;
            }
        }
        else
        { 
            for( index=0;index<aliquot;++index) 
            { 
                _128_o=_mm_mul_ps(_mm_load_ps(p_Orient) ,_m128_oMult); 
                _128i_o0=_mm_cvttps_epi32(_mm_add_ps(_128_o,half));
                _128i_o0=_mm_and_si128(_mm_cmplt_epi32(_128i_o0,_m128i_oMax),_128i_o0); 
                _mm_store_si128((__m128i*)p_o0,_128i_o0);
                _mm_store_si128((__m128i*)p_o1,_m128i_Zero);
                _128_m=_mm_mul_ps(_mm_load_ps(p_Grad),_m128_norm); 
                _mm_store_ps(p_M0,_128_m);                
                _mm_store_ps(p_M1,_m128_Zero);
    
                p_Orient+=4;
                p_Grad+=4;
                p_o0+=4;
                p_o1+=4;
                p_M0+=4;
                p_M1+=4;
            }
            for(index = index*4;index<imgSize;++index)
            {                
                _float_o = *p_Orient*oMult; 
                _int_o0=(int) (_float_o+.5f);
                if(_int_o0>=nOrients) 
                    _int_o0=0; 
                *p_o0 = _int_o0;
                *p_o1=0;
                *p_M0 = *p_Grad*_float_norm; 
                *p_M1=0; 

                p_Orient+=1;
                p_Grad+=1;
                p_o0+=1;
                p_o1+=1;
                p_M0+=1;
                p_M1+=1;
            }
        }
    #else
        if(useBilinearOrientationAssigment)
        {
            for(index=0;index<imgSize;++index)
            {
                _float_o = p_Orient[index]*oMult;
                _int_o0=(int) _float_o;  
                _float_od=_float_o-_int_o0;
                p_o0[index] = _int_o0;
                _int_o1=_int_o0+1; 
                if(_int_o1==nOrients) 
                    _int_o1=0; 
                p_o1[index]=_int_o1;
                _float_m=p_Grad[index]*_float_norm; 
                p_M1[index]=_float_od*_float_m; 
                p_M0[index]=_float_m-p_M1[index];
            }
        }
        else 
        {
            // std::cout<<"AAA"<<std::endl;
            for(index=0;index<imgSize;++index)
            { 
                _float_o = p_Orient[index]*oMult; 
                _int_o0=(int) (_float_o+.5f);
                if(_int_o0>=nOrients) 
                    _int_o0=0; 
                p_o0[index]=_int_o0;
                p_M0[index]=p_Grad[index]*_float_norm; 
                p_M1[index]=0; 
                p_o1[index]=0;
            }
        }
    #endif
}

void Hog::trilinear_interpolation()
{
    float hy;
    int biny;
    float hx;
    float wy1,wy2;
    float weight;
    int32_t* p_o0 = Orient0;
    int32_t* p_o1 = Orient1;
    float* p_M0 = Grad0;
    float* p_M1 = Grad1;
    #define at(x,y,k) (hog[(x) + (y) * hogWidth + (k) * hogStride])    

    #ifdef SSE_ACC
        float* __attribute__((aligned(16)))  wx1 = new float[4];
        float* __attribute__((aligned(16)))  wx2 = new float[4];
        int32_t*  __attribute__((aligned(16))) binx = new int32_t[4];
        __m128 half = _mm_set1_ps(0.5f);
        __m128 _128_1_bin = _mm_set1_ps(1.0f/cellSize);
        __m128 _m128_One = _mm_set1_ps(1.0f);
        const unsigned int aliqW4 = (width/4)*4;

        __m128 _m128_hx,_m128_wx2;
        __m128i _m128i_binx;

        for(row=0; row<height; ++row) 
        {
            hy = (row+0.5f)/cellSize-0.5f;
            biny = floor(hy);
            wy2 = hy - biny ;
            wy1 = 1.0 - wy2 ;
            for(col=0; col<aliqW4; col+=4)
            {
                _m128_hx = _mm_set_ps(float (col+3), float (col+2), float (col+1), float (col));
                _m128_hx = _mm_sub_ps(_mm_mul_ps(_mm_add_ps(_m128_hx,half),_128_1_bin),half);
                _m128i_binx = _mm_cvtps_epi32(_mm_sub_ps(_m128_hx,half));
                _mm_store_si128((__m128i*)binx,_m128i_binx);
                _m128_wx2 = _mm_sub_ps(_m128_hx,_mm_cvtepi32_ps(_m128i_binx));
                _mm_store_ps(wx2, _m128_wx2);
                _mm_store_ps(wx1, _mm_sub_ps(_m128_One,_m128_wx2));

                for(int i=0;i<4;++i)
                {               
                    if(binx[i] >= 0 && biny >=0)
                    {
                        weight = wx1[i] * wy1;
                        at(binx[i],biny,p_o0[i]) += p_M0[i] * weight ;
                        at(binx[i],biny,p_o1[i]) += p_M1[i] * weight ;
                    }
                    if(binx[i] < (signed)hogWidth - 1 && biny >=0)
                    {
                        weight = wx2[i] * wy1;
                        at(binx[i]+1,biny,p_o0[i]) += p_M0[i] * weight ;
                        at(binx[i]+1,biny,p_o1[i]) += p_M1[i] * weight ;
                    }
                    if (binx[i] < (signed)hogWidth - 1 && biny < (signed)hogHeight - 1) 
                    {
                        weight = wx2[i] * wy2;
                        at(binx[i]+1,biny+1,p_o0[i]) += p_M0[i] * weight ;
                        at(binx[i]+1,biny+1,p_o1[i]) += p_M1[i] * weight ;
                    }
                    if (binx[i] >= 0 && biny < (signed)hogHeight - 1) 
                    {
                        weight = wx1[i] * wy2;
                        at(binx[i],biny+1,p_o0[i]) += p_M0[i] * weight ;
                        at(binx[i],biny+1,p_o1[i]) += p_M1[i] * weight ;
                    }
                }
                p_o0 +=4;                p_o1 +=4;
                p_M0 +=4;                p_M1 +=4;
            }
            for(; col<width; ++col)
            {
                hx = (col + 0.5) / cellSize - 0.5 ;
                binx[0] = floor(hx) ;
                wx2[0] = hx - binx[0] ;
                wx1[0] = 1.0 - wx2[0] ;
                if(binx[0] >= 0 && biny >=0)
                {
                    weight = wx1[0] * wy1;
                    at(binx[0],biny,*p_o0) += *p_M0 * weight ;
                    at(binx[0],biny,*p_o1) += *p_M1 * weight ;
                }
                if(binx[0] < (signed)hogWidth - 1 && biny >=0)
                {
                    weight = wx2[0] * wy1;
                    at(binx[0]+1,biny,*p_o0) += *p_M0 * weight ;
                    at(binx[0]+1,biny,*p_o1) += *p_M1 * weight ;
                }
                if(binx[0] < (signed)hogWidth - 1 && biny < (signed)hogHeight - 1) 
                {
                    weight = wx2[0] * wy2;
                    at(binx[0]+1,biny+1,*p_o0) += *p_M0 * weight ;
                    at(binx[0]+1,biny+1,*p_o1) += *p_M1 * weight ;
                }
                if (binx[0] >= 0 && biny < (signed)hogHeight - 1) 
                {
                    weight = wx1[0] * wy2;
                    at(binx[0],biny+1,*p_o0) += *p_M0 * weight ;
                    at(binx[0],biny+1,*p_o1) += *p_M1 * weight ;
                }
                ++p_M0;                ++p_M1;     
                ++p_o0;                ++p_o1;                
            }
        }
        delete[] wx1;
        delete[] wx2;
        delete[] binx;
    #else
        int binx;
        float wx1,wx2;
        for(row=0; row<height; ++row) 
        {
            hy = (row+0.5f)/cellSize-0.5f;
            biny = floor(hy);
            wy2 = hy - biny ;
            wy1 = 1.0 - wy2 ;
            
            for(col=0; col<width; ++col)
            {
                hx = (col + 0.5) / cellSize - 0.5 ;
                binx = floor(hx) ;
                wx2 = hx - binx ;
                wx1 = 1.0 - wx2 ;

                if (binx >= 0 && biny >=0) {
                    weight =  wx1 * wy1;
                    at(binx,biny,*p_o0) += *p_M0 * weight;
                    at(binx,biny,*p_o1) += *p_M1 * weight;
                }
                if (binx < (signed)hogWidth-1 && biny >=0) {
                    weight = wx2 * wy1;
                    at(binx+1,biny,*p_o0) += *p_M0 * weight;
                    at(binx+1,biny,*p_o1) += *p_M1 * weight;
                }
                if (binx < (signed)hogWidth-1 && biny < (signed)hogHeight-1) {
                    weight = wx2 * wy2;
                    at(binx+1,biny+1,*p_o0) += *p_M0 * weight;
                    at(binx+1,biny+1,*p_o1) += *p_M1 * weight;
                }
                if (binx >= 0 && biny < (signed)hogHeight - 1) {
                    weight = wx1 * wy2;
                    at(binx,biny+1,*p_o0) += *p_M0 * weight;
                    at(binx,biny+1,*p_o1) += *p_M1 * weight;
                }
                ++p_M0;                ++p_M1;     
                ++p_o0;                ++p_o1;     
            }
        }
    #endif
}

void Hog::norm_matrix()
{
    const float eps = 1e-4f/4/cellSize/cellSize/cellSize/cellSize; 
    float* N1 = hogNorm;
    int o;
    #ifdef SSE_ACC
        /****计算平方和****************/
        unsigned int aliqA = (hogStride/4)*4;
        __m128 _m128_H;
        // [1,nOrients) 方向      
        float *H_nOri;  
        for( o=0; o<numOrient; o++) 
        {
            N1 = hogNorm;
            H_nOri = hog2 + o * hogStride;
            for(index=0;index<aliqA;index+=4)
            {
                _m128_H = _mm_loadu_ps(H_nOri);
                _mm_store_ps(N1,_mm_add_ps( _mm_load_ps(N1), _mm_mul_ps(_m128_H,_m128_H)));
                N1+=4;
                H_nOri+=4;
            }
            for(;index<hogStride;++index)
            {
                *N1 = *H_nOri* *H_nOri;
                ++N1;
            }
        }
    #else
        for( o=0; o<numOrient; o++ ) 
        {
            int owh = o*hogStride;
            for( index=0; index<hogStride; index++ ) 
                N1[index] += hog2[owh+index]*hog2[owh+index];
        }
    #endif
    // 四个方格求和再根号
    float  *n , *n1; 
    for( row=0; row<hogHeight-1; row++ ) 
        for( col=0; col<hogWidth-1; col++ ) 
        {
            n=hogNorm+row*hogWidth+col; 
            *n=1/float(sqrt(n[0]+n[1]+n[hogWidth]+n[hogWidth+1]+eps)); 
        }
    n = hogNorm+ (hogHeight-1)*hogWidth;
    n1 = hogNorm+ (hogHeight-2)*hogWidth;
    memcpy(n,n1,sizeof(float)*(hogWidth-1));

    for(row=0;row<hogHeight-1;++row)
    {
        n = hogNorm+(row+1)*hogWidth-1;
        n1 = hogNorm+(row+1)*hogWidth-2;
        *n = *n1;
    }
    n = hogNorm+hogStride-1;
    n1 = hogNorm+hogStride -hogWidth -2;
    *n = *n1;
}



void Hog::channels( float *H, float *R, int nOrients, float clip, int type)
{
    const float r=.2357f; 
    int o; 
    float t;
    const int nBins=nOrients*hogStride;
    int aliqA = (hogStride/4)*4;
    float *H1, *R1, *N1;
    #ifdef SSE_ACC
    {
        int aliqA1 = hogWidth*(hogHeight-1)/4;
        aliqA1= aliqA1*4 + hogWidth;
        __m128 _m128_Mul,_m128_Cmp;
        __m128 _m128_clip = _mm_set1_ps(clip);
        __m128 half = _mm_set1_ps(0.5f);
        __m128 _m128_r = _mm_set1_ps(r);
        if( type==0) 
        {
            for( o=0; o<nOrients; o++ ) 
            {
                {   // 0
                    H1 = H+o*hogStride;
                    R1 = R+o*hogStride;
                    N1 = hogNorm;
                    for(index =0;index<aliqA;index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;
                    }
                    for(;index<hogStride;++index)
                    {
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;
                    }
                }
                {   // 1
                    // 第一行特殊处理
                    H1 = H+nBins+o*hogStride;
                    R1 = R+o*hogStride;
                    N1 = hogNorm;
                    for(index=0; index+4<=hogWidth; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<hogWidth; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;                      
                    }
                    // 其余行
                    N1 = hogNorm;
                    for(; index<aliqA1; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }                    
                    for(; index<hogStride; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;                      
                    }

                }
                {   // 2
                    H1 = H+hogStride*o+2*nBins;
                    float *R1 = R+o*hogStride;
                    float *N1 = hogNorm;
                    for( row=0; row<hogHeight; ++row ) 
                    {
                        // 第一列特殊处理
                        t = R1[0]*N1[0];
                        *H1 =  (t>clip) ? clip:t;
                        // 其他列
                        ++H1;
                        ++R1;
                        for( col=1; col+4<=hogWidth; col+=4 ) 
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                            H1+=4;
                            R1+=4;
                            N1+=4;
                        }
                        for( ; col<hogWidth; ++col ) 
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 = t;
                            ++H1;   ++R1;   ++N1;
                        } 
                        ++N1;
                    }
                }
                {   // 3
                    // 第一行 第一列
                    float *H1 = H+hogStride*o+3*nBins;
                    float *R1 = R+o*hogStride;
                    float *N1 = hogNorm;
                    t = *R1 * *N1;
                    if(t>clip)
                        t=clip;
                    *H1 = t;
                    //第一行其他列 
                    ++H1;
                    ++R1;
                    for(index=1; index+4<=hogWidth; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<hogWidth; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;   ++R1;   ++N1;                      
                    }     
                    N1=hogNorm;
                    for(row=1;row<hogHeight;++row)
                    {
                        // 第一列
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t;
                        ++H1;
                        ++R1;
                        for(col=1;col+4<=hogWidth;col+=4)
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)));
                            H1+=4;
                            R1+=4;
                            N1+=4;  
                        }
                        for(;col<hogWidth;++col)
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 = t;
                            ++H1;
                            ++R1;
                            ++N1;  
                        }
                        ++N1;  
                    }
                }
            }
        }
        else if(type==1)
        {
            for( o=0; o<nOrients; o++ ) 
            {
                {   // 0
                    H1 = H+o*hogStride;
                    R1 = R+o*hogStride;
                    N1 = hogNorm;
                    for(index =0;index<aliqA;index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_mul_ps(half, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul))));
                        H1+=4;
                        R1+=4;
                        N1+=4;
                    }
                    for(;index<hogStride;++index)
                    {
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 = t*0.5f;
                        ++H1;   ++R1;   ++N1;
                    }
                }
                {   // 1
                    // 第一行特殊处理
                    H1 = H+o*hogStride;
                    R1 = R+o*hogStride;
                    N1 = hogNorm;
                    for(index=0; index+4<=hogWidth; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1), _mm_mul_ps(half,_mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<hogWidth; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*0.5f;
                        ++H1;   ++R1;   ++N1;                      
                    }
                    // 其余行
                    N1 = hogNorm;
                    for(; index<aliqA1; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_load_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1), _mm_mul_ps(half,_mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }                    
                    for(; index<hogStride; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*0.5;
                        ++H1;   ++R1;   ++N1;                      
                    }

                }
                {   // 2
                    H1 = H+hogStride*o;
                    float *R1 = R+o*hogStride;
                    float *N1 = hogNorm;
                    for( row=0; row<hogHeight; ++row ) 
                    {
                        // 第一列特殊处理
                        t = R1[0]*N1[0];
                        if(t>clip)
                            t = clip;
                        *H1 += t*0.5;
                        // 其他列
                        ++H1;
                        ++R1;
                        for( col=1; col+4<=hogWidth; col+=4 ) 
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1), _mm_mul_ps(half,_mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                            H1+=4;
                            R1+=4;
                            N1+=4;
                        }
                        for( ; col<hogWidth; ++col ) 
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 += t*0.5f;
                            ++H1;   ++R1;   ++N1;
                        } 
                        ++N1;
                    }
                }
                {   // 3
                    // 第一行 第一列
                    float *H1 = H+hogStride*o;
                    float *R1 = R+o*hogStride;
                    float *N1 = hogNorm;
                    t = *R1 * *N1;
                    if(t>clip)
                        t=clip;
                    *H1 += t*0.5f;
                    //第一行其他列 
                    ++H1;
                    ++R1;
                    for(index=1; index+4<=hogWidth; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1), _mm_mul_ps(half,_mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<hogWidth; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*0.5f;
                        ++H1;   ++R1;   ++N1;                      
                    }     
                    N1=hogNorm;
                    for(row=1;row<hogHeight;++row)
                    {
                        // 第一列
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*0.5f;
                        ++H1;
                        ++R1;
                        for(col=1;col+4<=hogWidth;col+=4)
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1), _mm_mul_ps(half,_mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                            H1+=4;
                            R1+=4;
                            N1+=4;  
                        }
                        for(;col<hogWidth;++col)
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 += t*0.5f;
                            ++H1;
                            ++R1;
                            ++N1;  
                        }
                        ++N1;  
                    }
                }
            }
        }
        else if(type==2)
        {
            for( o=0; o<nOrients; o++ ) 
            {
                {   // 0
                    H1 = H;
                    R1 = R+o*hogStride;
                    N1 = hogNorm;
                    for(index =0;index<aliqA;index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1),_mm_mul_ps(_m128_r, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                        H1+=4;
                        R1+=4;
                        N1+=4;
                    }
                    for(;index<hogStride;++index)
                    {
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*r;
                        ++H1;   ++R1;   ++N1;
                    }
                }
                {   // 1
                    // 第一行特殊处理
                    H1 = H+hogStride;
                    R1 = R+o*hogStride;
                    N1 = hogNorm;
                    for(index=0; index+4<=hogWidth; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1),_mm_mul_ps(_m128_r, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<hogWidth; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*r;
                        ++H1;   ++R1;   ++N1;                      
                    }
                    // 其余行
                    N1 = hogNorm;
                    for(; index<aliqA1; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1),_mm_mul_ps(_m128_r, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }                    
                    for(; index<hogStride; ++index)
                    {
                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*r;
                        ++H1;   ++R1;   ++N1;                      
                    }

                }
                {   // 2
                    H1 = H+hogStride*2;
                    float *R1 = R+o*hogStride;
                    float *N1 = hogNorm;
                    for( row=0; row<hogHeight; ++row ) 
                    {
                        // 第一列特殊处理
                        t = R1[0]*N1[0];
                        if(t>clip) 
                            t= clip;
                        *H1 +=  t*r;
                        // 其他列
                        ++H1;
                        ++R1;
                        for( col=1; col+4<=hogWidth; col+=4 ) 
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1),_mm_mul_ps(_m128_r, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                            H1+=4;
                            R1+=4;
                            N1+=4;
                        }
                        for( ; col<hogWidth; ++col ) 
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 += t*r;
                            ++H1;   ++R1;   ++N1;
                        } 
                        ++N1;
                    }
                }
                {   // 3
                    // 第一行 第一列
                    float *H1 = H+3*hogStride;
                    float *R1 = R+o*hogStride;
                    float *N1 = hogNorm;
                    t = *R1 * *N1;
                    if(t>clip)
                        t=clip;
                    *H1 += t*r;
                    //第一行其他列 
                    ++H1;
                    ++R1;
                    for(index=1; index+4<=hogWidth; index+=4)
                    {
                        _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                        _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                        _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1),_mm_mul_ps(_m128_r, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                        H1+=4;
                        R1+=4;
                        N1+=4;                        
                    }
                    for(; index<hogWidth; ++index)
                    {                        
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*r;
                        ++H1;   ++R1;   ++N1;                      
                    }     
                    N1=hogNorm;
                    for(row=1;row<hogHeight;++row)
                    {
                        // 第一列
                        t = *R1 * *N1;
                        if(t>clip)
                            t=clip;
                        *H1 += t*r;
                        ++H1;
                        ++R1;
                        for(col=1;col+4<=hogWidth;col+=4)
                        {
                            _m128_Mul = _mm_mul_ps(_mm_loadu_ps(R1),_mm_loadu_ps(N1));
                            _m128_Cmp = _mm_cmpgt_ps(_m128_clip,_m128_Mul);
                            _mm_storeu_ps(H1, _mm_add_ps(_mm_loadu_ps(H1),_mm_mul_ps(_m128_r, _mm_or_ps(_mm_andnot_ps(_m128_Cmp,_m128_clip),_mm_and_ps(_m128_Cmp,_m128_Mul)))));
                            H1+=4;
                            R1+=4;
                            N1+=4;  
                        }
                        for(;col<hogWidth;++col)
                        {
                            t = *R1 * *N1;
                            if(t>clip)
                                t=clip;
                            *H1 += t*r;
                            ++H1;
                            ++R1;
                            ++N1;  
                        }
                        ++N1;  
                    }
                }
            }
        }
    }
    #else 
    {
        if(type==0)
        {
            N1 = hogNorm;
            for( o=0; o<nOrients; o++ ) 
            {
                R1 = R+ o*hogStride;
                { // 0
                    H1 = H+ o*hogStride;
                    for( index=0; index<hogStride; ++index ) 
                    {
                        t = R1[index]*N1[index];
                        if(t>clip) 
                            t=clip;
                        H1[index] = t;
                    }
                }
                { // 1
                    H1 = H+o*hogStride+nBins;
                    // 第一行处理
                    for(index=0;index<hogWidth;++index)
                    { 
                        t = R1[index]*N1[index];
                        if(t>clip) 
                            t=clip;
                        H1[index] = t;
                    }

                    for(;index<hogStride;++index)
                    { 
                        t = R1[index]*N1[index-hogWidth];
                        if(t>clip) 
                            t=clip;
                        H1[index] = t;
                    }
                }
                { // 2
                    H1 = H+o*hogStride+2*nBins;
                    for(row=0;row<hogHeight;++row)
                    {
                        // 第一列处理
                        int rw = row*hogWidth;
                        t = R1[rw] * N1[rw];
                        if(t>clip) 
                            t=clip;
                        H1[rw] = t;
                        // 其他列处理
                        for(col=1;col<hogWidth;++col)
                        {
                            t = R1[rw+col] * N1[rw+col-1];
                            if(t>clip) 
                                t=clip;
                            H1[rw+col] = t;
                        }
                    }
                }
                { // 3
                    H1 = H+o*hogStride+3*nBins;
                    //第一行第一列
                    t = R1[0]*N1[0];
                    if(t>clip) 
                        t=clip;
                    H1[0] = t;
                    //第一行 其他
                    for(col=1;col<hogWidth;++col)
                    {
                        t = R1[col] * hogNorm[col-1];
                        if(t>clip) 
                            t=clip;
                        H1[col] = t;
                    }
                    for(row=1;row<hogHeight;++row)
                    {
                        // 第一列处理
                        int rw = row*hogWidth;
                        int r1w = (row-1)*hogWidth;
                        t = R1[rw] * N1[r1w];
                        if(t>clip) 
                            t=clip;
                        H1[rw] = t;
                        // 其他列处理
                        for(col=1;col<hogWidth;++col)
                        {
                            t = R1[rw+col] * N1[r1w+col-1];
                            if(t>clip) 
                                t=clip;
                            H1[rw+col] = t;
                        }
                    }
                }
            }
        }
        else if(type==1)
        {            
            N1 = hogNorm;
            for( o=0; o<nOrients; o++ ) 
            {
                R1 = R+ o*hogStride;
                H1 = H+ o*hogStride;
                { // 0
                    for( index=0; index<hogStride; ++index ) 
                    {
                        t = R1[index]*N1[index];
                        if(t>clip) 
                            t=clip;
                        H1[index] = t* 0.5f;
                    }
                }
                { // 1
                    // 第一行处理
                    for(index=0;index<hogWidth;++index)
                    { 
                        t = R1[index]*N1[index];
                        if(t>clip) 
                            t=clip;
                        H1[index] += (t*0.5f);
                    }

                    for(;index<hogStride;++index)
                    { 
                        t = R1[index]*N1[index-hogWidth];
                        if(t>clip) 
                            t=clip;
                        H1[index] += (t*0.5f);
                    }
                }
                { // 2
                    for(row=0;row<hogHeight;++row)
                    {
                        // 第一列处理
                        int rw = row*hogWidth;
                        t = R1[rw] * N1[rw];
                        if(t>clip) 
                            t=clip;
                        H1[rw] += (t*0.5f);
                        // 其他列处理
                        for(col=1;col<hogWidth;++col)
                        {
                            t = R1[rw+col] * N1[rw+col-1];
                            if(t>clip) 
                                t=clip;
                            H1[rw+col] += (t*0.5f);
                        }
                    }
                }
                { // 3
                    //第一行第一列
                    t = R1[0]*N1[0];
                    if(t>clip) 
                        t=clip;
                    H1[0] += (t*0.5f);
                    //第一行 其他
                    for(col=1;col<hogWidth;++col)
                    {
                        t = R1[col] * hogNorm[col-1];
                        if(t>clip) 
                            t=clip;
                        H1[col] += (t*0.5f);
                    }
                    for(row=1;row<hogHeight;++row)
                    {
                        // 第一列处理
                        int rw = row*hogWidth;
                        int r1w = (row-1)*hogWidth;
                        t = R1[rw] * N1[r1w];
                        if(t>clip) 
                            t=clip;
                        H1[rw] += (t*0.5f);
                        // 其他列处理
                        for(col=1;col<hogWidth;++col)
                        {
                            t = R1[rw+col] * N1[r1w+col-1];
                            if(t>clip) 
                                t=clip;
                            H1[rw+col] += (t*0.5f);
                        }
                    }
                }
            }
        }
        else if(type==2)
        {
            N1 = hogNorm;
            for( o=0; o<nOrients; o++ ) 
            {
                R1 = R+ o*hogStride;
                { // 0
                    H1 = H;
                    for( index=0; index<hogStride; ++index ) 
                    {
                        t = R1[index]*N1[index];
                        if(t>clip) 
                            t=clip;
                        H1[index] += t*r;
                    }
                }
                { // 1
                    H1 = H+hogStride;
                    // 第一行处理
                    for(index=0;index<hogWidth;++index)
                    { 
                        t = R1[index]*N1[index];
                        if(t>clip) 
                            t=clip;
                        H1[index] += t*r;
                    }

                    for(;index<hogStride;++index)
                    { 
                        t = R1[index]*N1[index-hogWidth];
                        if(t>clip) 
                            t=clip;
                        H1[index] += t*r;
                    }
                }
                { // 2
                    H1 = H+2*hogStride;
                    for(row=0;row<hogHeight;++row)
                    {
                        // 第一列处理
                        int rw = row*hogWidth;
                        t = R1[rw] * N1[rw];
                        if(t>clip) 
                            t=clip;
                        H1[rw] += t*r;
                        // 其他列处理
                        for(col=1;col<hogWidth;++col)
                        {
                            t = R1[rw+col] * N1[rw+col-1];
                            if(t>clip) 
                                t=clip;
                            H1[rw+col] += t*r;
                        }
                    }
                }
                { // 3
                    H1 = H+3*hogStride;
                    //第一行第一列
                    t = R1[0]*N1[0];
                    if(t>clip) 
                        t=clip;
                    H1[0] += t*r;
                    //第一行 其他
                    for(col=1;col<hogWidth;++col)
                    {
                        t = R1[col] * hogNorm[col-1];
                        if(t>clip) 
                            t=clip;
                        H1[col] += t*r;
                    }
                    for(row=1;row<hogHeight;++row)
                    {
                        // 第一列处理
                        int rw = row*hogWidth;
                        int r1w = (row-1)*hogWidth;
                        t = R1[rw] * N1[r1w];
                        if(t>clip) 
                            t=clip;
                        H1[rw] += t*r;
                        // 其他列处理
                        for(col=1;col<hogWidth;++col)
                        {
                            t = R1[rw+col] * N1[r1w+col-1];
                            if(t>clip) 
                                t=clip;
                            H1[rw+col] += t*r;
                        }
                    }
                }
            }
        }
    }
    #endif
}

void Hog::put_image( cv::Mat _grayImg)
{
    ENSURE(_grayImg.channels() == 1);

    width = _grayImg.cols;
    height = _grayImg.rows;
    imgSize = width*height;
    
    hogWidth = (width + cellSize/2) / cellSize ;
    hogHeight = (height + cellSize/2) / cellSize ;
    hogStride = hogWidth * hogHeight ;

	clock_t start = clock();  
    
    if(!useMemoryPool)    
        prepare_buffers();
    else
    {
        memset(hog,0,sizeof(float)*hogStride * numOrient * 2);
        memset(hogNorm,0,sizeof(float)*hogStride);
    }
    cal_grad(_grayImg);
    orient2Bin();
    trilinear_interpolation();
    float *p_hog = hog, *p_hogPlus =hog+hogStride*numOrient, *p_hog2 = hog2;

    const unsigned int total =  hogStride*numOrient;
    #ifdef SSE_ACC
        const unsigned int aliquot = total/4;
        for(index=0;index<aliquot;++index)
        {
            _mm_store_ps(p_hog2, _mm_add_ps(_mm_load_ps(p_hog),_mm_loadu_ps(p_hogPlus)));
            p_hog+=4;
            p_hogPlus+=4;
            p_hog2+=4;
        }
        for(index=index*4;index<total;++index)
        {
            *p_hog2 = *p_hog+*p_hogPlus;
            ++p_hog;
            ++p_hogPlus;
            ++p_hog2;
        }
    #else
        for(index=0;index<total;++index)
        {
            *p_hog2 = *p_hog+*p_hogPlus;
            ++p_hog;
            ++p_hogPlus;
            ++p_hog2;
        }
    #endif
    norm_matrix();

    float* __attribute__((aligned(16))) HogResult = new float[(3*numOrient+4)*hogStride];

    float clip = 0.2f;
    channels(HogResult,hog,numOrient*2,clip,1);
    channels(HogResult+total*2,hog2,numOrient,clip,1);
    channels(HogResult+total*3,hog,numOrient*2,clip,2);


    clock_t end   = clock();  

    std::cout<<" time "<<(double)(end-start)/CLOCKS_PER_SEC<<std::endl;
    #ifdef Debug_Out
        show_grad();
    #endif

    free_buffers();
    delete[] HogResult;
}

#ifdef Debug_Out
void Hog::show_grad()
{
    #ifdef SSE_ACC
        std::ofstream ofs_gx("./gx.txt",std::ios::trunc);
        std::ofstream ofs_gy("./gy.txt",std::ios::trunc);
        std::ofstream ofs_grad("./grad.txt",std::ios::trunc);
        std::ofstream ofs_orient("./orient.txt",std::ios::trunc);
        std::ofstream ofs_grad0("./grad_0.txt",std::ios::trunc);
        std::ofstream ofs_grad1("./grad_1.txt",std::ios::trunc);
        std::ofstream ofs_orient0("./orient_0.txt",std::ios::trunc);
        std::ofstream ofs_orient1("./orient_1.txt",std::ios::trunc);
        std::ofstream ofs_hog("./hog.txt",std::ios::trunc);
        std::ofstream ofs_hog2("./hog2.txt",std::ios::trunc);
        std::ofstream ofs_hogNorm("./hogNorm.txt",std::ios::trunc);
    #else
        std::ofstream ofs_gx("./gx1.txt",std::ios::trunc);
        std::ofstream ofs_gy("./gy1.txt",std::ios::trunc);
        std::ofstream ofs_grad("./grad1.txt",std::ios::trunc);
        std::ofstream ofs_orient("./orient1.txt",std::ios::trunc);
        std::ofstream ofs_grad0("./grad1_0.txt",std::ios::trunc);
        std::ofstream ofs_grad1("./grad1_1.txt",std::ios::trunc);
        std::ofstream ofs_orient0("./orient1_0.txt",std::ios::trunc);
        std::ofstream ofs_orient1("./orient1_1.txt",std::ios::trunc);
        std::ofstream ofs_hog("./hog1.txt",std::ios::trunc);
        std::ofstream ofs_hog2("./hog21.txt",std::ios::trunc);
        std::ofstream ofs_hogNorm("./hogNorm1.txt",std::ios::trunc);
    #endif

    for(row=0;row<height;++row)
    {
        for(col=0;col<width;++col)
        {
            ofs_gx<<Gradx[row*width+col]<<" ";
            ofs_gy<<Grady[row*width+col]<<" ";
            ofs_grad<<Grad[row*width+col]<<" ";
            ofs_orient<<Orient[row*width+col]<<" ";
            ofs_grad0<<Grad0[row*width+col]<<" ";
            ofs_grad1<<Grad1[row*width+col]<<" ";
            ofs_orient0<<Orient0[row*width+col]<<" ";
            ofs_orient1<<Orient1[row*width+col]<<" ";
        }
        ofs_gx<<std::endl;
        ofs_gy<<std::endl;
        ofs_grad<<std::endl;
        ofs_orient<<std::endl;
        ofs_grad0<<std::endl;
        ofs_grad1<<std::endl;
        ofs_orient0<<std::endl;
        ofs_orient1<<std::endl;
    }

    for(int o=0;o<2*numOrient;++o)
        for(row=0;row<hogHeight;++row)
        {
            for(col=0;col<hogWidth;++col)
            {
                ofs_hog<<hog[o*hogStride+row*hogWidth+col]<<" ";
            }
            ofs_hog<<std::endl;
        }
    for(int o=0;o<numOrient;++o)
        for(row=0;row<hogHeight;++row)
        {
            for(col=0;col<hogWidth;++col)
            {
                ofs_hog2<<hog2[o*hogStride+row*hogWidth+col]<<" ";
            }
            ofs_hog2<<std::endl;
        }
    for(row=0;row<hogHeight;++row)
    {
        for(col=0;col<hogWidth;++col)
        {
            ofs_hogNorm<<hogNorm[row*hogWidth+col]<<" ";
        }
        ofs_hogNorm<<std::endl;
    }

    cv::Mat DImgGx(height,width,CV_8UC1);
    cv::Mat DImgGy(height,width,CV_8UC1);
    cv::Mat DImgGrad(height,width,CV_8UC1);
    cv::Mat DImgOrient(height,width,CV_8UC1);

    cv::Mat DImgGrad0(height,width,CV_8UC1);
    cv::Mat DImgOrient0(height,width,CV_8UC1);
    cv::Mat DImgGrad1(height,width,CV_8UC1);
    cv::Mat DImgOrient1(height,width,CV_8UC1);

    uchar* dataGx = DImgGx.data;
    uchar* dataGy = DImgGy.data;
    uchar* dataGrad = DImgGrad.data;
    uchar* dataOrient = DImgOrient.data;

    uchar* dataGrad0 = DImgGrad0.data;
    uchar* dataOrient0 = DImgOrient0.data;
    uchar* dataGrad1 = DImgGrad1.data;
    uchar* dataOrient1 = DImgOrient1.data;
    for(row=0;row<height;++row)
    {
        for(col=0;col<width;++col)
        {
            dataGx[row*width+col] = std::abs(Gradx[row*width+col]/2);
            dataGy[row*width+col] = std::abs(Grady[row*width+col]/2);
            dataGrad[row*width+col] = (int)(Grad[row*width+col]/1.414);
            dataOrient[row*width+col] = (int)(Orient[row*width+col]/2/PI*255);
            
            // std::cout<<Orient0[row*width+col]<<std::endl;
            dataGrad0[row*width+col] = (int)(Grad0[row*width+col]*cellSize*cellSize/1.414);
            // std::cout<<(int)dataGrad0[row*width+col]<<std::endl;
            dataGrad1[row*width+col] = (int)(Grad1[row*width+col]*cellSize*cellSize/1.414);
            int nOrients;
            if(variant == HogVariantDalalTriggs)
                nOrients = numOrient;
            else 
                nOrients = numOrient*2;

            dataOrient0[row*width+col] = (int)(Orient0[row*width+col]*255.0f/nOrients);
            dataOrient1[row*width+col] = (int)(Orient1[row*width+col]*255/nOrients);
        }
    }

    cv::imshow("Gx",DImgGx);
    cv::imshow("Gy",DImgGy);
    cv::imshow("Grad",DImgGrad);
    cv::imshow("Orient",DImgOrient);
    cv::imshow("Grad0",DImgGrad0);
    cv::imshow("Orient0",DImgOrient0);
    cv::imshow("Grad1",DImgGrad1);
    cv::imshow("Orient1",DImgOrient1);
    cv::waitKey(0);

}
#endif

} // namespace 