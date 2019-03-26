#pragma once
#include<cstring>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#ifdef __SSE2__
#include <emmintrin.h>
#endif
using namespace std;
namespace Feature{

class Feature{
public:
    float* __attribute__((aligned(16))) data;
    int dim1;   // 特征的行数
    int dim2;   // 特征的列数
    int dim3;   // 特征的维度
    int size;   // 大小

    Feature():data(NULL),dim1(0),dim2(0),dim3(0),size(0)
    {
    }    
    Feature(int _dim1,int _dim2,int _dim3):dim1(_dim1),dim2(_dim2),dim3(_dim3),size(_dim1*_dim2*_dim3)
    {
        data = new float[size];
        if(data == NULL)
        {
            printf("Error,Allocate memory failure，New Feature\n");
            exit(-1);
        }
        for(int i=0;i<size;++i)
            data[i] = 0.f;
    }
    Feature(Feature& _tem)
    {
        this->dim1 = _tem.dim1;
        this->dim2 = _tem.dim2;
        this->dim3 = _tem.dim3;
        this->size = _tem.size;
        this->data = new float[size];
        if(data == NULL)
        {
            printf("Error,Allocate memory failure，New Feature\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(float)*size);
    }
    Feature(const Feature& _tem)
    {
        this->dim1 = _tem.dim1;
        this->dim2 = _tem.dim2;
        this->dim3 = _tem.dim3;
        this->size = _tem.size;
        this->data = new float[size];
        if(data == NULL)
        {
            printf("Error,Allocate memory failure，New Feature\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(float)*size);
    }
    ~Feature()
    {
        delete[] data;
        data = NULL;
    }
    void resize(int _dim1,int _dim2,int _dim3)
    {
        int tmp_size = _dim1*_dim2*_dim3;
        if(data == NULL)   // data == NULL
        {
            this->dim1 = _dim1;
            this->dim2 = _dim2;
            this->dim3 = _dim3;
            this->size = tmp_size;
            this->data = new float[tmp_size];
        }
        else if(tmp_size>size) //要求的内存比当前内存大，重新分配内存
        {
            delete[] data;
            this->dim1 = _dim1;
            this->dim2 = _dim2;
            this->dim3 = _dim3;
            this->size = tmp_size;
            this->data = new float[tmp_size];
        }
        else
        {
            this->dim1 = _dim1;
            this->dim2 = _dim2;
            this->dim3 = _dim3;
            this->size = tmp_size;
        }
        if(data == NULL)
        {
            printf("Error,Allocate memory failure，New Feature\n");
            exit(-1);
        }
        for(int i=0;i<size;++i)
            data[i] = 0.f;
    }

    Feature& operator = (Feature& _tem)
    {
        int tmp_size = _tem.dim1*_tem.dim2*_tem.dim3;
        if(data == NULL)   // data == NULL
        {
            this->dim1 = _tem.dim1;
            this->dim2 = _tem.dim2;
            this->dim3 = _tem.dim3;
            this->size = tmp_size;
            this->data = new float[tmp_size];
        }
        else if(tmp_size>size) //要求的内存比当前内存大，重新分配内存
        {
            delete[] data;
            this->dim1 = _tem.dim1;
            this->dim2 = _tem.dim2;
            this->dim3 = _tem.dim3;
            this->size = tmp_size;
            this->data = new float[tmp_size];
        }
        else
        {
            this->dim1 = _tem.dim1;
            this->dim2 = _tem.dim2;
            this->dim3 = _tem.dim3;
            this->size = tmp_size;
        }
        if(data == NULL)
        {
            printf("Error,Allocate memory failure，New Feature\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(float)*size);
        return *this;
    }

    Feature& operator = (const Feature& _tem)
    {
        int tmp_size = _tem.dim1*_tem.dim2*_tem.dim3;
        if(data == NULL)   // data == NULL
        {
            this->dim1 = _tem.dim1;
            this->dim2 = _tem.dim2;
            this->dim3 = _tem.dim3;
            this->size = tmp_size;
            this->data = new float[tmp_size];
        }
        else if(tmp_size>size) //要求的内存比当前内存大，重新分配内存
        {
            delete[] data;
            this->dim1 = _tem.dim1;
            this->dim2 = _tem.dim2;
            this->dim3 = _tem.dim3;
            this->size = tmp_size;
            this->data = new float[tmp_size];
        }
        else
        {
            this->dim1 = _tem.dim1;
            this->dim2 = _tem.dim2;
            this->dim3 = _tem.dim3;
            this->size = tmp_size;
        }
        if(data == NULL)
        {
            printf("Error,Allocate memory failure，New Feature\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(float)*size);
        return *this;
    }

    const float eps = 2.22e-16;
    void norm(int normalize_power,int normalize_size,int normalize_dim)
    {
        float sum = 0;
        // #ifdef __SSE2__
        //     // 后续改为16字节对齐
        //     __m128 _128_sum = _mm_set1_ps(0.0f);
        //     __m128 _128_data;
        //     float f_4[4];
        //     const int ele_nums = dim1*dim2*dim3;
        //     const int aliq = ele_nums-4;
        //     int index=0;
        //     for(;index<=aliq;index+=4)
        //     {
        //         _128_data = _mm_loadu_ps(data+index);
        //         _128_sum = _mm_add_ps(_128_sum,_mm_mul_ps(_128_data,_128_data));
        //     }
        //     _mm_storeu_ps(f_4,_128_sum);
        //     sum  = f_4[0] + f_4[1] + f_4[2] + f_4[3];
        //     for(;index<ele_nums;index++)
        //     {
        //         sum += (data[index]*data[index]);
        //     }
        //     float times = sqrt(pow(dim1*dim2,normalize_size) * pow(dim3,normalize_dim) / (sum +eps));

        //     __m128 _128_times = _mm_set1_ps(times);
        //     index=0;
        //     for(;index<=aliq;index+=4)
        //     {
        //         _mm_storeu_ps(data+index, _mm_mul_ps(_mm_loadu_ps(data+index),_128_times));
        //     }
        //     for(;index<ele_nums;index++)
        //     {
        //         data[index] *= times;
        //     }
        // #else
            int index=0;
            for(;index<size;index++)
            {
                sum += pow(data[index],2);
            }
            float times = sqrt(pow(dim1*dim2,normalize_size) * pow(dim3,normalize_dim) / (sum +eps));
            for(index=0;index<size;index++)
            {
                data[index] *= times;
            }
        // #endif
    }

};

}