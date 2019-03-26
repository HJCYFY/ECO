#pragma once 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace Eigen;
using namespace std;
using namespace cv;
namespace Track{

const double PI = 3.14159265358979323846;

enum SEARCH_AREA_SHAPE{
    proportional,
    square,
    fix_padding,
    custom
};
enum SIZE_MODE{
    same,
    exact,
    odd_cells
};
enum INTERPOLATION_METHOD{
    none,
    ideal,
    bicubic
};
enum PROJ_INIT_METHOD{
    pca,
    rand_uni
};
enum SAMPLE_MERGE_TYPE{
    replace,
    merge
};

struct CG_OPTS{
    bool CG_use_FR;
    float tol;
    bool CG_standard_alpha;
    bool debug;
    float init_forget_factor;
    int maxit;

    CG_OPTS():CG_use_FR(true),tol(1e-6),CG_standard_alpha(true),debug(false),init_forget_factor(0)
    {        
    }
};

template <typename T>
class Array4D{
public:
    typedef Matrix<T,Dynamic,Dynamic,RowMajor> MatrixTr;
    typedef Matrix<T,Dynamic,Dynamic,ColMajor> MatrixTc;

    Array4D():dim1(0),dim2(0),dim3(0),dim4(0),stride12(0),stride34(0),size(0),data(NULL){};
    Array4D(int _dim1,int _dim2,int _dim3,int _dim4):dim1(_dim1),dim2(_dim2),dim3(_dim3),dim4(_dim4),stride12(_dim1*_dim2),stride34(_dim3*_dim4),size(stride12*stride34)
    {
        data = new T[size];
        if(!data)
        {
            printf(" new failed !\n");
            exit(-1);
        }
    }
    Array4D(Array4D& _tem)
    {
        cout<<"Array4D(Array4D& _tem)"<<endl;
        this->dim1 = _tem.dim1;
        this->dim2 = _tem.dim2;
        this->dim3 = _tem.dim3;
        this->dim4 = _tem.dim4;
        this->stride12 = _tem.stride12;
        this->stride34 = _tem.stride34;
        this->size = _tem.size;
        this->data = new T[size];
        if(!data)
        {
            printf(" new failed !\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(T)*size);
    }
    Array4D(const Array4D& _tem)
    {
        this->dim1 = _tem.dim1;
        this->dim2 = _tem.dim2;
        this->dim3 = _tem.dim3;
        this->dim4 = _tem.dim4;
        this->stride12 = _tem.stride12;
        this->stride34 = _tem.stride34;
        this->size = _tem.size;
        this->data = new T[size];
        if(!data)
        {
            printf(" new failed !\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(T)*size);
    } 
    ~Array4D()
    {
        if(data)
            delete[] data;
    }
    void resize(int _dim1,int _dim2, int _dim3,int _dim4)
    {
        int tmp_size = _dim1*_dim2*_dim3*_dim4;
        if(data ==NULL)
        {
            data = new T[tmp_size];
        }
        else if(size < tmp_size)
        {
            delete[] data;
            data = new T[tmp_size];
        }
        dim1 = _dim1;
        dim2 = _dim2;
        dim3 = _dim3;
        dim4 = _dim4;
        stride12 = dim1 * dim2;
        stride34 = dim3 * dim4;
        size = tmp_size;
        if(!data)
        {
            printf(" new failed !\n");
            exit(-1);
        }
    }

    void resize(int _dim1,int _dim2, int _dim3,int _dim4,T value)
    {
        int tmp_size = _dim1*_dim2*_dim3*_dim4;
        if(data ==NULL)
        {
            data = new T[tmp_size];
        }
        else if(size < tmp_size)
        {
            delete[] data;
            data = new T[tmp_size];
        }
        dim1 = _dim1;
        dim2 = _dim2;
        dim3 = _dim3;
        dim4 = _dim4;
        stride12 = dim1 * dim2;
        stride34 = dim3 * dim4;
        size = tmp_size;
        if(!data)
        {
            printf(" new failed !\n");
            exit(-1);
        }
        for(int i=0;i<size;++i)
        {
            data[i] = value;
        }
    }

    Array4D& operator = (Array4D& _tem)
    {
        if(this->data == NULL)
        {
            this->data = new T[_tem.size];
        }
        else if(size < _tem.size);
        {
            delete[] this->data;
            this->data = new T[_tem.size];
        }
        this->dim1 = _tem.dim1;
        this->dim2 = _tem.dim2;
        this->dim3 = _tem.dim3;
        this->dim4 = _tem.dim4;
        this->stride12 = _tem.stride12;
        this->stride34 = _tem.stride34;
        this->size = _tem.size;
        if(!data)
        {
            printf(" new failed !\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(T)*size);
        return *this;
    }

    Array4D& operator = (const Array4D& _tem)
    {
        if(this->data == NULL)
        {
            this->data = new T[_tem.size];
        }
        else if(size != _tem.size);
        {
            delete[] this->data;
            this->data = new T[_tem.size];
        }
        this->dim1 = _tem.dim1;
        this->dim2 = _tem.dim2;
        this->dim3 = _tem.dim3;
        this->dim4 = _tem.dim4;
        this->stride12 = _tem.stride12;
        this->stride34 = _tem.stride34;
        this->size = _tem.size;
        if(!data)
        {
            printf(" new failed !\n");
            exit(-1);
        }
        memcpy(this->data,_tem.data,sizeof(T)*size);
        return *this;
    }

    inline Map<MatrixTc> getMatrix12(int i) const
    {
        return Map<MatrixTc>(data+i*stride12,dim1,dim2);
    }

    inline Map<MatrixTc> getMatrix12(int i)
    {
        return Map<MatrixTc>(data+i*stride12,dim1,dim2);
    }

    inline Map<MatrixTc> getMatrix34() const
    {
        return Map<MatrixTc>(data,dim3,dim4);
    }

    inline Map<MatrixTc> getMatrix34()
    {
        return Map<MatrixTc>(data,dim3,dim4);
    }

    inline Map<MatrixTc> getVector() const
    {
        return Map<MatrixTc>(data,size,1);
    }

    inline Map<MatrixTc> getVector()
    {
        return Map<MatrixTc>(data,size,1);
    }

    inline Map<MatrixTc> getVector_T() const
    {
        return Map<MatrixTc>(data,1,size);
    }

    inline Map<MatrixTc> getVector_T()
    {
        return Map<MatrixTc>(data,1,size);
    }

    void output_size()
    {
        cout<<" dim1 "<<dim1<<" dim2 "<<dim2<<" dim3 "<<dim3<<" dim4 "<<dim4<<endl;
    }

    int dim1;
    int dim2;
    int dim3;
    int dim4;
    int stride12;
    int stride34;
    int size;
    T* data;
};


}