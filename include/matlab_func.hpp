#pragma once
#include "type.hpp"
#include <iomanip>
using namespace Track;
namespace Matlab{

MatrixXd hannWin(int N);
void cfft2(Array4D<float>& in,Array4D<complex<float> >& out);
void fft2(Array4D<float>& in,Array4D<complex<float> >& out);
void ifft2(MatrixXcf& in, MatrixXf& out,int rows,int cols);
void ifft(const MatrixXcf& in, MatrixXf& out,int N);
void permute4312(const Array4D<complex<float> >& in,Array4D<complex<float> >& out);
void permute3412(const Array4D<complex<float> >& in,Array4D<complex<float> >& out);
void permute3421(const Array4D<complex<float> >& in,Array4D<complex<float> >& out);
// void permute4312_f(Array4D<complex<float> >& in,Array4D<complex<float> >& out);
// void permute3412_f(Array4D<complex<float> >& in,Array4D<complex<float> >& out);
enum Conv_Type{
    convn_full = 0,
    convn_valid
};
void conv3(const Array4D<complex<float> >& a,const MatrixXf& b,int type,Array4D<complex<float> >&c);

void read_mat(std::string file, Array4D<float>& array4d);
void read_mat(std::string file, Array4D<complex<float> >& array4d);

// void read_mat(std::string file, int rows, int cols,double* data);
void read_mat(std::string file, int rows, int cols,complex<float>* data);
void read_mat(std::string file, int rows, int cols, int dims, float* data);
// void read_mat(std::string file, int rows, int cols, int dims, complex<float>* data);
// void read_mat(std::string file, int d1, int d2, int d3,int d4, complex<float>* data);
void read_mat(std::string file, int rows, int cols,float* data);
void read_mat(std::string file, int rows, int cols,double* data);

template<typename T>
void write_mat(std::string file, int dim1, int dim2, int dim3, T* data, int type = 0)
{
    assert(data != NULL);
    std::ofstream ofs;
    ofs.open(file,ios::trunc);
    assert(ofs.is_open());
    if(type == 0)
    {
        for(int d3=0;d3<dim3;++d3)
            for(int d2=0;d2<dim2;++d2)
            {
                for(int d1=0;d1<dim1;++d1)
                    ofs<<setiosflags(ios::scientific)<<data[d3*dim2*dim1+d2*dim1+d1]<<" ";
                ofs<<endl;
            }
    }
    else if(type == 1)
    {
        for(int d3=0;d3<dim3;++d3)
            for(int d2=0;d2<dim2;++d2)
            {
                for(int d1=0;d1<dim1;++d1)
                    ofs<<setiosflags(ios::scientific)<<(int)data[d3*dim2*dim1+d2*dim1+d1]<<" ";
                ofs<<endl;
            }
    }
    ofs.close();
}

void write_mat(std::string file, int rows, int cols, float* data);
void write_mat(std::string file, int rows, int cols, complex<double>* data);
void write_mat(std::string file, int rows, int cols, double* data);
void write_mat(std::string file, int rows, int cols, uchar* data);
void write_mat(std::string file, int rows, int cols, complex<float>* data);
void write_mat(std::string file,const  Array4D<float>& mat);
void write_mat(std::string file,const  Array4D<complex<float> >& mat);
}
