#include "matlab_func.hpp"
#include <fftw3.h>

namespace Matlab{

MatrixXd hannWin(int N)
{
    int n;
    MatrixXd ret(N,1);
    double* ret_data = ret.data();
    for(n = 0; n < N; n++)
    {
        ret_data[n] = 0.5 * (1 - cos(2*PI*(double)n/(N-1)));
    }
    return ret;
}

template<typename T>
void fftshift(T* in,T*out,int rows,int cols,int offset_x, int offset_y)
{
    while(offset_x > cols)
        offset_x -= cols;
    while(offset_y > rows)
        offset_y -= rows;
    while(offset_x < 0)
        offset_x += cols;
    while(offset_y < 0)
        offset_y += rows;
    int rbh = rows - offset_y;
    int rbw = cols - offset_x;
    int type_size = sizeof(T);

    T *in_data;
    T *out_data;
    for(int row=0;row<rbh;++row)
    {
        in_data = in+row*cols;
        out_data = out+(offset_y+row)*cols+offset_x;
        memcpy(out_data,in_data,type_size*rbw);
    }
    for(int row=0;row<offset_y;++row)
    {
        in_data = in+(row+rbh)*cols;
        out_data = out+row*cols+offset_x;
        memcpy(out_data,in_data,type_size*rbw);
    }
    for(int row=0;row<offset_y;++row)
    {
        in_data = in+(row+rbh)*cols+rbw;
        out_data = out+row*cols;
        memcpy(out_data,in_data,type_size*offset_x);
    }
    for(int row=0;row<rbh;++row)
    {
        in_data = in+row*cols+rbw;
        out_data = out+(row+offset_y)*cols;
        memcpy(out_data,in_data,type_size*offset_x);
    }
}

void cfft2(Array4D<float>& in,Array4D<complex<float> >& out)
{
    assert(in.dim4 == 1);
    // 选取的 采样空间的方法 保证了 第一维度和第二维度 是奇数
    if(in.dim1%2==1 && in.dim2%2==1)
    {
        out.resize(in.dim1,in.dim2,in.dim3,1);
        double  *ptr_in;
        fftw_complex *ptr_out;
        fftw_plan p;
        ptr_in = (double*) fftw_malloc(sizeof(double) * in.stride12);
        ptr_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (in.dim1/2+1)*in.dim2);
        p = fftw_plan_dft_r2c_2d(in.dim2, in.dim1, ptr_in, ptr_out,FFTW_ESTIMATE);
        int h1=(in.dim1+1)/2, h2=in.dim1-h1, w1=(in.dim2+1)/2, w2=in.dim2-w1;
        MatrixXcf temp(in.dim1,in.dim2);
        for(int dim=0;dim<in.dim3;++dim)
        {
            Map<MatrixXd>(ptr_in,1,in.stride12) = Map<MatrixXf>(in.data+dim*in.stride12,1,in.stride12).cast<double>();      

            fftw_execute(p); 
            // fft shift
            temp.topRows(in.dim1/2+1) = Map<MatrixXcd>((complex<double>*)ptr_out,in.dim1/2+1,in.dim2).cast<complex<float> >();
            temp.bottomLeftCorner(in.dim1-in.dim1/2-1,1) = Map<MatrixXcd>((complex<double>*)ptr_out,in.dim1/2+1,in.dim2).middleRows(1,in.dim1-in.dim1/2-1).leftCols(1).colwise().reverse().conjugate().cast<complex<float> >();
            temp.bottomRightCorner(in.dim1-in.dim1/2-1,in.dim2-1) = Map<MatrixXcd>((complex<double>*)ptr_out,in.dim1/2+1,in.dim2).middleRows(1,in.dim1-in.dim1/2-1).rightCols(in.dim2-1).colwise().reverse().rowwise().reverse().conjugate().cast<complex<float> >();
            
            Map<MatrixXcf>(out.data+dim*out.stride12,out.dim1,out.dim2).topLeftCorner(h2,w2) = temp.bottomRightCorner(h2,w2);
            Map<MatrixXcf>(out.data+dim*out.stride12,out.dim1,out.dim2).topRightCorner(h2,w1) = temp.bottomLeftCorner(h2,w1);
            Map<MatrixXcf>(out.data+dim*out.stride12,out.dim1,out.dim2).bottomLeftCorner(h1,w2) = temp.topRightCorner(h1,w2);
            Map<MatrixXcf>(out.data+dim*out.stride12,out.dim1,out.dim2).bottomRightCorner(h1,w1) = temp.topLeftCorner(h1,w1);
        }
        fftw_destroy_plan(p);
        fftw_free(ptr_in); fftw_free(ptr_out);
    }
    else
    {
        // 这里待测试
        printf("Wait To Do \n");
        exit(0);
    }
}

void fft2(Array4D<float>& in,Array4D<complex<float> >& out)
{
    if(in.dim1%2==1)
        out.resize(in.dim1,in.dim2/2+1,in.dim3,1);
    else
        out.resize(in.dim1+1,in.dim2/2+1,in.dim3,1);
    double  *ptr_in;
    fftw_complex *ptr_out;
    fftw_plan p;
    ptr_in = (double*) fftw_malloc(sizeof(double) * in.stride12);
    ptr_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * in.dim1*(in.dim2/2+1));
    p = fftw_plan_dft_r2c_2d(in.dim2, in.dim1, ptr_in, ptr_out,FFTW_ESTIMATE);
    MatrixXcf temp(in.dim1,in.dim2);
    for(int dim=0;dim<in.dim3;++dim)
    {
        Map<MatrixXd>(ptr_in,1,in.stride12) = Map<MatrixXf>(in.data+dim*in.stride12,1,in.stride12).cast<double>();
        fftw_execute(p);
        out.getMatrix12(dim).bottomRightCorner(in.dim1/2+1,1) = Map<MatrixXcd>((complex<double>*)ptr_out,in.dim1/2+1,in.dim2).leftCols(1).cast<complex<float> >();
        out.getMatrix12(dim).bottomLeftCorner(in.dim1/2+1,in.dim2/2) = Map<MatrixXcd>((complex<double>*)ptr_out,in.dim1/2+1,in.dim2).rightCols(in.dim2/2).cast<complex<float> >();
        out.getMatrix12(dim).topRightCorner(in.dim1/2,1) = Map<MatrixXcd>((complex<double>*)ptr_out,in.dim1/2+1,in.dim2).leftCols(1).bottomRows(in.dim1/2).colwise().reverse().conjugate().cast<complex<float> >();
        out.getMatrix12(dim).topLeftCorner(in.dim1/2,in.dim2/2) = Map<MatrixXcd>((complex<double>*)ptr_out,in.dim1/2+1,in.dim2).middleCols(1,in.dim2/2).bottomRows(in.dim1/2).colwise().reverse().rowwise().reverse().conjugate().cast<complex<float> >();
    }
    fftw_destroy_plan(p);
    fftw_free(ptr_in); fftw_free(ptr_out);
}

void ifft2(MatrixXcf& in, MatrixXf& out,int rows,int cols)
{
    double *ptr_in = (double*) fftw_malloc(sizeof(double) * rows*cols);
    fftw_complex *ptr_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (rows/2+1)*cols);   
    fftw_plan p = fftw_plan_dft_c2r_2d(cols, rows, ptr_out, ptr_in ,FFTW_ESTIMATE); 

    Map<MatrixXcd>((complex<double>*)ptr_out,(rows/2+1),cols) = in.cast<complex<double> >();
    fftw_execute(p);
    out = Map<MatrixXd>(ptr_in,rows,cols).cast<float>();
    fftw_destroy_plan(p);
    fftw_free(ptr_in); fftw_free(ptr_out);
}

void ifft(const MatrixXcf& in, MatrixXf& out,int N)
{
    double *in_ptr = (double*)fftw_malloc(sizeof(double) * N);
    fftw_complex *out_ptr = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N/2+1));
    fftw_plan p = fftw_plan_dft_c2r_1d(N, out_ptr, in_ptr, FFTW_ESTIMATE);
    Map<MatrixXcd>((complex<double>*)out_ptr,1,N/2+1) = in.cast<complex<double> >();
    fftw_execute(p); /* repeat as needed */
    out = Map<MatrixXd>(in_ptr,1,N).cast<float>();
    fftw_destroy_plan(p);
    fftw_free(in_ptr); fftw_free(out_ptr);
}

// void permute4312_f(Array4D<complex<float> >& in,Array4D<complex<float> >& out){
//     int stride4 = in.dim1*in.dim2*in.dim3;
//     int stride3 = in.dim1*in.dim2;
//     // int stride2 = 1;
//     int stride1 = in.dim1;

//     complex<float> *in_data1 = in.data;
//     complex<float> *in_data2, *in_data3, *in_data4;
//     complex<float> *out_data = out.data;

//     for(int d2=0;d2<in.dim2;++d2)
//     {
//         in_data2 = in_data1;
//         for(int d1=0;d1<in.dim1;++d1)
//         {
//             in_data3 = in_data2;
//             for(int d3=0;d3<in.dim3;++d3)
//             {
//                 in_data4 = in_data3;
//                 for(int d4=0;d4<in.dim4;++d4)
//                 {
//                     *out_data = *in_data4;
//                     ++out_data;
//                     in_data4 += stride4;
//                 }
//                 in_data3 += stride3;
//             }
//             in_data2 += 1;
//         }
//         in_data1+=stride1;
//     }
// }

// void permute3412_f(Array4D<complex<float> >& in,Array4D<complex<float> >& out){
//     int stride4 = in.dim1*in.dim2*in.dim3;
//     int stride3 = in.dim1*in.dim2;
//     // int stride2 = 1;
//     int stride1 = in.dim1;

//     complex<float> *in_data1 = in.data;
//     complex<float> *in_data2, *in_data3, *in_data4;
//     complex<float> *out_data = out.data;

//     for(int d2=0;d2<in.dim2;++d2)
//     {
//         in_data2 = in_data1;
//         for(int d1=0;d1<in.dim1;++d1)
//         {
//             in_data3 = in_data2;
//             for(int d3=0;d3<in.dim4;++d3)
//             {
//                 in_data4 = in_data3;
//                 for(int d4=0;d4<in.dim3;++d4)
//                 {
//                     *out_data = *in_data4;
//                     ++out_data;
//                     in_data4 += stride3;
//                 }
//                 in_data3 += stride4;
//             }
//             in_data2 += 1;
//         }
//         in_data1+=stride1;
//     }
// }


void permute4312(const Array4D<complex<float> >& in,Array4D<complex<float> >& out)
{
    assert(in.dim4==1);
    if(out.data==NULL)
        out.resize(1,in.dim3,in.dim1,in.dim2);
    vector<complex<float>* > in_ptr(in.dim3);
    complex<float>* out_ptr = out.data;

    // 指向第三维度
    for(int i=0;i<in.dim3;++i)
        in_ptr[i] = in.data + i*in.stride12;

    for(int index=0;index<in.stride12;++index)
    {
        for(int i=0;i<in.dim3;++i)
        {
            *out_ptr = in_ptr[i][index];
            ++out_ptr;
        }
    }
}

void permute3412(const Array4D<complex<float> >& in,Array4D<complex<float> >& out)
{
    assert(in.dim4==1);
    if(out.data==NULL)
        out.resize(in.dim3,1,in.dim1,in.dim2);
    vector<complex<float>* > in_ptr(in.dim3);
    complex<float>* out_ptr = out.data;

    // 指向第三维度
    for(int i=0;i<in.dim3;++i)
        in_ptr[i] = in.data + i*in.stride12;

    for(int index=0;index<in.stride12;++index)
    {
        for(int i=0;i<in.dim3;++i)
        {
            *out_ptr = in_ptr[i][index];
            ++out_ptr;
        }
    }
}

void permute3421(const Array4D<complex<float> >& in,Array4D<complex<float> >& out)
{
    assert(in.dim1==1 || in.dim2==1);
    if(out.data==NULL)
        out.resize(in.dim3,in.dim4,in.stride12,1);
    complex<float>* in_data = in.data;
    vector<complex<float>* > out_data(in.stride12);
    int stride = in.stride34;
    for(int i=0;i<in.stride12;++i)
        out_data[i] = out.data+i*stride;

    for(int index=0;index<stride;++index)
    {
        for(int i=0;i<in.stride12;++i)
        {
            out_data[i][index] = *in_data;
            ++in_data;
        }
    }
}


// 0 full 1 same 2 valid
void conv3(const Array4D<complex<float> >& a,const MatrixXf& b,int type,Array4D<complex<float> >&c)
{
    const int a_rows = a.dim1;
    const int b_rows = b.rows();
    const int a_cols = a.dim2;
    const int b_cols = b.cols();
    const int a_dims = a.dim3;

    MatrixXf br = b.colwise().reverse().rowwise().reverse();

    if(type ==0)
    {

        int N_rows = a_rows+b_rows-1;
        int N_cols = a_cols+b_cols-1;
        int N_dims = a_dims;
        c.resize(N_rows,N_cols,N_dims,1);
        complex<float>* c_data = c.data;
        int front,top,bottom,left,right,height ,width,startx,starty;
        for(int d=0;d<N_dims;++d)
        {
            front = d;
            for(int c=0;c<N_cols;++c)
            {
                right = min(c+1,a_cols);
                left = max(c+1-b_cols,0);
                width = right-left;
                startx = max(b_cols-(c+1),0);
                for(int r=0;r<N_rows;++r)
                {
                    bottom = min(r+1,a_rows);
                    top = max(r+1-b_rows,0);
                    height = bottom - top;
                    starty = max(b_rows-(r+1),0);
                    *c_data = ( Map<MatrixXcf>(a.data+front*a.stride12,a.dim1,a.dim2).middleCols(left, width).middleRows(top,height).array() * br.middleCols(startx,width).middleRows(starty,height).array() ).sum();      
                    ++c_data;
                }
            }
        }
    }
    else if(type == Matlab::convn_valid)
    {
        assert(a_rows >= b_rows);
        assert(a_cols >= b_cols);

        int N_rows = a_rows-b_rows+1;
        int N_cols = a_cols-b_cols+1;
        int N_dims = a_dims;
        c.resize(N_rows,N_cols,N_dims,1);
        complex<float>* c_data = c.data;
        int front,top,left;
        for(int d=0;d<N_dims;++d)
        {
            front = d;
            for(int c=b_cols;c<=a_cols;++c)
            {
                left = c-b_cols;
                for(int r=b_rows;r<=a_rows;++r)
                {
                    top = r-b_rows;
                    *c_data = (Map<MatrixXcf>(a.data+front*a.stride12,a.dim1,a.dim2).middleCols(left, b_cols).middleRows(top,b_rows).array() * br.array() ).sum();      
                    ++c_data;
                }
            }
        }
    }
    else 
    {
        printf("wrong conv type.\n");
        exit(0); 
    }
}

void read_mat(std::string file, Array4D<float>& array4d)
{
        assert(array4d.data != NULL);
        float* data = array4d.data;
        std::string line;
        float real;
        int index=0;
        std::ifstream cin(file);
        assert(cin.is_open());
        while(getline(cin,line))
        {
            std::istringstream read_data(line);
            while(read_data>>real)
            {
                data[index] = real;
                ++index;
            }
        }
        assert(array4d.size == index);
}

void read_mat(std::string file, Array4D<complex<float> >& array4d)
{
        assert(array4d.data != NULL);
        complex<float>* data = array4d.data;
        std::string line;
        float real,imag;
        int index=0;
        std::ifstream cin(file);
        assert(cin.is_open());
        while(getline(cin,line))
        {
            std::istringstream read_data(line);
            while(read_data>>real)
            {
                read_data>>imag;
                complex<float> temp(real,imag);
                data[index] = temp;
                ++index;
            }
        }
        if(index != array4d.size)
        {
            cout<<"file name "<<file<<endl;
            cout<<"size "<<array4d.size<<endl;
            cout<<"index "<<index<<endl;
            cout<<"dim1 "<<array4d.dim1<<endl;
            cout<<"dim2 "<<array4d.dim2<<endl;
            cout<<"dim3 "<<array4d.dim3<<endl;
            cout<<"dim4 "<<array4d.dim4<<endl;
            exit(0);
        }
}

// void read_mat(std::string file, int rows, int cols, int dims, complex<float>* data)
// {
//     assert(data != NULL);
//     std::string line;
//     float real,imag;
//     int index=0;
//     std::ifstream cin(file);
//     assert(cin.is_open());
//     while(getline(cin,line))
//     {
//         std::istringstream read_data(line);
//         while(read_data>>real)
//         {
//             read_data>>imag;
//             complex<float> temp(real,imag);
//             data[index] = temp;
//             ++index;
//         }
//     }
//     assert(rows*cols*dims == index);
// }

void read_mat(std::string file, int rows, int cols, int dims, float* data)
{
    assert(data != NULL);
    std::string line;
    float temp;
    int index=0;
    std::ifstream cin(file);
    assert(cin.is_open());
    while(getline(cin,line))
    {
        std::istringstream read_data(line);
        while(read_data>>temp)
        {
            data[index] = temp;
            ++index;
        }
    }
    assert(rows*cols*dims == index);
}

void read_mat(std::string file, int rows, int cols,float* data)
{
    assert(data != NULL);
    std::string line;
    string temp;
    int index=0;
    std::ifstream cin(file);
    assert(cin.is_open());
    while(getline(cin,line))
    {
        std::istringstream read_data(line);
        while(read_data>>temp)
        {
            data[index] = ::atof(temp.c_str());
            ++index;
        }
    }
    assert(rows*cols == index);
}

void read_mat(std::string file, int rows, int cols,double* data)
{
    assert(data != NULL);
    std::string line;
    string temp;
    int index=0;
    std::ifstream cin(file);
    assert(cin.is_open());
    while(getline(cin,line))
    {
        std::istringstream read_data(line);
        while(read_data>>temp)
        {
            data[index] = ::atof(temp.c_str());
            ++index;
        }
    }
    assert(rows*cols == index);
}

// void read_mat(std::string file, int rows, int cols,double* data)
// {
//     assert(data != NULL);
//     std::string line;
//     string temp;
//     int index=0;
//     std::ifstream cin(file);
//     assert(cin.is_open());
//     while(getline(cin,line))
//     {
//         std::istringstream read_data(line);
//         while(read_data>>temp)
//         {
//             data[index] = ::atof(temp.c_str());
//             ++index;
//         }
//     }
//     assert(rows*cols == index);
// }


void read_mat(std::string file, int rows, int cols,complex<float>* data)
{
    assert(data != NULL);
    std::string line;
    float real,imag;
    int index=0;
    std::ifstream cin(file);
    assert(cin.is_open());
    while(getline(cin,line))
    {
        std::istringstream read_data(line);
        while(read_data>>real)
        {
            read_data>>imag;
            data[index] = complex<float>(real,imag);
            ++index;
        }
    }
    assert(rows*cols == index);
}

// void read_mat(std::string file, int d1, int d2, int d3,int d4, complex<float>* data)
// {
//     assert(data != NULL);
//     std::string line;
//     float real,imag;
//     int index=0;
//     std::ifstream cin(file);
//     assert(cin.is_open());
//     while(getline(cin,line))
//     {
//         std::istringstream read_data(line);
//         while(read_data>>real)
//         {
//             read_data>>imag;
//             complex<float> temp(real,imag);
//             data[index] = temp;
//             ++index;
//         }
//     }
//     assert(d1*d2*d3*d4 == index);
// }

// void write_mat(std::string file, int rows, int cols, int dims, float* data)
// {
//     assert(data != NULL);
//     std::ofstream ofs(file,ios::trunc);
//     assert(ofs.is_open());
//     for(int d=0;d<dims;++d)
//         for(int r=0;r<rows;++r)
//         {
//             for(int c=0;c<cols;++c)
//                 ofs<<setiosflags(ios::scientific)<<data[d*rows*cols+r*cols+c]<<" ";
//             ofs<<endl;
//         }
//     ofs.close();
// }

void write_mat(std::string file,const  Array4D<float>& mat)
{
    assert(mat.data != NULL);
    std::ofstream ofs(file,ios::trunc);
    assert(ofs.is_open());

    float* data = mat.data;
    for(int d4=0;d4<mat.dim4;++d4)
        for(int d3=0;d3<mat.dim3;++d3)
            for(int d2=0;d2<mat.dim2;++d2)
            {
                for(int d1=0;d1<mat.dim1;++d1)
                    ofs<<setiosflags(ios::scientific)<<*(data++)<<" ";
                ofs<<endl;
            }
    ofs.close();
}

void write_mat(std::string file, const Array4D<complex<float> >& mat)
{
    assert(mat.data != NULL);
    std::ofstream ofs(file,ios::trunc);
    assert(ofs.is_open());

    complex<float>* data = mat.data;
    for(int d4=0;d4<mat.dim4;++d4)
        for(int d3=0;d3<mat.dim3;++d3)
            for(int d2=0;d2<mat.dim2;++d2)
            {
                for(int d1=0;d1<mat.dim1;++d1)
                {
                    ofs<<setiosflags(ios::scientific)<<(*(data)).real()<<" "<<(*(data)).imag()<<" ";
                    data++;
                }
                ofs<<endl;
            }
    ofs.close();
}


void write_mat(std::string file, int rows, int cols, float* data)
{
    assert(data != NULL);
    std::ofstream ofs(file,ios::trunc);
    assert(ofs.is_open());
    for(int c=0;c<cols;++c)
    {
        for(int r=0;r<rows;++r)
            ofs<<setiosflags(ios::scientific)<<data[c*rows+r]<<" ";
        ofs<<endl;
    }
    ofs.close();
}


void write_mat(std::string file, int rows, int cols, complex<double>* data)
{
    assert(data != NULL);
    std::ofstream ofs(file,ios::trunc);
    assert(ofs.is_open());
    for(int c=0;c<cols;++c)
    {
        for(int r=0;r<rows;++r)
            ofs<<setiosflags(ios::scientific)<<data[c*rows+r].real()<<" "<<data[c*rows+r].imag()<<" ";
        ofs<<endl;
    }
    ofs.close();
}

void write_mat(std::string file, int rows, int cols, complex<float>* data)
{
    assert(data != NULL);
    std::ofstream ofs(file,ios::trunc);
    assert(ofs.is_open());
    for(int c=0;c<cols;++c)
    {
        for(int r=0;r<rows;++r)
            ofs<<setiosflags(ios::scientific)<<data[c*rows+r].real()<<" "<<data[c*rows+r].imag()<<" ";
        ofs<<endl;
    }
    ofs.close();
}
void write_mat(std::string file, int rows, int cols, double* data)
{
    assert(data != NULL);
    std::ofstream ofs(file,ios::trunc);
    assert(ofs.is_open());
    for(int c=0;c<cols;++c)
    {
        for(int r=0;r<rows;++r)
            ofs<<setiosflags(ios::scientific)<<data[c*rows+r]<<" ";
        ofs<<endl;
    }
    ofs.close();
}
// void write_mat(std::string file, int rows, int cols, uchar* data)
// {
//     assert(data != NULL);
//     std::ofstream ofs(file,ios::trunc);
//     assert(ofs.is_open());
//     for(int r=0;r<rows;++r)
//     {
//         for(int c=0;c<cols;++c)
//             ofs<<setw(4)<<(int)data[r*cols+c]<<" ";
//         ofs<<endl;
//     }
//     ofs.close();
// }

// void write_mat(std::string file, int rows, int cols, complex<float>* data)
// {
//     assert(data != NULL);
//     std::ofstream ofs(file,ios::trunc);
//     assert(ofs.is_open());
//     for(int r=0;r<rows;++r)
//     {
//         for(int c=0;c<cols;++c)
//             ofs<<setiosflags(ios::scientific)<<data[r*cols+c].real()<<" "<<data[r*cols+c].imag()<<" ";
//         ofs<<endl;
//     }
//     ofs.close();
// }
}