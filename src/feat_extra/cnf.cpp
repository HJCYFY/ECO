#include "cnf.hpp"
namespace Feature{
CNf::CNf(int _binSize)
{
    binSize = _binSize;
    dims = 10;
    size = 32768;
    table_data = new float[dims*size];

    assert(table_data != NULL);
    string line;
    float element;
    int index=0;
    ifstream cin("./resource/CNnorm.txt");
    assert(cin.is_open());
    while(getline(cin,line))
    {
        istringstream data(line);
        while(data>>element)
            table_data[index++] = element;
    }
}

CNf::~CNf()
{
    delete[] table_data;
}

shared_ptr<Feature> CNf::extract(cv::Mat img)
{
    // assert(img.channels()==3);
    shared_ptr<Feature> im =  lookup_table(img);
    if(binSize>1)
    {
        shared_ptr<Feature> features = average_feature_region(im,img.rows,img.cols);
        return features;
    }    
    return im;
}

shared_ptr<Feature> CNf::lookup_table(cv::Mat img)
{
    width = img.cols;
    height = img.rows;
    stride = height*width;
    int index=0;
    const int  w3 = width*3;
        
    int32_t* imgB = new int32_t[stride];
    int32_t* imgG = new int32_t[stride];
    int32_t* imgR = new int32_t[stride];
    int32_t *p1,*p2,*p3;
    p1 = imgB;
    p2 = imgG;
    p3 = imgR;
    uchar *img_data;
    for(col=0;col<width;++col)
    {
        img_data = img.data + col*3;
        for(row=0;row<height;++row)
        {
            *p1 = img_data[0]; 
            *p2 = img_data[1]; 
            *p3 = img_data[2]; 
            ++p1;   ++p2;   ++p3;
            img_data += w3;
        }
    }

    p1 = imgB;      // 指向 B
    p2 = imgG;   // 指向 G
    p3 = imgR;    // 指向 R

    int32_t *index_im = new int32_t[stride];    
    for(index=0;index<stride;++index)
        index_im[index] = (p3[index]>>3) + ((p2[index]>>3)<<5) + ((p1[index]>>3)<<10);

    shared_ptr<Feature> table_feats = make_shared<Feature>(height,width,dims);
    float* t_data = table_feats->data;
    vector<float*> ptr(dims);
    for(int i=0;i<dims;++i)
        ptr[i] = t_data + i*stride;
    for(index=0;index<stride;++index)
    {
        // 可以将这里的乘法提到上面
        t_data = table_data + index_im[index] * dims;
        for(col=0;col<dims;++col)
        {
            *(ptr[col]) = t_data[col];
            ++(ptr[col]);
        }
    }
    delete[] imgB;
    delete[] imgG;
    delete[] imgR;
    delete[] index_im;
    return table_feats;
}

shared_ptr<Feature> CNf::average_feature_region(shared_ptr<Feature>& im, int height,int width)
{
    float area = binSize*binSize;
    shared_ptr<Feature> iImage = integralImage(im->data,height,width);

    shared_ptr<Feature> features = make_shared<Feature>(height/binSize,width/binSize, dims);

    float *p_ret = features->data;
    float* iim ;
    for(int d=0; d<dims; ++d)
    {
        iim = iImage->data + d * (height+1) * (width+1) + binSize * (height+1);
        for(col=binSize; col<=width; col+=binSize)
        {
            for(row=binSize; row<=height; row+=binSize)
            {   
                *p_ret = (iim[row] - iim[row-(height+1)*binSize] - iim[row - binSize] + iim[row-(height+1)*binSize- binSize]) / area;
                ++p_ret;
            }
            iim += binSize * (height+1);
        }
    }        
    return features;
}

shared_ptr<Feature> CNf::integralImage(float* mat ,int height,int width)
{
    const int inte_height = height+1;
    const int inte_width = width+1;
    const int inte_stride = inte_height * inte_width;
    const int sizef = sizeof(float);

    shared_ptr<Feature> integralImg = make_shared<Feature>(inte_height,inte_width,dims);

    vector<float*> ptr_src(dims); 
    vector<float*> ptr_dst(dims); 
    ptr_src[0] = mat;
    ptr_dst[0] = integralImg->data + inte_height + 1;
    for(int i=1;i<dims;++i) //维度
    {
        ptr_src[i] = ptr_src[i-1] + stride;
        ptr_dst[i] = ptr_dst[i-1] + inte_stride;
    }
    for(int i=0;i<dims;++i) //维度
    {
        // 1 列
        memcpy(ptr_dst[i],ptr_src[i],sizef*height);
        ptr_src[i] += height;
        for(col=0;col<width-1;++col)
        {   
            for(row=0;row<height;++row)
                ptr_dst[i][row+inte_height] = ptr_dst[i][row] + ptr_src[i][row];
            ptr_dst[i] += inte_height;
            ptr_src[i] += height;
        }

        ptr_dst[i] = integralImg->data + i * inte_stride + inte_height+1;
        for(col=0;col<width;++col)
        {
            for(row=0;row<height-1;++row)
            {
                ptr_dst[i][row+1] += ptr_dst[i][row];
            }
            ptr_dst[i] += inte_height;
        }
    }
    return integralImg;
}


}