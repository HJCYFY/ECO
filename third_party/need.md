程序需要的第三方库包括 Eigen FFTW 和 OpenCV;
Eigen 和 FFTW 将作为源码成为程序的一部分;
Eigen 主要负责矩阵运算;
FFTW 负责FFT计算，FFTW的性能与平台有关，编译市需要格外注意，具体参见FFTW官网;
OpenCV 负责图像操作，包括resize cvtColor等;
