#include "fhog.hpp"
#include <iostream>
namespace Feature{

#define PI 3.14159265f
#define RETf inline __m128
#define RETi inline __m128i

// wrapper functions if compiling from C/C++
inline void wrError(const char *errormsg) { throw errormsg; }
inline void* wrCalloc( size_t num, size_t size ) { return calloc(num,size); }
inline void* wrMalloc( size_t size ) { return malloc(size); }
inline void wrFree( void * ptr ) { free(ptr); }
// platform independent aligned memory allocation (see also alFree)
void* alMalloc( size_t size, int alignment ) {
  const size_t pSize = sizeof(void*), a = alignment-1;
  void *raw = wrMalloc(size + a + pSize);
  void *aligned = (void*) (((size_t) raw + pSize + a) & ~a);
  *(void**) ((size_t) aligned-pSize) = raw;
  return aligned;
}

// platform independent alignned memory de-allocation (see also alMalloc)
void alFree(void* aligned) {
  void* raw = *(void**)((char*)aligned-sizeof(void*));
  wrFree(raw);
}

// set, load and store values
RETf SET( const float &x ) { return _mm_set1_ps(x); }
RETf SET( float x, float y, float z, float w ) { return _mm_set_ps(x,y,z,w); }
RETi SET( const int &x ) { return _mm_set1_epi32(x); }
RETf LD( const float &x ) { return _mm_load_ps(&x); }
RETf LDu( const float &x ) { return _mm_loadu_ps(&x); }
RETf STR( float &x, const __m128 y ) { _mm_store_ps(&x,y); return y; }
RETf STR1( float &x, const __m128 y ) { _mm_store_ss(&x,y); return y; }
RETf STRu( float &x, const __m128 y ) { _mm_storeu_ps(&x,y); return y; }
RETf STR( float &x, const float y ) { return STR(x,SET(y)); }

// arithmetic operators
RETi ADD( const __m128i x, const __m128i y ) { return _mm_add_epi32(x,y); }
RETf ADD( const __m128 x, const __m128 y ) { return _mm_add_ps(x,y); }
RETf ADD( const __m128 x, const __m128 y, const __m128 z ) {
  return ADD(ADD(x,y),z); }
RETf ADD( const __m128 a, const __m128 b, const __m128 c, const __m128 &d ) {
  return ADD(ADD(ADD(a,b),c),d); }
RETf SUB( const __m128 x, const __m128 y ) { return _mm_sub_ps(x,y); }
RETf MUL( const __m128 x, const __m128 y ) { return _mm_mul_ps(x,y); }
RETf MUL( const __m128 x, const float y ) { return MUL(x,SET(y)); }
RETf MUL( const float x, const __m128 y ) { return MUL(SET(x),y); }
RETf INC( __m128 &x, const __m128 y ) { return x = ADD(x,y); }
RETf INC( float &x, const __m128 y ) { __m128 t=ADD(LD(x),y); return STR(x,t); }
RETf DEC( __m128 &x, const __m128 y ) { return x = SUB(x,y); }
RETf DEC( float &x, const __m128 y ) { __m128 t=SUB(LD(x),y); return STR(x,t); }
// RETf MIN( const __m128 x, const __m128 y ) { return _mm_min_ps(x,y); }
RETf RCP( const __m128 x ) { return _mm_rcp_ps(x); }
RETf RCPSQRT( const __m128 x ) { return _mm_rsqrt_ps(x); }

// logical operators
RETf AND( const __m128 x, const __m128 y ) { return _mm_and_ps(x,y); }
RETi AND( const __m128i x, const __m128i y ) { return _mm_and_si128(x,y); }
RETf ANDNOT( const __m128 x, const __m128 y ) { return _mm_andnot_ps(x,y); }
RETf OR( const __m128 x, const __m128 y ) { return _mm_or_ps(x,y); }
RETf XOR( const __m128 x, const __m128 y ) { return _mm_xor_ps(x,y); }

// comparison operators
RETf CMPGT( const __m128 x, const __m128 y ) { return _mm_cmpgt_ps(x,y); }
RETf CMPLT( const __m128 x, const __m128 y ) { return _mm_cmplt_ps(x,y); }
RETi CMPGT( const __m128i x, const __m128i y ) { return _mm_cmpgt_epi32(x,y); }
RETi CMPLT( const __m128i x, const __m128i y ) { return _mm_cmplt_epi32(x,y); }

// conversion operators
RETf CVT( const __m128i x ) { return _mm_cvtepi32_ps(x); }
RETi CVT( const __m128 x ) { return _mm_cvttps_epi32(x); }


fHog::fHog(int _binSize,int _nOrients):
            binSize(_binSize),nOrients(_nOrients)
{
    const long int n=10000;
    const long int b=10;
    acos_table = new float[2*(n+b)];
    acost = acos_table+n+b;
    int i;
    for( i=-n-b; i<-n; i++ )   acost[i]=PI;
    for( i=-n; i<n; i++ )      acost[i]=float(std::acos(i/float(n)));
    for( i=n; i<n+b; i++ )     acost[i]=0;
    for( i=-n-b; i<n/10; i++ ) if( acost[i] > PI-1e-6f ) acost[i]=PI-1e-6f;
}

fHog::~fHog()
{
    delete[] acos_table;
    acos_table = NULL;
}

shared_ptr<Feature> fHog::extract(cv::Mat img)
{
    int N = img.rows*img.cols*img.channels();
    float *I = new float[N];
    float *IMG = I;
    uchar* data;
    if(img.channels() == 3)
      for(int i=0;i<img.cols;++i)
          for(int j=0;j<img.rows;++j)
          {
              data = img.data + j*img.cols*3 + i*3;
              IMG[0] = data[0];
              IMG[img.rows*img.cols] = data[1];
              IMG[img.rows*img.cols*2] = data[2];
              IMG++;
          }
      else{
        for(int i=0;i<img.cols;++i)
            for(int j=0;j<img.rows;++j)
            {
              data = img.data + j*img.cols + i;
              IMG[0] = data[0];
              IMG++;
            }
      }

    float *M = new float[img.rows*img.cols];
    float *O = new float[img.rows*img.cols];
    gradMag( I, M, O, img.rows, img.cols, img.channels(), 1 );

    int hb = img.rows/binSize; 
    int wb = img.cols/binSize;
    int nChns = 3*nOrients+4;
    shared_ptr<Feature> H = make_shared<Feature>(hb,wb,nChns);
    fhog( M, O, H->data, img.rows, img.cols, binSize, nOrients, -1, 0.2 );
    delete[] I;
    delete[] M;
    delete[] O;
    return H;
}

void fHog::grad1( float *I, float *Gx, float *Gy, int h, int w, int x ) {
  int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
  // compute column of Gx
  Ip=I-h; In=I+h; r=.5f;
  if(x==0) { r=1; Ip+=h; } else if(x==w-1) { r=1; In-=h; }
  if( h<4 || h%4>0 || (size_t(I)&15) || (size_t(Gx)&15) ) {
    for( y=0; y<h; y++ ) *Gx++=(*In++-*Ip++)*r;
  } else {
    _G=(__m128*) Gx; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = SET(r);
    for(y=0; y<h; y+=4) *_G++=MUL(SUB(*_In++,*_Ip++),_r);
  }
  // compute column of Gy
  #define GRADY(r) *Gy++=(*In++-*Ip++)*r;
  Ip=I; In=Ip+1;
  // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
  y1=((~((size_t) Gy) + 1) & 15)/4; if(y1==0) y1=4; if(y1>h-1) y1=h-1;
  GRADY(1); Ip--; for(y=1; y<y1; y++) GRADY(.5f);
  _r = SET(.5f); _G=(__m128*) Gy;
  for(; y+4<h-1; y+=4, Ip+=4, In+=4, Gy+=4)
    *_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
  for(; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
  #undef GRADY
}

// compute gradient magnitude and orientation at each location (uses sse)
void fHog::gradMag( float *I, float *M, float *O, int h, int w, int d, bool full ) {
    int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
    float acMult=10000.0f;
    // allocate memory for storing one column of output (padded so h4%4==0)
    h4=(h%4==0) ? h : h-(h%4)+4; s=d*h4*sizeof(float);
    M2=(float*) alMalloc(s,16); _M2=(__m128*) M2;
    Gx=(float*) alMalloc(s,16); _Gx=(__m128*) Gx;
    Gy=(float*) alMalloc(s,16); _Gy=(__m128*) Gy;

    // compute gradient magnitude and orientation for each column
    for( x=0; x<w; x++ ) {
      // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
        for(c=0; c<d; c++) {
            // 求c通道 x列 的 Gx Gy
            grad1( I+x*h+c*w*h, Gx+c*h4, Gy+c*h4, h, w, x );         
            for( y=0; y<h4/4; y++ ) {
                y1=h4/4*c+y;
                // 梯度平方和
                _M2[y1]=ADD(MUL(_Gx[y1],_Gx[y1]),MUL(_Gy[y1],_Gy[y1]));
                if( c==0 ) continue; _m = CMPGT( _M2[y1], _M2[y] );
                _M2[y] = OR( AND(_m,_M2[y1]), ANDNOT(_m,_M2[y]) );
                _Gx[y] = OR( AND(_m,_Gx[y1]), ANDNOT(_m,_Gx[y]) );
                _Gy[y] = OR( AND(_m,_Gy[y1]), ANDNOT(_m,_Gy[y]) );
            }
        }

        // compute gradient mangitude (M) and normalize Gx
        for( y=0; y<h4/4; y++ ) {
            _m = MIN( RCPSQRT(_M2[y]), SET(1e10f) );
            _M2[y] = RCP(_m);
            if(O) _Gx[y] = MUL( MUL(_Gx[y],_m), SET(acMult) );
            if(O) _Gx[y] = XOR( _Gx[y], AND(_Gy[y], SET(-0.f)) );
        };
        memcpy( M+x*h, M2, h*sizeof(float) );
        // compute and store gradient orientation (O) via table lookup
        if( O!=0 ) for( y=0; y<h; y++ ) O[x*h+y] = acost[(int)Gx[y]];
        if( O!=0 && full ) {
            y1=((~size_t(O+x*h)+1)&15)/4; y=0;
            for( ; y<y1; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
            for( ; y<h-4; y+=4 ) STRu( O[y+x*h],
              ADD( LDu(O[y+x*h]), AND(CMPLT(LDu(Gy[y]),SET(0.f)),SET(PI)) ) );
            for( ; y<h; y++ ) O[y+x*h]+=(Gy[y]<0)*PI;
        }
    }
    alFree(Gx); alFree(Gy); alFree(M2);
}

void fHog::hogChannels( float *H, const float *R, const float *N,
  int hb, int wb, int nOrients, float clip, int type )
{
    #define GETT(blk) t=R1[y]*N1[y-(blk)]; if(t>clip) t=clip; c++;
    const float r=.2357f; int o, x, y, c; float t;
    const int nb=wb*hb, nbo=nOrients*nb, hb1=hb+1;
    for( o=0; o<nOrients; o++ ) for( x=0; x<wb; x++ ) {
      const float *R1=R+o*nb+x*hb, *N1=N+x*hb1+hb1+1;
      float *H1 = (type<=1) ? (H+o*nb+x*hb) : (H+x*hb);
      if( type==0) for( y=0; y<hb; y++ ) {
        // store each orientation and normalization (nOrients*4 channels)
        c=-1; GETT(0); H1[c*nbo+y]=t; GETT(1); H1[c*nbo+y]=t;
        GETT(hb1); H1[c*nbo+y]=t; GETT(hb1+1); H1[c*nbo+y]=t;
      } else if( type==1 ) for( y=0; y<hb; y++ ) {
        // sum across all normalizations (nOrients channels)
        c=-1; GETT(0); H1[y]+=t*.5f; GETT(1); H1[y]+=t*.5f;
        GETT(hb1); H1[y]+=t*.5f; GETT(hb1+1); H1[y]+=t*.5f;
      } else if( type==2 ) for( y=0; y<hb; y++ ) {
        // sum across all orientations (4 channels)
        c=-1; GETT(0); H1[c*nb+y]+=t*r; GETT(1); H1[c*nb+y]+=t*r;
        GETT(hb1); H1[c*nb+y]+=t*r; GETT(hb1+1); H1[c*nb+y]+=t*r;
      }
    }
    #undef GETT
}

// HOG helper: compute 2x2 block normalization values (padded by 1 pixel)
float* fHog::hogNormMatrix( float *H, int nOrients, int hb, int wb, int bin ) {
    float *N, *N1, *n; int o, x, y, dx, dy, hb1=hb+1, wb1=wb+1;
    float eps = 1e-4f/4/bin/bin/bin/bin; // precise backward equality
    N = (float*) wrCalloc(hb1*wb1,sizeof(float)); N1=N+hb1+1;
    for( o=0; o<nOrients; o++ ) for( x=0; x<wb; x++ ) for( y=0; y<hb; y++ )
      N1[x*hb1+y] += H[o*wb*hb+x*hb+y]*H[o*wb*hb+x*hb+y];
    for( x=0; x<wb-1; x++ ) for( y=0; y<hb-1; y++ ) {
      n=N1+x*hb1+y; *n=1/float(sqrt(n[0]+n[1]+n[hb1]+n[hb1+1]+eps)); }
    x=0;     dx= 1; dy= 1; y=0;                  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=0;     dx= 1; dy= 0; for(y=0; y<hb1; y++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=0;     dx= 1; dy=-1; y=hb1-1;              N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy= 1; y=0;                  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy= 0; for( y=0; y<hb1; y++) N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    x=wb1-1; dx=-1; dy=-1; y=hb1-1;              N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    y=0;     dx= 0; dy= 1; for(x=0; x<wb1; x++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    y=hb1-1; dx= 0; dy=-1; for(x=0; x<wb1; x++)  N[x*hb1+y]=N[(x+dx)*hb1+y+dy];
    return N;
}
// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void fHog::gradQuantize( float *O, float *M, int *O0, int *O1, float *M0, float *M1,
  int nb, int n, float norm, int nOrients, bool full, bool interpolate )
{
  // assumes all *OUTPUT* matrices are 4-byte aligned
  int i, o0, o1; float o, od, m;
  __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
  // define useful constants
  const float oMult=(float)nOrients/(full?2*PI:PI); const int oMax=nOrients*nb;
  const __m128 _norm=SET(norm), _oMult=SET(oMult), _nbf=SET((float)nb);
  const __m128i _oMax=SET(oMax), _nb=SET(nb);
  // perform the majority of the work with sse
  _O0=(__m128i*) O0; _O1=(__m128i*) O1; _M0=(__m128*) M0; _M1=(__m128*) M1;
  if( interpolate ) for( i=0; i<=n-4; i+=4 ) {
    // 原本的角度 × nOrients / (2×PI) // 角度归一化到 0～nOrients
    _o=MUL(LDu(O[i]),_oMult); 
    // 整数部分
    _o0=CVT(_o); 
    // 小数部分
    _od=SUB(_o,CVT(_o0));
    // 整数部分 × nb
    _o0=CVT(MUL(CVT(_o0),_nbf)); 
    // 整数部分 < nOrients 返回_o0 否则返回 0
    _o0=AND(CMPGT(_oMax,_o0),_o0); 
    // 给O0赋值 
    *_O0++=_o0;
    // (整数部分+1)*nb
    _o1=ADD(_o0,_nb); 
    //  (整数部分+1) < nOrients 返回_o1 否则返回 0
    _o1=AND(CMPGT(_oMax,_o1),_o1); 
    // 给O1赋值 
    *_O1++=_o1;
    // 幅度除以Cell面积
    _m=MUL(LDu(M[i]),_norm); 
    // 再乘以 小数部分 给M1赋值
    *_M1=MUL(_od,_m); 
    // 给M0赋值
    *_M0++=SUB(_m,*_M1); _M1++;
  } else for( i=0; i<=n-4; i+=4 ) {
    _o=MUL(LDu(O[i]),_oMult); _o0=CVT(ADD(_o,SET(.5f)));
    _o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
    *_M0++=MUL(LDu(M[i]),_norm); *_M1++=SET(0.f); *_O1++=SET(0);
  }
  // compute trailing locations without sse
  if( interpolate ) for(; i<n; i++ ) {
    o=O[i]*oMult; o0=(int) o; od=o-o0;
    o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
    o1=o0+nb; if(o1==oMax) o1=0; O1[i]=o1;
    m=M[i]*norm; M1[i]=od*m; M0[i]=m-M1[i];
  } else for(; i<n; i++ ) {
    o=O[i]*oMult; o0=(int) (o+.5f);
    o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
    M0[i]=M[i]*norm; M1[i]=0; O1[i]=0;
  }
}

void fHog::gradHist( float *M, float *O, float *H, int h, int w,
  int bin, int nOrients, int softBin, bool full )
{
    const int hb=h/bin, wb=w/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
    const float s=(float)bin, sInv=1/s, sInv2=1/s/s;
    float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float xb, init;
    O0=(int*)alMalloc(h*sizeof(int),16); M0=(float*) alMalloc(h*sizeof(float),16);
    O1=(int*)alMalloc(h*sizeof(int),16); M1=(float*) alMalloc(h*sizeof(float),16);
    // main loop
    for( x=0; x<w0; x++ ) {
      // compute target orientation bins for entire column - very fast
      gradQuantize(O+x*h,M+x*h,O0,O1,M0,M1,nb,h0,sInv2,nOrients,full,softBin>=0);

      if( softBin<0 && softBin%2==0 ) {
        // no interpolation w.r.t. either orienation or spatial bin
        H1=H+(x/bin)*hb;
        #define GH H1[O0[y]]+=M0[y]; y++;
        if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
        else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
        else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
        else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
        else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
        #undef GH

      } else if( softBin%2==0 || bin==1 ) {
        // interpolate w.r.t. orientation only, not spatial bin
        H1=H+(x/bin)*hb;
        #define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
        if( bin==1 )      for(y=0; y<h0;) { GH; H1++; }
        else if( bin==2 ) for(y=0; y<h0;) { GH; GH; H1++; }
        else if( bin==3 ) for(y=0; y<h0;) { GH; GH; GH; H1++; }
        else if( bin==4 ) for(y=0; y<h0;) { GH; GH; GH; GH; H1++; }
        else for( y=0; y<h0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
        #undef GH

      } else {
        // interpolate using trilinear interpolation
        float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
        bool hasLf, hasRt; int xb0, yb0;
        if( x==0 ) { init=(0+.5f)*sInv-0.5f; xb=init; }
        // 第一列 xb = 1/(2×sInv)-0.5 , hasLf = false
        hasLf = xb>=0; 
        // 第一列 xb0 = -1 ,
        xb0 = hasLf?(int)xb:-1; 
        // 第一列 hasRt true
        hasRt = xb0 < wb-1;
        // xd = xb+1 ,  xb+=sInv ,yb=init;
        xd=xb-xb0; xb+=sInv; yb=init; y=0;
        // macros for code conciseness

        //              
        #define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
          ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
        #define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
        // leading rows, no top bin
        for( ; y<bin/2; y++ ) {
          yb0=-1; GHinit;
          if(hasLf) { H0[O0[y]+1]+=ms[1]*M0[y]; H0[O1[y]+1]+=ms[1]*M1[y]; }
          if(hasRt) { H0[O0[y]+hb+1]+=ms[3]*M0[y]; H0[O1[y]+hb+1]+=ms[3]*M1[y]; }
        }
        // main rows, has top and bottom bins, use SSE for minor speedup
        if( softBin<0 ) for( ; ; y++ ) {
          yb0 = (int) yb; if(yb0>=hb-1) break; GHinit; _m0=SET(M0[y]);
          if(hasLf) { _m=SET(0,0,ms[1],ms[0]); GH(H0+O0[y],_m,_m0); }
          if(hasRt) { _m=SET(0,0,ms[3],ms[2]); GH(H0+O0[y]+hb,_m,_m0); }
        } else for( ; ; y++ ) {
          yb0 = (int) yb; if(yb0>=hb-1) break; GHinit;
          _m0=SET(M0[y]); _m1=SET(M1[y]);
          if(hasLf) { _m=SET(0,0,ms[1],ms[0]);
            GH(H0+O0[y],_m,_m0); GH(H0+O1[y],_m,_m1); }
          if(hasRt) { _m=SET(0,0,ms[3],ms[2]);
            GH(H0+O0[y]+hb,_m,_m0); GH(H0+O1[y]+hb,_m,_m1); }
        }
        // final rows, no bottom bin
        for( ; y<h0; y++ ) {
          yb0 = (int) yb; GHinit;
          if(hasLf) { H0[O0[y]]+=ms[0]*M0[y]; H0[O1[y]]+=ms[0]*M1[y]; }
          if(hasRt) { H0[O0[y]+hb]+=ms[2]*M0[y]; H0[O1[y]+hb]+=ms[2]*M1[y]; }
        }
        #undef GHinit
        #undef GH
      }
    }
    alFree(O0); alFree(O1); alFree(M0); alFree(M1);
    // normalize boundary bins which only get 7/8 of weight of interior bins
    if( softBin%2!=0 ) for( int o=0; o<nOrients; o++ ) {
      x=0; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
      y=0; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
      x=wb-1; for( y=0; y<hb; y++ ) H[o*nb+x*hb+y]*=8.f/7.f;
      y=hb-1; for( x=0; x<wb; x++ ) H[o*nb+x*hb+y]*=8.f/7.f;
    }
}

// compute FHOG features
void fHog::fhog( float *M, float *O, float *H, int h, int w, int binSize,
  int nOrients, int softBin, float clip )
{
    const int hb=h/binSize, wb=w/binSize, nb=hb*wb, nbo=nb*nOrients;
    float *N, *R1, *R2; int o, x;
    // compute unnormalized constrast sensitive histograms
    R1 = (float*) wrCalloc(wb*hb*nOrients*2,sizeof(float));
    gradHist( M, O, R1, h, w, binSize, nOrients*2, softBin, true );
    // compute unnormalized contrast insensitive histograms
    R2 = (float*) wrCalloc(wb*hb*nOrients,sizeof(float));
    for( o=0; o<nOrients; o++ ) for( x=0; x<nb; x++ )
      R2[o*nb+x] = R1[o*nb+x]+R1[(o+nOrients)*nb+x];
    // compute block normalization values
    N = hogNormMatrix( R2, nOrients, hb, wb, binSize );
    // normalized histograms and texture channels
    hogChannels( H+nbo*0, R1, N, hb, wb, nOrients*2, clip, 1 );
    hogChannels( H+nbo*2, R2, N, hb, wb, nOrients*1, clip, 1 );
    hogChannels( H+nbo*3, R1, N, hb, wb, nOrients*2, clip, 2 );
    wrFree(N); wrFree(R1); wrFree(R2);
}
}