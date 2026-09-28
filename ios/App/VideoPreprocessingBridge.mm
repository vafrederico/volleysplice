#include "VideoPreprocessingBridge.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {
struct Color {
    std::array<int,256> luma, redV, greenU, greenV, blueU;
    Color(double kr, double kb, bool full) {
        double kg=1-kr-kb, ys=full?1:255.0/219, cs=full?1:255.0/224;
        auto fixed=[](double v) { return int(std::floor(v*65536+0.5)); };
        for(int i=0;i<256;i++) {
            double c=(i-128)*cs;
            luma[i]=fixed((i-(full?0:16))*ys);
            redV[i]=fixed(2*(1-kr)*c); blueU[i]=fixed(2*(1-kb)*c);
            greenU[i]=fixed(-2*kb*(1-kb)/kg*c); greenV[i]=fixed(-2*kr*(1-kr)/kg*c);
        }
    }
    std::array<int,3> rgb(int y,int u,int v) const {
        auto channel=[](int n) { return std::clamp((n+32768)>>16,0,255); };
        return {channel(luma[y]+redV[v]),channel(luma[y]+greenU[u]+greenV[v]),channel(luma[y]+blueU[u])};
    }
};
const Color &color(int matrix,bool full) {
    static const Color tables[3][2]={{Color(.299,.114,false),Color(.299,.114,true)},
        {Color(.2126,.0722,false),Color(.2126,.0722,true)},
        {Color(.2627,.0593,false),Color(.2627,.0593,true)}};
    return tables[matrix][full?1:0];
}
struct Image {
    const uint8_t *y,*uv; int ys,uvs,left,top,w,h,rotation; const Color &conversion;
    std::array<int,3> pixel(int x,int row) const {
        int sx=x,sy=row;
        if(rotation==90) { sx=row; sy=h-1-x; }
        else if(rotation==180) { sx=w-1-x; sy=h-1-row; }
        else if(rotation==270) { sx=w-1-row; sy=x; }
        sx+=left; sy+=top;
        int offset=(sy/2)*uvs+(sx/2)*2;
        return conversion.rgb(y[sy*ys+sx],uv[offset],uv[offset+1]);
    }
};
bool valid(const uint8_t *y,int ys,const uint8_t *uv,int uvs,int left,int top,int w,int h,int rotation,int matrix) {
    return y && uv && left>=0 && top>=0 && w>0 && h>0 && ys>=left+w && uvs>=((left+w+1)/2)*2
        && (rotation==0 || rotation==90 || rotation==180 || rotation==270) && matrix>=0 && matrix<=2;
}
struct Span { int first; std::vector<double> weights; };
std::vector<Span> spans(int source,int target) {
    std::vector<Span> result; result.reserve(target);
    double scale=double(source)/target;
    for(int i=0;i<target;i++) {
        double start=i*scale,end=std::min(double(source),(i+1)*scale);
        Span s{int(std::floor(start)),{}};
        for(int x=s.first;x<int(std::ceil(end));x++) s.weights.push_back(std::max(0.0,std::min(end,double(x+1))-std::max(start,double(x))));
        result.push_back(std::move(s));
    }
    return result;
}
}

int vc_nv12_area(const uint8_t *y,int ys,const uint8_t *uv,int uvs,int left,int top,int w,int h,
                 int rotation,int matrix,int full,const double *roi,int ow,int oh,uint8_t *rgba) {
    if(!valid(y,ys,uv,uvs,left,top,w,h,rotation,matrix) || !roi || !rgba || ow<=0 || oh<=0) return -1;
    for(int i=0;i<4;i++) if(!std::isfinite(roi[i])) return -1;
    if(roi[0]<0 || roi[1]<0 || roi[2]<=0 || roi[3]<=0 || roi[0]+roi[2]>1 || roi[1]+roi[3]>1) return -1;
    int dw=(rotation==90 || rotation==270)?h:w,dh=(rotation==90 || rotation==270)?w:h;
    int x0=std::clamp(int(std::nearbyint(roi[0]*dw)),0,dw-1), y0=std::clamp(int(std::nearbyint(roi[1]*dh)),0,dh-1);
    int cw=std::clamp(int(std::nearbyint((roi[0]+roi[2])*dw)),x0+1,dw)-x0;
    int ch=std::clamp(int(std::nearbyint((roi[1]+roi[3])*dh)),y0+1,dh)-y0;
    bool transpose=rotation==90 || rotation==270;
    int sw=transpose?ch:cw,sh=transpose?cw:ch,rw=transpose?oh:ow,rh=transpose?ow:oh;
    int sx0=rotation==90?y0:rotation==180?w-x0-cw:rotation==270?w-y0-ch:x0;
    int sy0=rotation==90?h-x0-cw:rotation==180?h-y0-ch:rotation==270?x0:y0;
    // Resample in source orientation, then rotate the reduced result. Besides
    // contiguous source access this preserves Android's floating-point sums.
    Image image{y,uv,ys,uvs,left,top,w,h,0,color(matrix,full!=0)};
    auto columns=spans(sw,rw),rows=spans(sh,rh);
    double denominator=(double(sw)/rw)*(double(sh)/rh);
    for(int row=0;row<rh;row++) for(int x=0;x<rw;x++) {
        double sum[3]={0,0,0}; const auto &a=columns[x]; const auto &b=rows[row];
        for(size_t iy=0;iy<b.weights.size();iy++) for(size_t ix=0;ix<a.weights.size();ix++) {
            auto rgb=image.pixel(sx0+a.first+int(ix),sy0+b.first+int(iy)); double weight=b.weights[iy]*a.weights[ix];
            for(int c=0;c<3;c++) sum[c]+=rgb[c]*weight;
        }
        int dx=rotation==90?ow-1-row:rotation==180?ow-1-x:rotation==270?row:x;
        int dy=rotation==90?x:rotation==180?oh-1-row:rotation==270?oh-1-x:row;
        int offset=(dy*ow+dx)*4;
        for(int c=0;c<3;c++) rgba[offset+c]=uint8_t(std::clamp(int(std::nearbyint(sum[c]/denominator)),0,255));
        rgba[offset+3]=255;
    }
    return 0;
}

int vc_nv12_letterbox(const uint8_t *y,int ys,const uint8_t *uv,int uvs,int left,int top,int w,int h,
                      int rotation,int matrix,int full,const int *g,int size,float *rgb) {
    if(!valid(y,ys,uv,uvs,left,top,w,h,rotation,matrix) || !g || !rgb || size<=0) return -1;
    int x0=g[0],y0=g[1],cw=g[2],ch=g[3],rw=g[4],rh=g[5],padX=g[6],padY=g[7];
    int dw=(rotation==90 || rotation==270)?h:w,dh=(rotation==90 || rotation==270)?w:h;
    if(x0<0 || y0<0 || cw<=0 || ch<=0 || x0+cw>dw || y0+ch>dh || rw<=0 || rh<=0 || padX<0 || padY<0 || padX+rw>size || padY+rh>size) return -1;
    Image image{y,uv,ys,uvs,left,top,w,h,rotation,color(matrix,full!=0)};
    std::fill(rgb,rgb+size*size*3,0);
    for(int row=0;row<rh;row++) for(int x=0;x<rw;x++) {
        double sx=std::clamp((x+.5)*cw/rw-.5,0.0,double(cw-1)),sy=std::clamp((row+.5)*ch/rh-.5,0.0,double(ch-1));
        int ix=int(sx),iy=int(sy); float dx=float(sx-ix),dy=float(sy-iy);
        auto a=image.pixel(x0+ix,y0+iy),b=image.pixel(x0+std::min(cw-1,ix+1),y0+iy);
        auto c=image.pixel(x0+ix,y0+std::min(ch-1,iy+1)),d=image.pixel(x0+std::min(cw-1,ix+1),y0+std::min(ch-1,iy+1));
        for(int channel=0;channel<3;channel++) {
            float value=a[channel]*(1-dx)*(1-dy)+b[channel]*dx*(1-dy)+c[channel]*(1-dx)*dy+d[channel]*dx*dy;
            // A convex interpolation is in [0,255], but Float summation can
            // produce 255.000015 for saturated pixels. Preserve fractional
            // values while bounding that rounding error before normalization.
            rgb[((row+padY)*size+x+padX)*3+channel]=std::clamp(value,0.0f,255.0f);
        }
    }
    return 0;
}
