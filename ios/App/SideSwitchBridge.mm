#include "SideSwitchBridge.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace {
constexpr int W=256,H=144,N=W*H;
cv::Mat hanning(const cv::Mat &source) {
    cv::Mat result(source.rows, source.cols, CV_32F);
    for(int y=0;y<source.rows;++y) {
        const double vertical=.5*(1-std::cos(2*CV_PI*y/(source.rows-1)));
        for(int x=0;x<source.cols;++x) {
            const double horizontal=.5*(1-std::cos(2*CV_PI*x/(source.cols-1)));
            result.at<float>(y,x)=static_cast<float>(source.at<uint8_t>(y,x)*std::sqrt(vertical*horizontal));
        }
    }
    return result;
}
cv::Mat alignmentWindow(const cv::Mat &bgr) {
    cv::Mat gray,blur;
    cv::cvtColor(bgr,gray,cv::COLOR_BGR2GRAY);
    // Android Math.round(144*.42f)==60; Gaussian ROI retains its parent-border semantics.
    cv::Mat roi=gray(cv::Rect(0,0,W,60));
    cv::GaussianBlur(roi,blur,cv::Size(7,7),0);
    return hanning(blur);
}
}
int vc_switch_net_line(const uint8_t *bgr,double *result) {
    try {
        cv::Mat source(H,W,CV_8UC3,const_cast<uint8_t *>(bgr)),gray,blur,edges;
        cv::cvtColor(source,gray,cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray,blur,cv::Size(5,5),0);cv::Canny(blur,edges,45,110);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges,lines,1,CV_PI/180,std::max(24,W/10),W*.32,W*.12);
        bool found=false;
        for(const auto &line:lines) {
            const int dx=line[2]-line[0],dy=line[3]-line[1];
            const double coverage=double(std::abs(dx))/W,position=double(line[1]+line[3])/(2*H);
            const double slope=double(std::abs(dy))/std::max(std::abs(dx),1);
            if(position<.25||position>.68||slope>.08||coverage<.3)continue;
            const double score=std::min(1.0,coverage*(1+.35*position)+.1*std::hypot(dx,dy)/W);
            if(!found||score>result[1]){result[0]=position;result[1]=score;found=true;}
        }
        return found?1:0;
    }catch(const cv::Exception &){return -1;}
}
int vc_switch_prepare(const uint8_t *frames,double net,uint8_t *grays,uint8_t *hsvs,double *quality) {
    try {
        net=std::clamp(net,.2,.8);
        cv::Mat mapX(H,W,CV_32F),mapY(H,W,CV_32F);
        for(int y=0;y<H;++y){
            const double outputY=double(y)/(H-1);
            const double sourceY=outputY<=.5?outputY*net/.5:net+(outputY-.5)*(1-net)/.5;
            for(int x=0;x<W;++x){mapX.at<float>(y,x)=float(x);mapY.at<float>(y,x)=float(sourceY*(H-1));}
        }
        std::vector<cv::Mat> normalized(7);
        for(int i=0;i<7;++i){
            cv::Mat input(H,W,CV_8UC3,const_cast<uint8_t *>(frames+i*N*3));
            cv::remap(input,normalized[i],mapX,mapY,cv::INTER_LINEAR,cv::BORDER_REPLICATE,cv::Scalar::all(0));
        }
        const cv::Mat reference=alignmentWindow(normalized[3]);
        quality[0]=0;quality[1]=std::numeric_limits<double>::infinity();
        for(int i=0;i<7;++i){
            cv::Mat aligned=normalized[i];
            try {
                cv::Mat window=alignmentWindow(normalized[i]);double response=0;
                const cv::Point2d shift=cv::phaseCorrelate(reference,window,cv::noArray(),&response);
                const double magnitude=std::hypot(shift.x/W,shift.y/H);
                if(!std::isfinite(response))response=0;
                quality[1]=std::min(quality[1],response);
                if(std::isfinite(magnitude)&&response>=.02&&magnitude<=.12){
                    const cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-shift.x,0,1,-shift.y);
                    cv::Mat output;cv::warpAffine(normalized[i],output,transform,cv::Size(W,H),cv::INTER_LINEAR,cv::BORDER_REFLECT,cv::Scalar::all(0));
                    aligned=output;quality[0]=std::max(quality[0],magnitude);
                }
            }catch(const cv::Exception &){quality[1]=std::min(quality[1],0.0);}
            cv::Mat gray(H,W,CV_8UC1,grays+i*N),hsv(H,W,CV_8UC3,hsvs+i*N*3);
            cv::cvtColor(aligned,gray,cv::COLOR_BGR2GRAY);cv::cvtColor(aligned,hsv,cv::COLOR_BGR2HSV);
        }
        if(!std::isfinite(quality[1]))quality[1]=0;
        return 0;
    }catch(const cv::Exception &){return -1;}
}
