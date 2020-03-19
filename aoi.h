#ifndef __AOI_H__
#define __AOI_H__
#include <opencv2/opencv.hpp>

// 抽色算子，对颜色通道设置上下限，输出掩码图
// gray：灰度图
// rgb：rgb彩色图
// mask：输出的掩码图
// params：参数数组，依次为灰度的阈值下限、上限，r 通道的下限、上限，g 下限、上限，b 下限、上限
void range_mask(const cv::Mat& gray, const cv::Mat& rgb, cv::Mat& mask, const int* params);

// 从灰度图生成直方图
// gray：灰度图
// n_bins：柱子的个数
// hist：输出的直方图，n_bins x 1 的矩阵，类型为 CV_32F，hist.data 是数据指针，数值类型float32
void histogram(const cv::Mat& gray, const int n_bins, cv::Mat& hist);


// 图像拼接（简单的基于已知相对位置的拼接）
// img: 拼接后的目标大图（输出），内存必须在调用之前预留好，不能为空
// roi: 新加进去的小图在大图中的区域，矩形框
// patch: 新加进去的单张小图
void add_patch(cv::Mat& img, const cv::Rect& roi, const cv::Mat& patch);

// 图像拼接，功能与前者相同，参数不同
// tl: 新加进去的小图的左上角顶点在大图中的位置
void add_patch(cv::Mat& img, const cv::Point& tl, const cv::Mat& patch);


// 图像拼接用的对齐边的定义
enum side { none=0, left=1, up=2, right=4, down=8 };

// 图像对齐（使用重叠区域大小作为输入参数）
// img: 拼接后的大图（输出），内存需预留，尺寸足够大，保证新接入的图能够完全放入并留有余量
// roi_ref: 用做对齐用的（在大图中已经填充了的）参考区域矩形框，框的范围必须在img尺寸范围内，否则会出错
// patch: 新加进去的单张小图
// roi_patch: 新加的小图在大图中的位置（输出）
// side1: 小图用于对齐的边的位置，必须；数值参考enum side
// overlap_lb1: 图像重叠区域尺寸的下限，以像素为单位。（至少重叠 overlap_lb 个像素）
// overlap_ub1: 图像重叠区域尺寸的上限，以像素为单位。（至多重叠 overlap_ub 个像素）
// drift_ub1: 在拼接方向的垂直方向上的错位的上限，以像素为单位。
// side2: 使用两边做对齐时的第二个对齐边，用法用side1；默认为0，即使用单边对齐
// overlap_lb2, overlap_ub2, drift_ub2 对应于第二个对齐边相应的值, side2=0时无效
// return: 0：正常；-1：处理中发生异常（此前会导致程序中断）
//
// 举例，如需把patch拼到img中roi_ref的右侧，如下图，此时
// patch中用于对齐的边为左，即side1=side::left
// overlap和drift如图所示
// overlap_lb, overlap_ub, drift_ub 需要根据机械精度做相应的设置
//  _____________________________________________________________
// |                         ____________________  ____          |
// |   img    ______________|_____               | ____ drift    |
// |         |roi_ref       |     |              |               |
// |         |              |     |              |               |
// |         |              |     |              |               |
// |         |              |     |  patch       |               |
// |         |              |     |              |               |
// |         |              |     |              |               |
// |         |              |     |    roi_patch |               |
// |         |              |_____|______________|               |
// |         |____________________|                              |
// |                                                             |
// |                        |<--->| overlap                      |
// |_____________________________________________________________|
// 
int stitch(cv::Mat& img, const cv::Rect& roi_ref, const cv::Mat& patch, cv::Rect& roi_patch, int side1, int overlap_lb1, int overlap_ub1, int drift_ub1, int side2=side::none, int overlap_lb2=0, int overlap_ub2=0, int drift_ub2=0);

// 图像对齐（使用候选区域左上顶点作为输入参数）
// img, patch, side1, overlap_lb1, side2, overlap_lb2, return 意义同上，此略
// tl_candidate: 对需要拼接的图像预置的候选区域的左上角顶点位置
// tl：tl_candidate经算法对齐调整后的值（输出）
// err_ub: 误差上限，以像素为单位。（tl与tl_candidate在x/y方向上至多偏差err_ub个像素）
// -err_ub <= (tl - tl_candidate).x <= err_ub && -err_ub <= (tl - tl_candidate).y <= err_ub
//
//  _____________________________________________________________
// |                                                             |
// |                   tl_candidate: 候选区域左上角顶点坐标      |
// |                        |                                    |
// |                       \|/                                   |
// |                        .____________________                |
// |   img    ______________|_____               |               |
// |         |roi_ref       |     |              |               |
// |         |              |     |              |               |
// |         |              |     |              |               |
// |         |              |     |  patch       |               |
// |         |              |     |              |               |
// |         |              |     |              |               |
// |         |              |     |    roi_patch |               |
// |         |              |_____|______________|               |
// |         |____________________|                              |
// |                                                             |
// |_____________________________________________________________|
// 
int stitch(cv::Mat& img, const cv::Point& tl_candidate, const cv::Mat& patch, cv::Point& tl, int err_ub, int side1, int overlap_lb1, int side2=side::none, int overlap_lb2=0);

// 旋转参数计算：通过 marker 位置做旋转对齐矫正时用到，需要两个点 p、q 在旋转前/后的对应坐标，p和q的坐标不能相同，否则会引起数值异常
// _p, _q: 旋转前 p, q 的二维坐标(x,y)的指针
// p_, q_: 旋转后的坐标
// rot: 输出的参数，需要五个double，参数值用于计算旋转矩阵
// return: 0：正常；-1：数值异常
int get_rotation_parameters(const double* _p, const double* _q, const double* p_, const double* q_, double* rot);

// 旋转矩阵计算：从旋转参数计算旋转变换矩阵
// rot: 旋转参数，即 get_rotation_parameters 输出的数组
// rmat: 旋转矩阵（输出）
void get_rotation_matrix_2d(const double* rot, cv::Mat& rmat);

// 旋转矩阵计算：get_rotation_parameters和get_rotation_matrix_2d的合成，直接从点对计算旋转矩阵；参数含义相同，此略
int get_rotation_matrix_2d(const double* _p, const double* _q, const double* p_, const double* q_, cv::Mat& rmat);

// 旋转变幻
// src: 需要变换的图像
// dst: 变换后的输出图像，需要预留内存，尺寸、类型与src相同
// rmat: 旋转矩阵，从get_rotation_matrix_2d得到
void apply_rotation_transform(const cv::Mat& src, cv::Mat& dst, const cv::Mat& rmat);

// 图像分割（根据颜色分区）
// src: 输入的待分割图像
// dst: 输出的分割结果，需要预留内存，尺寸、类型与src相同
// N: 颜色种类个数（默认为2：前景/背景）
void segment(const cv::Mat& src, cv::Mat& dst, unsigned N = 2);

// 图像匹配算子
// img: 待搜索的图像
// templ: 目标模板
// pos: 匹配结果（输出），目标模板templ在img内的区域的左上角顶点位置
// binarize: 是否在预处理中加二值化操作，默认不进行(false)
// method: 匹配度指标
// return: 匹配结果得分
//     method使用默认时，值不大于1；得分越高说明匹配程度越高；反之，接近0可认为基本不匹配
//     method=1时，值不小于0；得分越低说明匹配程度越高
double image_match(const cv::Mat& img, const cv::Mat& templ, cv::Point* pos=NULL, bool binarize=false, int method=5);

/////////////////////////////////////////////////////////////////
//                   以下部分未完，待定                        //
/////////////////////////////////////////////////////////////////

// 求灰度图均值
bool mean(const cv::Mat& gray, const int* params);

// 像素最大，最小，范围判定
bool min_max_range(const cv::Mat& gray, const int* params, unsigned mode=0);

// scale 算子
bool scale(const cv::Mat& gray, const cv::Mat& rgb, const int* params);

// 金线缺陷检测用的差分算子
void take_diff(const cv::Mat& img, const cv::Mat& ref, cv::Mat& diff, int sensitivity=3, int min_area=9);

#endif /* ifndef __AOI_H__ */
