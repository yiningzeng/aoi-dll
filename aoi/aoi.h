﻿// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

// 添加要在此处预编译的标头
#include "framework.h"
#include <opencv2/opencv.hpp>
// 图像拼接用的对齐边的定义
enum side { none = 0, left = 1, up = 2, right = 4, down = 8 };
#ifdef __cplusplus //(内置宏,如果是c++,在编译器预处理的时候加上extern,如果是c语言调用的时候是不处理的)
extern "C"
{
#endif

	__declspec(dllexport) void hello(cv::Mat& img);
	__declspec(dllexport) void copy_to(cv::Mat& img, cv::Mat& patch, const cv::Rect& roi_ref);

	// 抽色算子，对颜色通道设置上下限，输出掩码图
	// gray：灰度图
	// rgb：rgb彩色图
	// mask：输出的掩码图
	// params：参数数组，依次为灰度的阈值下限、上限，r 通道的下限、上限，g 下限、上限，b 下限、上限
	__declspec(dllexport) void range_mask(const cv::Mat& gray, const cv::Mat& rgb, cv::Mat& mask, const int* params);

	// 从灰度图生成直方图
	// gray：灰度图
	// n_bins：柱子的个数
	// hist：输出的直方图，n_bins x 1 的矩阵，类型为 CV_32F，hist.data 是数据指针，数值类型float32
	__declspec(dllexport) void histogram(const cv::Mat& gray, const int n_bins, cv::Mat& hist);


	// 图像拼接（简单的基于已知相对位置的拼接）
	// img: 拼接后的目标大图（输出），内存必须在调用之前预留好，不能为空
	// roi: 新加进去的小图在大图中的区域，矩形框
	// patch: 新加进去的单张小图
	__declspec(dllexport) void add_patch(cv::Mat& img, const cv::Rect& roi, const cv::Mat& patch);

	// 图像拼接，功能与前者相同，参数不同
	// tl: 新加进去的小图的左上角顶点在大图中的位置
	__declspec(dllexport) void add_patch_2(cv::Mat& img, const cv::Point& tl, const cv::Mat& patch);

	// 图像拼接
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
	__declspec(dllexport) int stitch_v2(cv::Mat& img, const cv::Rect& roi_ref, const cv::Mat& patch, cv::Rect& roi_patch, int side1, int overlap_lb1, int overlap_ub1, int drift_ub1, int side2 = side::none, int overlap_lb2 = 0, int overlap_ub2 = 0, int drift_ub2 = 0);




	// 旋转参数计算：通过 marker 位置做旋转对齐矫正时用到，需要两个点 p、q 在旋转前/后的对应坐标，p和q的坐标不能相同，否则会引起数值异常
	// _p, _q: 旋转前 p, q 的二维坐标(x,y)的指针
	// p_, q_: 旋转后的坐标
	// rot: 输出的参数，需要五个double，参数值用于计算旋转矩阵
	// return: 0：正常；-1：数值异常
	__declspec(dllexport) int get_rotation_parameters(const double* _p, const double* _q, const double* p_, const double* q_, double* rot);

	// 旋转矩阵计算：从旋转参数计算旋转变换矩阵
	// rot: 旋转参数，即 get_rotation_parameters 输出的数组
	// rmat: 旋转矩阵（输出）
	__declspec(dllexport) void get_rotation_matrix_2d(const double* rot, cv::Mat& rmat);

	// 旋转矩阵计算：get_rotation_parameters和get_rotation_matrix_2d的合成，直接从点对计算旋转矩阵；参数含义相同，此略
	__declspec(dllexport) int get_rotation_matrix_2d_2(const double* _p, const double* _q, const double* p_, const double* q_, cv::Mat& rmat);

	// 旋转变幻
	// src: 需要变换的图像
	// dst: 变换后的输出图像，需要预留内存，尺寸与src相同
	// rmat: 旋转矩阵，从get_rotation_matrix_2d得到
	__declspec(dllexport) void apply_rotation_transform(const cv::Mat& src, cv::Mat& dst, const cv::Mat& rmat);


	/////////////////////////////////////////////////////////////////
	//                   以下部分未完，待定                        //
	/////////////////////////////////////////////////////////////////

	// 求灰度图均值
	__declspec(dllexport) bool mean(const cv::Mat& gray, const int* params);

	// 像素最大，最小，范围判定
	__declspec(dllexport) bool min_max_range(const cv::Mat& gray, const int* params, unsigned mode = 0);

	// scale 算子
	__declspec(dllexport) bool scale(const cv::Mat& gray, const cv::Mat& rgb, const int* params);

	// 图像匹配算子
	__declspec(dllexport) double image_match(const cv::Mat& img, const cv::Mat& templ, cv::Point* pos = NULL, bool binarize = false);

	// 金线缺陷检测用的差分算子
	__declspec(dllexport) void take_diff(const cv::Mat& img, const cv::Mat& ref, cv::Mat& diff, int sensitivity = 3, int min_area = 9);
#ifdef __cplusplus
}
#endif
#endif //PCH_H
