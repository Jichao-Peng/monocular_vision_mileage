//
// Created by leo on 18-5-28.
//

#ifndef MONOCULAR_VISION_MILEAGE_OPTIMIZE_H
#define MONOCULAR_VISION_MILEAGE_OPTIMIZE_H

#include <queue>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace cv;
using namespace std;

class KeyFrame
{
public:
    Mat image;
    Mat R_f;//关键帧相对于世界坐标系的矩阵
    Mat t_f;
    Mat R;//关键帧相对于前一关键帧帧的矩阵
    Mat t;
};

class Optimize
{
public:
    Optimize(int keyframe_num, float f, double px, double py);

    bool Run(const Mat &curimage, const Mat &R_f, const Mat &t_f, Mat& T);

    void SetFirstKeyFrame(const Mat &image);


private:
    Mat BundleAdjustment();
    void Triangulation(const vector<Point2f> &featurepoints1, const vector<Point2f> &featurepoints2,
                       const Mat &R, const Mat &t, const Mat &R_f, const Mat &t_f, vector<Point3f> &ponits, float f, double px, double py);
    bool IsKeyFrame(const Mat &curimage, int features_num, int threshold,vector<Point2f> &featurepoints1,vector<Point2f> &featurepoints2);
    void SetKeyFrame(const Mat &image, const Mat &R_f, const Mat &t_f, const Mat &R, const Mat &t);
    void SetKeyFeaturePoints(vector<Point2f> featurepoints1, vector<Point2f> featurepoints2);
    void SetKeyMarkPoints(vector<Point3f> markpoints);

    Point2f Pixel2Cam (const Point2d& p, const Matx33d& K);

    vector<KeyFrame> keyframes_buf;
    vector<vector<Point2f>> featurepoints_buf;
    vector<vector<Point3f>> markpoints_buf;

    Mat preimage;

    const int kKeyFrameNum;
    const float kF;
    const float kPx;
    const float kPy;
};


#endif //MONOCULAR_VISION_MILEAGE_OPTIMIZE_H
