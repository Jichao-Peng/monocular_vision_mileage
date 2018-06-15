//
// Created by leo on 18-5-26.
//

#ifndef MONOCULAR_VISION_MILEAGE_MILEAGE_H
#define MONOCULAR_VISION_MILEAGE_MILEAGE_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <string>
#include <iostream>
#include <fstream>

#include "optimize.h"

using namespace cv;
using namespace std;

class Mileage
{
public:
    Mileage(float f, double px, double dy, int frame_max);
    void Run_LK();
    void Run_ORB();

private:
    Mat GetImage(int frame_num);
    vector<double> GetAbsoluteScale(int frame_num,Mat& result);

    void FeatureDetection_FAST(Mat image,vector<Point2f>& points);
    void FeatureTracking_LK(Mat image1,Mat image2,vector<Point2f>& points1,vector<Point2f>& points2,vector<uchar>& status);
    void FeatureDetectionTracking_ORB(Mat image1,Mat image2,vector<Point2f>& points1,vector<Point2f>& points2);
    void CalHomogeneous(vector<Point2f>& features1,vector<Point2f>& features2,Mat& R,Mat& t);

    Optimize* pOptimize;

    const float kF;
    const double kPx;
    const double kPy;
    const int kFrameMax;
};


#endif //MONOCULAR_VISION_MILEAGE_MILEAGE_H
