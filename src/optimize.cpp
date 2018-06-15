//
// Created by leo on 18-5-28.
//

#include "optimize.h"

Optimize::Optimize(int keyframe_num, float f, double px, double py):kKeyFrameNum(keyframe_num),kF(f),kPx(px),kPy(py) {};


//BA优化传入当前图像，和前一帧的R，t，和世界坐标系的R_f,t_f,以及内参f，px，py，返回值是优化后的T
bool Optimize::Run(const Mat &curimage, const Mat &R_f, const Mat &t_f, Mat &T)
{
    vector<Point2f> featurepoints1,featurepoints2;
    if(IsKeyFrame(curimage,1000,200,featurepoints1,featurepoints2))//用当前帧和前一帧比较比较判断当前帧是否位关键帧，并且返回两帧匹配的关键点，1为前一关键帧，2为当前帧
    {
        //求前后两个关键帧之间的R，t
        Point2d p(kPx,kPy);
        Mat R,t;
        vector<uchar> status;

        //求本质矩阵
        Mat E = findEssentialMat(featurepoints1,featurepoints2,kF,p,RANSAC,0.999,1.0);
        //求旋转矩阵和平移向量
        recoverPose(E,featurepoints1,featurepoints2,R,t,kF,p);

        vector<Point3f> markponits;
        //利用前后两个关键帧求深度
        Triangulation(featurepoints1,featurepoints2,R,t,R_f,t_f,markponits,kF,kPx,kPy);

        SetKeyFrame(curimage,R_f,t_f,R,t);
        SetKeyFeaturePoints(featurepoints1,featurepoints2);
        SetKeyMarkPoints(markponits);

        T = BundleAdjustment();
        return true;
    }
    else
    {
        return false;
    }
}

Mat Optimize::BundleAdjustment()
{
    if((keyframes_buf.size() == kKeyFrameNum) && (markpoints_buf.size() == (kKeyFrameNum-1)) && (featurepoints_buf.size() == (kKeyFrameNum*2-2)))
    {
        cout<<keyframes_buf.size()<<endl<<markpoints_buf.size()<<endl<<featurepoints_buf.size()<<endl;
        // 初始化g2o
        typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
        std::unique_ptr<Block::LinearSolverType> linearSolver ( new g2o::LinearSolverCSparse<Block::PoseMatrixType>());
        std::unique_ptr<Block> solver_ptr ( new Block ( std::move(linearSolver)));     // 矩阵块求解器
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm ( solver );
        optimizer.setVerbose( false );

        // 添加位姿节点
        for ( int i=0; i<kKeyFrameNum; i++ )
        {
            g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
            Eigen::Matrix3d R_mat;
            Mat R = keyframes_buf.at(i).R_f;
            Mat t = keyframes_buf.at(i).t_f;
            R_mat <<
                    R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
                    R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
                    R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
            pose->setId(i);
            pose->setEstimate ( g2o::SE3Quat (
                    R_mat,
                    Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
            ) );
            if (i == 0)
                pose->setFixed(true); // 第一个点固定为零
            optimizer.addVertex(pose);
        }

        // 添加特征点的节点
        int index = 0;
        for( int i=0; i<kKeyFrameNum-1; i++)
        {
            vector<Point3f> markpoints = markpoints_buf.at(i);
            for (size_t j = 0; j < markpoints.size(); j++)
            {
                g2o::VertexSBAPointXYZ *mark = new g2o::VertexSBAPointXYZ();
                mark->setId(kKeyFrameNum + index + j);
                mark->setMarginalized(true);
                mark->setEstimate(Eigen::Vector3d(markpoints[i].x, markpoints[i].y, markpoints[i].z));
                optimizer.addVertex(mark);
            }
            index = index + markpoints.size();
        }

        // 准备相机参数
        g2o::CameraParameters* camera = new g2o::CameraParameters( kF, Eigen::Vector2d(kPx, kPy), 0 );
        camera->setId(0);
        optimizer.addParameter( camera );

        // 准备边
        for(int i=0; i<kKeyFrameNum-1; i++)
        {
            // 第一帧
            vector<g2o::EdgeProjectXYZ2UV *> edges;
            vector<Point2f> featurepoints1 = featurepoints_buf.at(i*2);
            vector<Point2f> featurepoints2 = featurepoints_buf.at(i*2+1);
            for (int j = 0; j < featurepoints1.size(); j++)
            {
                g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
                edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>   (optimizer.vertex(j + kKeyFrameNum + i)));
                edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>     (optimizer.vertex(i)));
                edge->setMeasurement(Eigen::Vector2d(featurepoints1[j].x, featurepoints1[j].y));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setParameterId(0, 0);
                // 核函数
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                optimizer.addEdge(edge);
                edges.push_back(edge);
            }
            // 第二帧
            for (int j = 0; j < featurepoints2.size(); j++)
            {
                g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();
                edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>   (optimizer.vertex(j + kKeyFrameNum + i)));
                edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>     (optimizer.vertex(i+1)));
                edge->setMeasurement(Eigen::Vector2d(featurepoints2[j].x, featurepoints2[j].y));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setParameterId(0, 0);
                // 核函数
                edge->setRobustKernel(new g2o::RobustKernelHuber());
                optimizer.addEdge(edge);
                edges.push_back(edge);
            }
        }
        cout<<"开始优化"<<endl;
        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cout<<"优化完毕"<<endl;

        //我们比较关心两帧之间的变换矩阵
        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(kKeyFrameNum - 1) );
        Eigen::Isometry3d pose = v->estimate();
        Mat T = Mat::zeros(4,4,CV_64F);
        Eigen::Matrix4d T_matrix = pose.matrix();
        for(int i=0; i<4; i++)
        {
            for(int j=0; j<4; j++)
            {
                T.at<double>(i,j) = T_matrix(i,j);
            }
        }
        return T;
    }
}



//三角化计算空间坐标,返回的points是相对于世界坐标系的坐标
void Optimize::Triangulation(const vector<Point2f> &featurepoints1, const vector<Point2f> &featurepoints2,
                             const Mat &R, const Mat &t, const Mat &R_f, const Mat &t_f, vector<Point3f> &ponits, float f, double px, double py)
{
    Matx34f T1(1,0,0,0,
               0,1,0,0,
               0,0,1,0);

    Matx34f T2(R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
               R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
               R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));

    Matx33d K(f, 0, px,
              0, f, py,
              0, 0, 1);


    vector<Point2f> points1, points2;
    for (int i = 0; i<featurepoints1.size(); i++)
    {
        // 将像素坐标转换至相机坐标
        points1.push_back ( Pixel2Cam(featurepoints1[i], K) );
        points2.push_back ( Pixel2Cam(featurepoints2[i], K) );
    }

    Mat points4d;//points4d输出的是points1的空间坐标
    cv::triangulatePoints( T1, T2, points1, points2, points4d );

    // 转换成非齐次坐标
    for ( int i=0; i<points4d.cols; i++ )
    {
        //第一张图像的关键点
        Mat x = points4d.col(i);
        x /= x.at<double>(3,0); // 归一化
        Point3d p1(x.at<double>(0,0),x.at<double>(1,0),x.at<double>(2,0));

        //第二张图像的关键点
        Mat y = R*(Mat_<double>(3,1) << p1.x,p1.y,p1.z) + t;
        Point3d p2(y.at<double>(0,0),y.at<double>(1,0),y.at<double>(2,0));

        //世界坐标系下的关键点的坐标
        Mat z = R_f.inv()*((Mat_<double>(3,1) << p2.x,p2.y,p2.z)-t_f);
        Point3d p3(z.at<double>(0,0),z.at<double>(1,0),z.at<double>(2,0));

        ponits.push_back(p3);
    }
}

//将像素坐标转化为空间坐标
Point2f Optimize::Pixel2Cam ( const Point2d& p, const Matx33d& K )
{
    return Point2f
    (
        ( p.x - K(0,2) ) / K(0,0),
        ( p.y - K(1,2) ) / K(1,1)
    );
}

//通过ORB法判断是否为关键帧
bool Optimize::IsKeyFrame(const Mat &curimage, int features_num, int threshold,vector<Point2f> &featurepoints1,vector<Point2f> &featurepoints2)
{
    //初始化
    vector<KeyPoint> keypoints1,keypoints2;
    Mat description1,description2;
    Ptr<ORB> orb = ORB::create(features_num);

    //检测Driented FAST角点位置
    orb->detect(keyframes_buf.at(0).image,keypoints1);
    orb->detect(curimage,keypoints2);

    //根据角点计算BRIEF描述子
    orb->compute(keyframes_buf.at(0).image,keypoints1,description1);
    orb->compute(curimage,keypoints2,description2);

    //暴力匹配 //FlannBasedMatcher 快速近似最近邻FLANN适合于匹配点数量极多的情况
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(description1,description2,matches);

    //匹配点筛选
    double min_dist = 10000, max_dist = 0;
    //找出所有匹配之间的最小距离和最大距离
    for(int i = 0; i<description1.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist)
        {
            min_dist = dist;
        }
        if(dist > max_dist)
        {
            max_dist = dist;
        }
    }
    vector<DMatch> good_matches;
    vector<KeyPoint> keypoints1_correction,keypoints2_correction;
    for(int i = 0; i < description1.rows; i++)
    {
        if(matches[i].distance <= max(2*min_dist,30.0))
        {
            good_matches.push_back(matches[i]);
            keypoints1_correction.push_back(keypoints1[matches[i].queryIdx]);
            keypoints2_correction.push_back(keypoints1[matches[i].trainIdx]);
            KeyPoint::convert(keypoints1_correction,featurepoints1,vector<int>());
            KeyPoint::convert(keypoints2_correction,featurepoints2,vector<int>());
        }
    }
    if(good_matches.size()>threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void Optimize::SetFirstKeyFrame(const Mat &image)
{
    Mat R_f = Mat::eye(3,3,CV_32F);
    Mat t_f = Mat::zeros(1,3,CV_32F);
    Mat R = Mat::eye(3,3,CV_32F);
    Mat t = Mat::zeros(1,3,CV_32F);
    SetKeyFrame(image,R_f,t_f,R,t);
}

//设置关键帧
void Optimize::SetKeyFrame(const Mat &image, const Mat &R_f, const Mat &t_f, const Mat &R, const Mat &t)
{
    KeyFrame keyframe;
    keyframe.image = image.clone();
    keyframe.R_f = R_f.clone();
    keyframe.t_f = t_f.clone();
    keyframe.R = R.clone();
    keyframe.t = t.clone();
    if(keyframes_buf.size()<kKeyFrameNum)
    {
        keyframes_buf.push_back(keyframe);
    }
    else
    {
        keyframes_buf.pop_back();
        keyframes_buf.push_back(keyframe);
    }
}

void Optimize::SetKeyFeaturePoints(vector<Point2f> featurepoints1, vector<Point2f> featurepoints2)
{
    if(featurepoints_buf.size()<kKeyFrameNum*2-2)
    {
        featurepoints_buf.push_back(featurepoints1);
        featurepoints_buf.push_back(featurepoints2);
    }
    else
    {
        featurepoints_buf.pop_back();
        featurepoints_buf.push_back(featurepoints1);
        featurepoints_buf.pop_back();
        featurepoints_buf.push_back(featurepoints2);
    }
}

void Optimize::SetKeyMarkPoints(vector<Point3f> markpoints)
{
    if(markpoints_buf.size()<kKeyFrameNum-1)
    {
        markpoints_buf.push_back(markpoints);
    }
    else
    {
        markpoints_buf.pop_back();
        markpoints_buf.push_back(markpoints);
    }
}

