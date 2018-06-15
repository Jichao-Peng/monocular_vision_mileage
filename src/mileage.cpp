//
// Created by leo on 18-5-26.
//

#include "mileage.h"

Mileage::Mileage(float f, double px, double py,int frame_max):kF(f),kPx(px),kPy(py),kFrameMax(frame_max)
{
    pOptimize = new Optimize(5,kF,kPx,kPy);
}

void Mileage::Run_LK()
{
    namedWindow("raw_image",1);
    namedWindow("result",1);
    Mat curimage,preimage;
    vector<Point2f> curfeatures,prefeatures;
    vector<uchar> status;
    Mat R, t;//相对
    Mat R_f,t_f;//绝对
    Mat T;
    Mat result = Mat::zeros(800,800,CV_8UC3);

    vector<double> scales = GetAbsoluteScale(kFrameMax,result);

    //利用第一第二张图片初始化
    Mat firstimage,secondimage;
    vector<Point2f> firstfeatures,secondfeatures;
    firstimage = GetImage(0);
    secondimage = GetImage(1);

    pOptimize->SetFirstKeyFrame(firstimage);

    //光流LK法
    FeatureDetection_FAST(firstimage,firstfeatures);
    FeatureTracking_LK(firstimage,secondimage,firstfeatures,secondfeatures,status);
    CalHomogeneous(secondfeatures,firstfeatures,R,t);

    R_f = R.clone();
    t_f = t.clone();

    preimage = secondimage.clone();
    prefeatures = secondfeatures;

    for(int frame_num = 2; frame_num < kFrameMax; frame_num++)
    {
        curimage = GetImage(frame_num);

        //光流LK法
        FeatureTracking_LK(preimage,curimage,prefeatures,curfeatures,status);
        //如果通过LK匹配出来的点数小于1000的话就需要重新进行角点检测并进行匹配
        if(curfeatures.size() < 1000)
        {
            FeatureDetection_FAST(preimage,prefeatures);
            FeatureTracking_LK(preimage,curimage,prefeatures,curfeatures,status);
        }
        //求齐次矩阵
        CalHomogeneous(curfeatures,prefeatures,R,t);

        double scale = scales.at(frame_num-2);

        if(( scale > 0.15 ) && ( abs(t.at<double>(2)) > abs(t.at<double>(1)) ) && ( abs(t.at<double>(2)) > abs(t.at<double>(0)) ) )
        {
            t_f = t_f + scale * (R_f*t);
            R_f = R*R_f;
        }

        //cout<<R_f<<endl<<t_f<<endl;
        //进行BA优化
        Mat T;
        if(pOptimize->Run(curimage,R_f,t_f,T))
        {
            if(T.cols == 4 && T.rows == 4)
            {
                R_f = T(Range(0, 3), Range(0, 3));
                t_f = T(Range(0, 3), Range(3, 4));
                //cout<<R_f<<endl<<t_f<<endl;
            }
        }


        //图片和特征点向前传递
        preimage = curimage.clone();
        prefeatures = curfeatures;

        //画出特征点
        Mat showimage = curimage;
        vector<Point2f>::iterator it;
        for(it = curfeatures.begin(); it != curfeatures.end(); it++)
        {
            circle(showimage, *it, 5, Scalar(255, 0, 0));
        }
        imshow("raw_image",showimage);

        //画出轨迹图
        int x = int(t_f.at<double>(0)) + 400;
        int y = int(t_f.at<double>(2)) + 700;
        circle(result, Point(x, y) ,1, Scalar(0,0,255), 2);
        imshow("result",result);

        waitKey(1);
    }
}

//读取绝对尺度值
vector<double> Mileage::GetAbsoluteScale(int frame_num,Mat& result)
{
    string line;
    int i = 0;
    ifstream file("/home/leo/Desktop/MachineVision_Homwork/monocular_vision_mileage/src/00.txt");
    double x=0,y=0,z=0;
    double x_prev,y_prev,z_prev;
    vector<double> scale;
    if(file.is_open())
    {
        while(getline(file,line)&&i<=frame_num)
        {
            x_prev = x;
            y_prev = y;
            z_prev = z;
            istringstream linein(line);
            for(int j=0; j<12; j++)
            {
                linein >> z;
                if(j == 3)
                {
                    x = z;
                }
                if(j == 7)
                {
                    y = z;
                }
            }
            circle(result, Point(x+400, -z+700) ,1, Scalar(0,255,0), 2);
            i++;
            scale.push_back(sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)));
        }
        file.close();
    }
    else
    {
        cout<<"Open file failed!"<<endl;
    }
    cout<<"Read all"<<endl;
    return scale;
}

//读取数据集图片
Mat Mileage::GetImage(int frame_num)
{
    char filename[100];
    sprintf(filename, "/home/leo/Desktop/MachineVision_Homwork/monocular_vision_mileage/image_0/%06d.png", frame_num);
    Mat image = imread(filename,0);
    return image;
}

void Mileage::CalHomogeneous(vector<Point2f> &features1, vector<Point2f> &features2, Mat& R, Mat& t)
{
    Point2d p(kPx,kPy);
    int f = kF;
    //求本质矩阵
    Mat E = findEssentialMat(features2,features1,f,p,RANSAC,0.999,1.0);
    //求旋转矩阵和平移向量
    recoverPose(E,features2,features1,R,t,f,p);
}

//FAST角点检测
void Mileage::FeatureDetection_FAST(Mat image, vector<Point2f> &points)
{
    vector<KeyPoint> keypoints;
    FAST(image,keypoints,20,true);
    KeyPoint::convert(keypoints,points,vector<int>());
}

//LK特征追踪
void Mileage::FeatureTracking_LK(Mat image1, Mat image2, vector<Point2f> &points1, vector<Point2f> &points2,
                              vector<uchar> &status)
{
    vector<float> err;
    try
    {
        calcOpticalFlowPyrLK(image1, image2, points1, points2, status, err);
    }
    catch(Exception)
    {
        cout<<"Some error happened!"<<endl;
    }

    int index_correctrion = 0;
    //除掉特征点里面没有匹配上的点
    for(int i=0; i<status.size(); i++)
    {
        Point2f point = points2.at(i - index_correctrion);
        if((status.at(i) == 0) || (point.x<0) || (point.y<0))
        {
            if((point.x<0) || (point.y<0))
            {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - index_correctrion));
            points2.erase(points2.begin() + (i - index_correctrion));
            index_correctrion++;
        }

    }
}

void Mileage::FeatureDetectionTracking_ORB(Mat image1, Mat image2, vector<Point2f> &points1,
                                           vector<Point2f> &points2)
{
    //初始化
    vector<KeyPoint> keypoints1,keypoints2;
    Mat description1,description2;
    Ptr<ORB> orb = ORB::create(2000);

    //检测Driented FAST角点位置
    orb->detect(image1,keypoints1);
    orb->detect(image1,keypoints2);

    //根据角点计算BRIEF描述子
    orb->compute(image1,keypoints1,description1);
    orb->compute(image2,keypoints2,description2);

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

    cout<<min_dist<<endl;
    //当描述子之间的距离大于两倍的最小距离时，即认为匹配有误。
    //但是有时候最小距离会非常小，设置一个经验值作为下限
    vector<KeyPoint> keypoints1_correction,keypoints2_correction;
    for(int i = 0; i < description1.rows; i++)
    {
        if(matches[i].distance <= max(3*min_dist,50.0))
        {
            keypoints1_correction.push_back(keypoints1[matches[i].queryIdx]);
            keypoints2_correction.push_back(keypoints1[matches[i].trainIdx]);
        }
    }
    cout<<"point1 "<<keypoints1_correction.size()<<endl<<"point2 "<<keypoints2_correction.size()<<endl;
    KeyPoint::convert(keypoints1_correction,points1,vector<int>());
    KeyPoint::convert(keypoints2_correction,points2,vector<int>());
}

void Mileage::Run_ORB()
{
    namedWindow("raw_image",1);
    namedWindow("result",1);
    Mat curimage,preimage;
    vector<Point2f> curfeatures,prefeatures;
    vector<uchar> status;
    Mat R, t;//相对
    Mat R_f,t_f;//绝对
    Mat result = Mat::zeros(600,600,CV_8UC3);

    //利用第一第二张图片初始化
    Mat firstimage,secondimage;
    vector<Point2f> firstfeatures,secondfeatures;
    firstimage = GetImage(0);
    secondimage = GetImage(1);

    FeatureDetectionTracking_ORB(firstimage,secondimage,firstfeatures,secondfeatures);
    CalHomogeneous(secondfeatures,firstfeatures,R,t);

    R_f = R.clone();
    t_f = t.clone();

    for(int frame_num = 2; frame_num < kFrameMax; frame_num++)
    {
        preimage = GetImage(frame_num-1);
        curimage = GetImage(frame_num);

        //ORB求特征点
        FeatureDetectionTracking_ORB(preimage,curimage,prefeatures,curfeatures);
        //求齐次矩阵
        CalHomogeneous(curfeatures,prefeatures,R,t);


        if( ( abs(t.at<double>(2)) > abs(t.at<double>(1)) ) && ( abs(t.at<double>(2)) > abs(t.at<double>(0)) ) )
        {
            t_f = t_f + (R_f*t);
            R_f = R*R_f;
        }

        //cout<<t_f<<endl;

        //画出特征点
        Mat showimage = curimage;
        vector<Point2f>::iterator it;
        for(it = curfeatures.begin(); it != curfeatures.end(); it++)
        {
            circle(showimage, *it, 5, Scalar(255, 0, 0));
        }
        imshow("raw_image",showimage);

        //画出轨迹图
        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 500;
        circle(result, Point(x, y) ,1, Scalar(0,0,255), 2);
        imshow("result",result);


        waitKey(1);
    }
}