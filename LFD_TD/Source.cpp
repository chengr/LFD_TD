// lens_flaw_detection_project_training_data.cpp : 定義主控台應用程式的進入點。
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cv.h>
#include <stdio.h>

#include <fstream> 
using namespace cv;
using namespace std;
double pi =3.141592653589793;
Mat find_circle_and_draw_rec(Mat src_img,int size);//輸入影像,輸出影像大小
int main(int argc, char** argv)
{
	//
	Mat src;
	Mat result;
	int num=657;
	for(int i=num;i<663;i++){
		ifstream f1("pic/DSC06"+to_string(i)+".jpg");
		if (f1) {
			src = imread("pic/DSC06"+to_string(i)+".jpg",1);
			result=find_circle_and_draw_rec(src,720);
			imwrite("pic/out/SQR"+to_string(i)+".jpg", result);
		}
	}
	namedWindow("Result", CV_WINDOW_AUTOSIZE );
	imshow("Result",result);
	waitKey(0);
	return 0;
}

Mat find_circle_and_draw_rec(Mat src_img,int size){
	Mat src_gray;
	//if exist
	if( !src_img.data )
	{return src_img; }
	/////////////////////////////////////////////////////////////////////////
	//////////////find circle////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////
	/// Convert it to gray
	cvtColor( src_img, src_gray, CV_RGB2GRAY );
	//sobel------------------------------------------------
	int scale = 2;
	int delta = 0;
	int ddepth = CV_16S;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );
	Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, src_gray );
	//namedWindow( "sobel", CV_WINDOW_AUTOSIZE );
	//imshow("sobel",src_gray);
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur( src_gray,src_gray, Size(9,9), 2, 2 );
	//namedWindow( "blur", CV_WINDOW_AUTOSIZE );
	//imshow("blur",src_gray);
	//find circle
	vector<Vec3f> circles;
	/// Apply the Hough Transform to find the circles 
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT,1,src_gray.rows/4, 200, 100, 900, 1100 );
	/// Draw the circles detected
	for( size_t i = 0; i < 1; i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle( src_gray , center, 3, Scalar(0,0,255), 10, 8, 0 );
		// circle outline
		circle( src_gray , center, radius, Scalar(0,0,255), 10, 8, 0 );
	}
	/////////////////////////////////////////////////////////////////////////
	//////////////new a square///////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////
	
	Mat result(size/2,size , CV_8UC3, Scalar(0,0,0));
	double h=cvRound(circles[0][0]);//A(h,k)圓心
	double k=cvRound(circles[0][1]);
	double r=cvRound(circles[0][2]);//radius
	double h2=0,k2=0;
	//x=Xm+r*cos(theata) y=Ym+r*sin(theata) 
	//(Xm,Ym)為圓心
	//r為半徑


	double dd=360/(double)size;
	for(int i=0;i<size;i++){
		double deg=(i*dd*pi)/180;
		double radius=(2*r)/size;
		for(int j=0;j<size/2;j++){
			h2=h+j*radius*cos(deg);//h
			k2=k+j*radius*sin(deg);//k
			Vec3b color=src_img.at<Vec3b>(Point(h2,k2));
			result.at<Vec3b>(Point(i,j)) = color;
		}

	}

	/*
	for(int i=0;i<size;i++){
		double deg=(i*dd*pi)/180;
		x=h+r*cos(deg);
		y=k+r*sin(deg);
		double a=(y-k)/(x-h);//y=ax; x=y/a;
		if(abs(a)>=1){//看y
			double ds=(y-k)/(double)size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				ty=k+j*ds;
				tx=h+j*ds/a;
				Vec3b color=src_img.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}
		else{//看X
			double ds=(x-h)/(double)size;
			double tx=0;
			double ty=0;
			for(int j=0;j<size;j++){
				tx=h+j*ds;
				ty=k+j*ds*a;
				Vec3b color=src_img.at<Vec3b>(Point(tx,ty));
				result.at<Vec3b>(Point(i,j)) = color;
			}
		}
	}*/
	//GaussianBlur( result,result, Size(9,9), 2, 2 );
	return result;
}
