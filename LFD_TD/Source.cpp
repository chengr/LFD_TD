// lens_flaw_detection_project_training_data.cpp : 定義主控台應用程式的進入點。
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <cv.h>
#include <stdio.h>
#include <fstream> 
using namespace cv;
using namespace std;
double pi =3.141592653589793;
Mat find_circle_and_draw_rec(Mat src_img,int size);//輸入影像,輸出影像大小
Mat diff(Mat i1);//二值化
int count_dn(Mat in);
int count_up(Mat in);
int main(int argc, char** argv)
{
	Mat src;
	Mat result;
	int num=657;
	//將圖片方型化
	/*
	for(int i=num;i<=884;i++){
		ifstream f1("pic/DSC06"+to_string(i)+".jpg");
		if (f1) {
			src = imread("pic/DSC06"+to_string(i)+".jpg",1);
			result=find_circle_and_draw_rec(src,1440);
			imwrite("pic/out/SQR"+to_string(i)+".jpg", result);
		}
	}
	*/
	//namedWindow("Result", CV_WINDOW_AUTOSIZE );
	//imshow("Result",result);
	//處理方型化的圖
	fstream fp;
	fp.open("out2.txt", ios::out);
	for(int i=num;i<=884;i++){
		ifstream f1("pic/out/SQR"+to_string(i)+".jpg");
		if (f1) {
			src = imread("pic/out/SQR"+to_string(i)+".jpg",1);
			src.convertTo(src, -1, 2.0, -100);// Enter the alpha value [1.0-3.0]: 2.2 Enter the beta value [0-100]: 50
			Mat element(3,3,CV_8U,Scalar(1));  
			erode(src,src,element); 
			Mat element2(3,3,CV_8U,Scalar(1));  
			dilate(src,src,element2); 
			//
			pyrMeanShiftFiltering(src, src, 5, 40, 4);  //dst,color,level
			String tmps="其他";
			if(i>=657&&i<=662){
				tmps="中央有高對比白點";
			}
			if(i>=738&&i<=745){
				tmps="中央有白點";
			}
			if(i>=663&&i<=678){
				tmps="外圈有高對比黑點";
			}
			if(i>=679&&i<=704){
				tmps="外圍有區域黑";
			}
			if(i>=708&&i<=730){
				tmps="外圍有高對比黑點或白點";
			}
			if(i>=731&&i<=737){
				tmps="有黑線";
			}
			if(i>=807&&i<=884){
				tmps="不規則雜點加線";
			}
			if(i>=770&&i<=806){
				tmps="標準";
			}
			if(i>=797&&i<=801){
				tmps="大範圍的雜色點";
			}

			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_1.jpg", src);
			int avg_up=count_up(src);
			int avg_dn=count_dn(src);
			src=diff(src);
			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_2.jpg", src);

			
			fp<<"id:"<<i<<"up:"<<avg_up<<"dn:"<<avg_dn<<endl;
			
		}
	}
	fp.close();
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
	GaussianBlur( src_gray,src_gray, Size(9,9), 2, 2 );
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
	
	Mat result(size/4,size , CV_8UC3, Scalar(0,0,0));
	double h=cvRound(circles[0][0]);//A(h,k)圓心
	double k=cvRound(circles[0][1]);
	double r=cvRound(circles[0][2]);//radius
	double h2=0,k2=0;
	double dd=360/(double)size;
	for(int i=0;i<size;i++){
		double deg=(i*dd*pi)/180;
		double radius=(4*r)/size;
		for(int j=0;j<size/4;j++){
			h2=h+j*radius*cos(deg);//h
			k2=k+j*radius*sin(deg);//k
			Vec3b color=src_img.at<Vec3b>(Point(h2,k2));
			result.at<Vec3b>(Point(i,j)) = color;
		}

	}
	return result;
}
Mat diff(Mat i1){
	//GaussianBlur( i2, i2, Size(9,9), 0, 0 );
	Mat result=i1;
	int up=27;
	int dn=-1;
	int up2=256;
	int dn2=200;
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			Vec3b color=i1.at<Vec3b>(Point(x,y));
			if(y>200&&y<340){
				if(color[2]<up&&color[2]>dn&&color[0]<up+25&&color[0]>dn&&color[1]<up&&color[1]>dn){
					color[0]=255;
					color[1]=255;
					color[2]=255;
				}
				else{
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}

			}
			else if(y<150&&y>50){
				if(color[0]<up2&&color[1]<up2&&color[2]<up2&&color[0]>dn2&&color[1]>dn2&&color[2]>dn2){
					color[0]=255;
					color[1]=255;
					color[2]=255;
				}
				else{
					color[0]=0;
					color[1]=0;
					color[2]=0;
				}
			}
			else{
				color[0]=0;
				color[1]=0;
				color[2]=0;
			}
			result.at<Vec3b>(Point(x,y)) = color;
		}
	}
	return result;
}
int count_up(Mat in){
	int avg=0;
	int c=0;
	for(int x=0;x<in.cols;x++){
		for(int y=0;y<in.rows;y++){
			if(y<150&&y>50){
				Vec3b color=in.at<Vec3b>(Point(x,y));
				avg=avg+color[0]+color[1]+color[2];
				c++;
			}
		}
	}

	return avg/c;
}
int count_dn(Mat in){
	int avg=0;
	int c=0;
	for(int x=0;x<in.cols;x++){
		for(int y=0;y<in.rows;y++){
			if(y>200&&y<340){
				Vec3b color=in.at<Vec3b>(Point(x,y));
				avg=avg+color[0]+color[1]+color[2];
				c++;
			}
		}
	}

	return avg/c;
}