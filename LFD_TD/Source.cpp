// lens_flaw_detection_project_training_data.cpp : 定義主控台應用程式的進入點。
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "ml.h"
#include <cv.h>
#include <stdio.h>
#include <fstream> 
using namespace cv;
using namespace std;
double pi =3.141592653589793;
Mat find_circle_and_draw_rec(Mat src_img,int size);//輸入影像,輸出影像大小
Mat diff(Mat i1);//二值化
Mat draw_f(Mat i1,Mat i2);
Mat draw_O(Mat i1,Mat i2);
Mat find_circle(Mat src_img,int size,int min,int max);
int count_dn(Mat in);
int count_up(Mat in);
int count_dn_a(Mat in);
int count_up_a(Mat in);


RNG rng(12345);
int main(int argc, char** argv)
{
	Mat src,src2;
	Mat result;
	int num=657;
	//將圖片方型化for(int i=num;i<=884;i++){
	/*
	for(int i=num;i<=884;i++){
		ifstream f1("pic/TEST/DSC06"+to_string(i)+".jpg");
		if (f1) {
			src = imread("pic/TEST/DSC06"+to_string(i)+".jpg",1);
			result=find_circle_and_draw_rec(src,1440);
			imwrite("pic/out/SQR"+to_string(i)+".jpg", result);
		}
	}
	for(int i=num;i<=884;i++){
		ifstream f1("pic/training/DSC06"+to_string(i)+".jpg");
		if (f1) {
			src = imread("pic/training/DSC06"+to_string(i)+".jpg",1);
			result=find_circle_and_draw_rec(src,1440);
			imwrite("pic/out_training/SQR"+to_string(i)+".jpg", result);
		}
	}
	*/
	//namedWindow("Result", CV_WINDOW_AUTOSIZE );
	//imshow("Result",result);
	//處理方型化的圖
	
	fstream fp,fp2;
	fp.open("training0601.txt", ios::out);
	for(int i=num;i<=884;i++){
		ifstream f1("pic/out_training/SQR"+to_string(i)+".jpg");
		if (f1) {
			cout<<i<<endl;
			src = imread("pic/out_training/SQR"+to_string(i)+".jpg",1);
			src2=imread("pic/out_training/SQR"+to_string(i)+".jpg",1);
			String tmps="其他";
			int tmp_i=0;
			if(i>=657&&i<=662){
				tmps="中央有高對比白點";
				tmp_i=1;
			}
			if(i>=738&&i<=745){
				tmps="中央有白點";
			}
			if(i>=663&&i<=678){
				tmps="外圈有高對比黑點";
				tmp_i=2;
			}
			if(i>=679&&i<=704){
				tmps="外圍有區域黑";
				tmp_i=2;
			}
			if(i>=708&&i<=730){
				tmps="外圍有高對比黑點或白點";
				tmp_i=2;
			}
			if(i>=731&&i<=737){
				tmps="有黑線";
			}
			if(i>=807&&i<=884){
				tmps="不規則雜點加線";
				tmp_i=6;
			}
			if(i>=770&&i<=806){
				tmps="標準";
				tmp_i=5;
			}
			if(i>=797&&i<=801){
				tmps="大範圍的雜色點";
			}
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_1org.jpg", src);
			Mat water_o;//----
			water_o=src.clone();//----
			medianBlur(water_o,water_o,9);
			water_o.convertTo(water_o,-1,  1.1, -30);
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_2after_blur+CV.jpg", water_o);
			blur(result, result, Size(9,9) );
			medianBlur(src,src,9);
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_2after_blur.jpg", src);
			
			
			//---------------------------------------------------------------------------------

			src.convertTo(src, -1, 0.3, 30);
			
			
			//---------------------------------------------------------------------------------
			cvtColor( src, src, CV_RGB2GRAY );
			Mat grad_x, grad_y;
			//Y
			Sobel( src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
			convertScaleAbs( grad_y, grad_y );
			//X
			Sobel( src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
			convertScaleAbs(grad_x, grad_x );
			addWeighted( grad_x, 0.99, grad_y, 0.01,0, src ); 
			src.convertTo(src, -1, 65, -300);
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_3sobel.jpg", src);

			
			Mat element(3,3,CV_8U,Scalar(1));  
			erode(src,src,element); 
			medianBlur(src,src,9);
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_4erode.jpg", src);
			//---------------------------------------------------------------------------------
			int avg_up=count_up(src);
			int avg_dn=count_dn(src);
			int avg_up_a=count_up_a(src);
			int avg_dn_a=count_dn_a(src);
			fp<<tmp_i<<" "<<avg_up<<" "<<avg_dn <<" "<<avg_up_a<<" "<<avg_dn_a<<endl;
			//---------------------------------------------------------------------------------
			int threshold_value = 20;//15
			int threshold_type = 0;
			int const max_BINARY_value = 255;
			threshold( src, src, threshold_value, max_BINARY_value,threshold_type );
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_5threshold.jpg", src);
			Mat elements1(15,15,CV_8U,Scalar(1));  //50
			dilate(src,src,elements1); 
			src=draw_f(src,src);
			Canny( src, src, 88, 88*1, 3 );

			Mat markers(src.size(),CV_8U,cv::Scalar(0));//-------
			

			vector<vector<Point> > contours;
		    vector<Vec4i> hierarchy;
			findContours( src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			for( int i = 0; i< contours.size(); i++ )
			{
			Scalar color = Scalar( 0, 0, 255 );
			//drawContours( src2, contours, i, color, 2, 8, hierarchy, 0, Point() );
 
			cv::Moments mom= cv::moments( contours[i]); 
			// draw mass center  
			//circle(src2,Point(mom.m10/mom.m00,mom.m01/mom.m00) , 2,color,2); // draw black dot 
			circle(markers,Point(mom.m10/mom.m00,mom.m01/mom.m00) , 2,Scalar( 255, 255, 255 ),2);//----
			
			}
			//imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_1draw_red.jpg", src2);


			
			Mat bg(src.size(),CV_8U,cv::Scalar(0));//-------
			bg=markers.clone();
			Mat elementbg(120,120,CV_8U,Scalar(1));  //-------
			dilate(bg,bg,elementbg); //-------
			threshold(bg,bg,1,128,cv::THRESH_BINARY_INV);//-------
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_waterBG.jpg",bg);
			markers=markers+bg;
			markers.convertTo(markers,CV_32S);//-------
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_waterM.jpg", markers);//-------
			watershed(water_o,markers);//-------

			//threshold( markers, markers, 250, 255,0);
			markers.convertTo(markers,CV_8U);
			threshold( markers, markers, 250, 255,0);
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_waterS.jpg", markers);//-------
			
			//src2=draw_O(src2,markers);
			//markers=draw_f(markers,markers);
			findContours( markers, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			for( int i = 0; i< contours.size(); i++ )
			{
				Scalar color = Scalar( 0, 0, 255 );
				drawContours( src2, contours, i, color, 2, 8, hierarchy, 0, Point() );
			}
			imwrite("pic/out2_training/"+tmps+"/F"+to_string(i)+"_1draw_red.jpg", src2);
		}
	}
	fp.close();
	fp2.open("test0601.txt", ios::out);
	for(int i=num;i<=884;i++){
		ifstream f1("pic/out/SQR"+to_string(i)+".jpg");
		if (f1) {
			cout<<i<<endl;
			src = imread("pic/out/SQR"+to_string(i)+".jpg",1);
			src2=imread("pic/out/SQR"+to_string(i)+".jpg",1);
			String tmps="其他";
			int tmp_i=0;
			if(i>=657&&i<=662){
				tmps="中央有高對比白點";
				tmp_i=1;
			}
			if(i>=738&&i<=745){
				tmps="中央有白點";
			}
			if(i>=663&&i<=678){
				tmps="外圈有高對比黑點";
				tmp_i=2;
			}
			if(i>=679&&i<=704){
				tmps="外圍有區域黑";
				tmp_i=2;
			}
			if(i>=708&&i<=730){
				tmps="外圍有高對比黑點或白點";
				tmp_i=2;
			}
			if(i>=731&&i<=737){
				tmps="有黑線";
			}
			if(i>=807&&i<=884){
				tmps="不規則雜點加線";
				tmp_i=6;
			}
			if(i>=770&&i<=806){
				tmps="標準";
				tmp_i=5;
			}
			if(i>=797&&i<=801){
				tmps="大範圍的雜色點";
			}
			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_1org.jpg", src);
			//---------------------------------------------------------------------------------
			//src.convertTo(src, -1, 0.3, 30);
			blur(result, result, Size(9,9) );
			//pyrMeanShiftFiltering(src, src, 2, 100, 1);
			medianBlur(src,src,9);
			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_2after_blur.jpg", src);
			src.convertTo(src, -1, 0.3, 30);
			//---------------------------------------------------------------------------------
			cvtColor( src, src, CV_RGB2GRAY );
			
			Mat grad_x, grad_y;
			//Y
			Sobel( src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
			convertScaleAbs( grad_y, grad_y );
			//X
			Sobel( src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
			convertScaleAbs(grad_x, grad_x );
			addWeighted( grad_x, 0.99, grad_y, 0.01,0, src ); 
			src.convertTo(src, -1, 65, -300);
			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_3sobel.jpg", src);
			Mat element(3,3,CV_8U,Scalar(1));  
			erode(src,src,element); 
			medianBlur(src,src,9);
			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_4erode.jpg", src);
			//---------------------------------------------------------------------------------
			int avg_up=count_up(src);
			int avg_dn=count_dn(src);
			int avg_up_a=count_up_a(src);
			int avg_dn_a=count_dn_a(src);
			fp2<<tmp_i<<" "<<avg_up<<" "<<avg_dn <<" "<<avg_up_a<<" "<<avg_dn_a<<endl;




			int threshold_value = 15;
			int threshold_type = 0;
			int const max_BINARY_value = 255;
			threshold( src, src, threshold_value, max_BINARY_value,threshold_type );
			
			
			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_5threshold.jpg", src);
			Mat elements1(60,60,CV_8U,Scalar(1));  
			dilate(src,src,elements1); 
			src=draw_f(src,src);
			Canny( src, src, 88, 88*1, 3 );
			vector<vector<Point> > contours;
		    vector<Vec4i> hierarchy;
			findContours( src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
			for( int i = 0; i< contours.size(); i++ )
			{
			Scalar color = Scalar( 0, 0, 255 );
			drawContours( src2, contours, i, color, 2, 8, hierarchy, 0, Point() );
			}
			//imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_1threshold.jpg", src);
			//---------------------------------------------------------------------------------
			//Canny( src, src, 88, 88*1, 3 );
			imwrite("pic/out2/"+tmps+"/F"+to_string(i)+"_1draw_red.jpg", src2);
			
		}
	}
	fp2.close();

	
	//---------------------------------------------------------------------------------------------------------------------------------------
   
	//---------------------------------------------------------------------------------------------------------------------------------------
	
	ifstream inputFile("training0601.txt");
	float t_class [47];
	float t_feature [47][4];
	int noth=0;
	for(int i=0;i<47;i++)
	{
		inputFile >> t_class[i];
		//inputFile >> noth;
		inputFile >> t_feature[i][0];
		inputFile >> t_feature[i][1];
		inputFile >> t_feature[i][2];
		inputFile >> t_feature[i][3];
	}

	for(int i=0;i<48;i++){
		//cout<<t_feature[i][0]<<","<<t_feature[i][1]<<","<<t_feature[i][2]<<","<<t_feature[i][3]<<","<<endl;
	}

	// Set up training data
    Mat labelsMat(47, 1, CV_32FC1, t_class);
    Mat trainingDataMat(47, 4, CV_32FC1, t_feature);
	// Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

	fstream fp3;
	fp3.open("accuracy.txt", ios::out);

	ifstream inputFile2("test0601.txt");
	int c1=0;
	int f1=0;
	int f2=0;
	int f3=0;
	int f0=0;
	float c_O=0;
	float c_X=0;
	for(int i=0;i<48;i++)
	{
		inputFile2 >> c1;
		//inputFile2 >> noth;
		inputFile2 >> f0;
		inputFile2 >> f1;
		inputFile2 >> f2;
		inputFile2 >> f3;
		Mat sampleMat = (Mat_<float>(1,4) << f0,f1,f2,f3);
		float response = SVM.predict(sampleMat);
		cout<<"True:"<<c1<<"RESPONSE:"<<response<<endl;
		string answer="X";
		if(c1==response){
			answer="O";
			c_O++;
		}
		else
			c_X++;

		fp3<<"True:"<<c1<<"   RESPONSE:"<<response<<"  accuracy:"<<answer<<endl;

	}
	fp3<<"O:"<<c_O<<"  X:"<<c_X<<"  accuracy:"<<c_O/(c_O+c_X)<<endl;
	fp3.close();
	
	//system("pause");
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
	for(int i=0;i<size;i++){//方型圖座標i,j
		double deg=(i*dd*pi)/180;//degree 0:360
		double radius=(4*r)/size;
		for(int j=0;j<size/4;j++){//方型圖座標i,j
			h2=h+j*radius*cos(deg);//h
			k2=k+j*radius*sin(deg);//k
			Vec3b color=src_img.at<Vec3b>(Point(h2,k2));
			result.at<Vec3b>(Point(i,j)) = color;
		}
	}
	return result;
}
int count_up(Mat in){
	int avg=0;
	int c=0;
	for(int x=0;x<in.cols;x++){
		for(int y=0;y<in.rows;y++){
			if(y<260){
				Vec3b color=in.at<Vec3b>(Point(x,y));
				avg=avg+color[0];
				//avg=avg+color[0]+color[1];
				c++;
			}
		}
	}

	return avg;
}
int count_dn(Mat in){
	int avg=0;
	int c=0;
	for(int x=0;x<in.cols;x++){
		for(int y=0;y<in.rows;y++){
			if(y>=260&&y<358){
				Vec3b color=in.at<Vec3b>(Point(x,y));
				avg=avg+color[0];
				c++;
			}
		}
	}

	return avg;
}
int count_up_a(Mat in){
	int avg=0;
	int c=0;
	for(int x=0;x<in.cols;x++){
		for(int y=0;y<in.rows;y++){
			if(y<260){
				Vec3b color=in.at<Vec3b>(Point(x,y));
				if(color[0]>50)
					c++;
			}
		}
	}

	return c;
}
int count_dn_a(Mat in){
	int avg=0;
	int c=0;
	for(int x=0;x<in.cols;x++){
		for(int y=0;y<in.rows;y++){
			if(y>=260&&y<358){
				Vec3b color=in.at<Vec3b>(Point(x,y));
				if(color[0]>50)
					c++;
			}
		}
	}

	return c;
}
Mat draw_f(Mat i1,Mat i2){
	//GaussianBlur( i2, i2, Size(9,9), 0, 0 );
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			if(x==0||y==0||x==1439||y==359){
					i1.at<uchar>(Point(x,y))=0;
			}
		}
	}
	return i1;
}

Mat draw_O(Mat i1,Mat i2){
	for(int x=0;x<i1.cols;x++){
		for(int y=0;y<i1.rows;y++){
			if(i2.at<uchar>(Point(x,y))==0){
					i1.at<Vec3b>(Point(x,y))[0]=0;
					i1.at<Vec3b>(Point(x,y))[1]=0;
					i1.at<Vec3b>(Point(x,y))[2]=255;
			}
		}
	}
	return i1;
}