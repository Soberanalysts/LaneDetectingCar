
#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace std;

int main ()
{

	IplImage *img=cvLoadImage("C:/FPGA_pic/Road8.jpg",CV_LOAD_IMAGE_GRAYSCALE);	// 이미지(왼쪽) 불러오기
	IplImage *img_bin = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U,1);
	
	cvThreshold(img, img_bin, 100, 255, CV_THRESH_TRUNC );

	cvShowImage ("IMG", img);																// 이미지 출력
	cvShowImage ("IMG_COPY", img_bin);																// 이미지 출력
	cvWaitKey (0);																				
	cvDestroyAllWindows ();																			
	cvReleaseImage (&img);																	// 할당된 이미지의 메모리 해제
	cvReleaseImage (&img_bin);		

	return 0;

}
