
#include <cv.h>
#include <highgui.h>
#include <iostream>

using namespace std;

int main ()
{

	IplImage *img=cvLoadImage("C:/FPGA_pic/Road8.jpg",CV_LOAD_IMAGE_GRAYSCALE);	// �̹���(����) �ҷ�����
	IplImage *img_bin = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U,1);
	
	cvThreshold(img, img_bin, 100, 255, CV_THRESH_TRUNC );

	cvShowImage ("IMG", img);																// �̹��� ���
	cvShowImage ("IMG_COPY", img_bin);																// �̹��� ���
	cvWaitKey (0);																				
	cvDestroyAllWindows ();																			
	cvReleaseImage (&img);																	// �Ҵ�� �̹����� �޸� ����
	cvReleaseImage (&img_bin);		

	return 0;

}
