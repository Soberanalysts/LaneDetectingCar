#include <opencv\cv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <opencv\highgui.h>
#include <iostream>
#include "tri_function.h"

#define WIDTH 640
#define HEIGHT 480
#define PI 3.14159265
//////////////ForGaussianFiltering//////////////////////
#define GAUSSIANmask 3
////////////////ForSobelFiltering///////////////////////
#define SOBELmask 3
//////////////////ForSharpening/////////////////////////
#define Laplacianmask 3
/////////////ForDoubleThresholding//////////////////////
#define HIGH 35
#define LOW 20
#define H_EDGE 255
#define L_EDGE 70
////////////////ForEdgetracking/////////////////////////
#define ROW 5
#define COL 5
////////////////ForHoughTransform///////////////////////
#define VOTE 60
#define ANGLE 145
#define rMAX 1000

typedef unsigned char BYTE;

unsigned char** MemAlloc_2D(int width, int height);
void MemFree_2D(unsigned char** arr, int height);
char** MemAlloc_2D_signed(int width, int height);
void MemFree_2D_signed(char** arr, int height);
void FileRead(char* filename, unsigned char** img_in, int width, int height);
void FileWrite(char* filename, unsigned char** img_out, int width, int height);
void GaussianFilter(IplImage* img_in, IplImage* img_out, int width, int height);
void SobelFilter(IplImage* img_in, char** img_gradx, char** img_grady, char** direction, IplImage* img_grad, IplImage* img_out, int width, int height);
void NonMaxSupp(IplImage* img_in, char** img_gradx, char** img_grady, char** direction, IplImage* img_out, int width, int height, int low, int high);
void EdgeTracking(IplImage* img_in, IplImage* img_out, int width, int height);
void Sharpening(IplImage* img_in, IplImage* img_out, int width, int height);
void HoughTransform(IplImage* img_in, IplImage* img_out, int width, int height);

int main()
{
	///////////////////////////////////////////////////////////////////
	///////////////////////시간측정시작////////////////////////////////

	int StartSec, FinishSec;
	StartSec = (int)clock();

	///////////////////////////////////////////////////////////////////
	//////////////////////////이미지읽기///////////////////////////////

	IplImage* img_ori = cvLoadImage("Road6.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* img_out = cvCreateImage(cvSize(img_ori->width, img_ori->height), IPL_DEPTH_8U, 1);
	IplImage* img_gaussian = cvCreateImage(cvSize(img_ori->width, img_ori->height), IPL_DEPTH_8U, 1);

	IplImage* img_grad = cvCreateImage(cvSize(img_ori->width, img_ori->height), IPL_DEPTH_8U, 1);
	IplImage* img_sobel = cvCreateImage(cvSize(img_ori->width, img_ori->height), IPL_DEPTH_8U, 1);
	IplImage* img_sharpening = cvCreateImage(cvSize(img_ori->width, img_ori->height), IPL_DEPTH_8U, 1);
	IplImage* img_houghTF = cvCreateImage(cvSize(img_ori->width, img_ori->height), IPL_DEPTH_8U, 1);

	CvMat *Mat_ori = cvCreateMat(HEIGHT, WIDTH, CV_8UC1);

	///////////////////////////////////////////////////////////////////
	//////////////////////////메모리할당///////////////////////////////

	char **img_gradx, **img_grady, **direction;

	img_gradx = MemAlloc_2D_signed(WIDTH, HEIGHT);
	img_grady = MemAlloc_2D_signed(WIDTH, HEIGHT);
	direction = MemAlloc_2D_signed(WIDTH, HEIGHT);

	///////////////////////////////////////////////////////////////////
	////////////////////////캐니엣지검출///////////////////////////////

	GaussianFilter(img_ori, img_gaussian, WIDTH, HEIGHT);
	SobelFilter(img_gaussian, img_gradx, img_grady, direction, img_grad, img_sobel, WIDTH, HEIGHT);
	Sharpening(img_sobel, img_sharpening, WIDTH, HEIGHT);
	HoughTransform(img_sharpening, img_houghTF, WIDTH, HEIGHT);
	
	///////////////////////////////////////////////////////////////////
	//////////////////////////메모리해제///////////////////////////////

	MemFree_2D_signed(img_gradx, HEIGHT);
	MemFree_2D_signed(img_grady, HEIGHT);
	MemFree_2D_signed(direction, HEIGHT);

	///////////////////////////////////////////////////////////////////
	/////////////////////////시간측정끝////////////////////////////////

	FinishSec = (int)clock();
	printf("측정 시간 : %dsec(%dms)\n", (FinishSec - StartSec) / 1000, FinishSec - StartSec);

	cvShowImage("Original", img_ori);
	cvShowImage("Gaussian", img_gaussian);
	cvShowImage("Grad", img_grad);
	cvShowImage("Sobel", img_sobel);
	cvShowImage("Shapening", img_sharpening);
	cvShowImage("HoughTransform", img_houghTF);

	cvWaitKey(0);

	cvDestroyAllWindows();

	cvReleaseImage(&img_ori);
	cvReleaseImage(&img_gaussian);
	cvReleaseImage(&img_grad);
	cvReleaseImage(&img_sobel);
	cvReleaseImage(&img_sharpening);
	cvReleaseImage(&img_houghTF);

	return 0;
}


void GaussianFilter(IplImage* img_in, IplImage* img_out, int width, int height){

	int padding = GAUSSIANmask / 2;
	int i, j, x, k;

	///////////////////////////////////////////////////////////////////
	///////////////////////////메모리할당//////////////////////////////

	unsigned char** img_in_padding = MemAlloc_2D(width + 2 * padding, height + 2 * padding);
	unsigned char** img_in_padding2 = MemAlloc_2D(width + 2 * padding, height + 2 * padding);

	///////////////////////////////////////////////////////////////////
	//////////////////1차원 가우시안 마스크 생성/////////////////////// 

	double mask_sum, pix_sum, pix_sum2;

	double gaussian_mask[3][3] = {						//5x5 Gaussian mask
		(0.0625, 0.1250, 0.0625),
		(0.1250, 0.2500, 0.1250),
		(0.0625, 0.1250, 0.0625)
	};

	////////////////////////////////////////////////////////////////////
	/////////////////////////ImagePadding//////////////////////////////

	for (i = 0; i<height; i++){
		for (j = 0; j<width; j++){
			img_in_padding[i + padding][j + padding] = img_in->imageData[img_in->width*i + j];
		}
	}
	for (i = padding; i<height + padding; i++){
		for (j = 0; j<padding; j++){
			img_in_padding[i][j] = img_in_padding[i][padding];
			img_in_padding[i][width + padding + j] = img_in_padding[i][width + padding - 1];
		}
	}
	for (j = padding; j<width + padding; j++){
		for (i = 0; i<padding; i++){
			img_in_padding[i][j] = img_in_padding[padding][j];
			img_in_padding[height + padding + i][j] = img_in_padding[height + padding - 1][j];
		}
	}
	for (i = 0; i<padding; i++){
		for (j = 0; j<padding; j++){
			img_in_padding[i][j] = img_in_padding[padding][padding];
			img_in_padding[i][width + padding + j] = img_in_padding[padding][width + padding - 1];
			img_in_padding[height + padding + i][j] = img_in_padding[height + padding - 1][padding];
			img_in_padding[height + padding + i][width + padding + j] = img_in_padding[height + padding - 1][width + padding - 1];
		}
	}


	///////////////////////////////////////////////////////////////////
	////////////////////////y축Convolution/////////////////////////////
	
	
	for (i = 0; i < height; i++){				//2-D Laplacian filtering
		for (j = 0; j < width; j++){
			pix_sum = 0;
			for (int m = 0; m < 3; m++){
				for (int n = 0; n < 3; n++){
					pix_sum += img_in_padding[i + m][j + n] * gaussian_mask[m][n];
					//화소값 범위 넘으면 자르기
				}
				if (pix_sum < 0){ pix_sum = 0; }
				else if (pix_sum > 255){ pix_sum = 255; }
			}
			cvSetReal2D(img_out, i, j, pix_sum);
		}
	}
	////////////////////////////////////////////////////////////////////
	///////////////////////////메모리해제///////////////////////////////
	
	MemFree_2D(img_in_padding, height + 2 * padding);
	MemFree_2D(img_in_padding2, height + 2 * padding);
}

void SobelFilter(IplImage* img_in, char** img_gradx, char** img_grady, char** direction, IplImage* img_grad, IplImage* img_out, int width, int height){

	int i, j, row, col;
	int padding = SOBELmask / 2;
	int pix_data, pix_data2, pix_data_out;
	double theta_rad;
	double theta_radpi;
	double theta_deg;

	int max, min;

	////////////////////////////////////////////////////////////////////
	/////////////////////////SobelMask생성//////////////////////////////

	double mask_ver[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
	double mask_hor[3][3] = { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

	///////////////////////////////////////////////////////////////////
	///////////////////////////메모리할당//////////////////////////////

	unsigned char** img_in_padding = MemAlloc_2D(width + 2 * padding, height + 2 * padding);

	////////////////////////////////////////////////////////////////////
	/////////////////////////ImagePadding//////////////////////////////

	for (i = 0; i<height; i++){
		for (j = 0; j<width; j++){
			img_in_padding[i + padding][j + padding] = img_in->imageData[img_in->width*i + j];
		}
	}
	for (i = padding; i<height + padding; i++){
		for (j = 0; j<padding; j++){
			img_in_padding[i][j] = img_in_padding[i][padding];
			img_in_padding[i][width + padding + j] = img_in_padding[i][width + padding - 1];
		}
	}
	for (j = padding; j<width + padding; j++){
		for (i = 0; i<padding; i++){
			img_in_padding[i][j] = img_in_padding[padding][j];
			img_in_padding[height + padding + i][j] = img_in_padding[height + padding - 1][j];
		}
	}
	for (i = 0; i<padding; i++){
		for (j = 0; j<padding; j++){
			img_in_padding[i][j] = img_in_padding[padding][padding];
			img_in_padding[i][width + padding + j] = img_in_padding[padding][width + padding - 1];
			img_in_padding[height + padding + i][j] = img_in_padding[height + padding - 1][padding];
			img_in_padding[height + padding + i][width + padding + j] = img_in_padding[height + padding - 1][width + padding - 1];
		}
	}

	////////////////////////////////////////////////////////////////////
	/////////////////////////SobelMask적용//////////////////////////////

	max = (int)-10e10; min = (int)10e10;
	for (j = 0; j<height; j++){
		for (i = 0; i<width; i++){
			pix_data = 0; pix_data2 = 0;
			for (row = 0; row<SOBELmask; row++){
				for (col = 0; col<SOBELmask; col++){
					pix_data += mask_ver[row][col] * img_in_padding[j + row][i + col];
					pix_data2 += mask_hor[row][col] * img_in_padding[j + row][i + col];
				}
			}
			pix_data = pix_data / 4;
			pix_data2 = pix_data2 / 4;
			pix_data_out = sqrt((pow(pix_data, 2.0) + pow(pix_data2, 2.0)));

			img_gradx[j][i] = pix_data;
			img_grady[j][i] = pix_data2;
			img_grad->imageData[img_grad->width*j + i] = pix_data_out;

			if (pix_data_out<min)	min = pix_data_out;
			if (pix_data_out>max)	max = pix_data_out;

			if (pix_data == 0) theta_rad = atan((double)pix_data2 / 0.000001);
			else			theta_rad = atan((double)pix_data2 / (double)pix_data);

			theta_radpi = PI / theta_rad;
			theta_deg = theta_rad * 180 / PI;

			direction[j][i] = theta_deg;

			theta_rad = 0; theta_deg = 0;
		}
	}
	////////////////////////////////////////////////////////////////////
	///////////////////////0~255이미지맵핑//////////////////////////////


	for (j = 0; j<height; j++){
		for (i = 0; i<width; i++){
			img_out->imageData[img_out->width*j + i] = img_grad->imageData[img_grad->width*j + i] * 255 / (max - min);
		}
	}
}

void Sharpening(IplImage* img_in, IplImage* img_out, int width, int height){
	int padding = Laplacianmask / 2;
	int i, j, x, k;

	///////////////////////////////////////////////////////////////////
	///////////////////////////메모리할당//////////////////////////////

	unsigned char** img_in_padding = MemAlloc_2D(width + 2 * padding, height + 2 * padding);
	unsigned char** img_in_padding2 = MemAlloc_2D(width + 2 * padding, height + 2 * padding);

	///////////////////////////////////////////////////////////////////
	//////////////////1차원 가우시안 마스크 생성/////////////////////// 

	double mask_sum, pix_sum, pix_sum2;

	double laplacian_mask[3][3] = { { 0, 1, 0 }, { 1, -4, 1 }, { 0, 1, 0 } };

	////////////////////////////////////////////////////////////////////
	/////////////////////////ImagePadding//////////////////////////////

	for (i = 0; i<height; i++){
		for (j = 0; j<width; j++){
			img_in_padding[i + padding][j + padding] = img_in->imageData[img_in->width*i + j];
		}
	}
	for (i = padding; i<height + padding; i++){
		for (j = 0; j<padding; j++){
			img_in_padding[i][j] = img_in_padding[i][padding];
			img_in_padding[i][width + padding + j] = img_in_padding[i][width + padding - 1];
		}
	}
	for (j = padding; j<width + padding; j++){
		for (i = 0; i<padding; i++){
			img_in_padding[i][j] = img_in_padding[padding][j];
			img_in_padding[height + padding + i][j] = img_in_padding[height + padding - 1][j];
		}
	}
	for (i = 0; i<padding; i++){
		for (j = 0; j<padding; j++){
			img_in_padding[i][j] = img_in_padding[padding][padding];
			img_in_padding[i][width + padding + j] = img_in_padding[padding][width + padding - 1];
			img_in_padding[height + padding + i][j] = img_in_padding[height + padding - 1][padding];
			img_in_padding[height + padding + i][width + padding + j] = img_in_padding[height + padding - 1][width + padding - 1];
		}
	}


	///////////////////////////////////////////////////////////////////
	////////////////////////y축Convolution/////////////////////////////
	for (i = 0; i < height; i++){				//2-D Laplacian filtering
		for (j = 0; j < width; j++){
			pix_sum = 0;
			for (int m = 0; m < 3; m++){
				for (int n = 0; n < 3; n++){
					pix_sum += img_in_padding[i + m][j + n] * laplacian_mask[m][n];
					//화소값 범위 넘으면 자르기
				}
				if (pix_sum < 0){ pix_sum = 0; }
				else if (pix_sum > 255){ pix_sum = 255; }
			}
			cvSetReal2D(img_out, i, j, pix_sum);
		}
	}

	////////////////////////////////////////////////////////////////////
	///////////////////////////메모리해제///////////////////////////////

	MemFree_2D(img_in_padding, height + 2 * padding);
	MemFree_2D(img_in_padding2, height + 2 * padding);
}

void HoughTransform(IplImage* img_in, IplImage* img_out, int width, int height){

	int voting[ANGLE][rMAX] = { 0 };

	int i, j, angle, r, x, y;

	char h_edge = H_EDGE;

	//double rad = PI / 180.0;

	double LUT_COS[ANGLE];
	double LUT_SIN[ANGLE];

	memset(LUT_COS, 0, sizeof(LUT_COS));
	memset(LUT_SIN, 0, sizeof(LUT_SIN));
	printf("on\n");

	for (angle = 35; angle<70; angle+=5){
		printf("Voting start");
		for (r = 0; r < rMAX; r++)	{
			voting[angle][r] = 0;
		}
	}
	for (angle = 110; angle<ANGLE; angle+=5){
		printf("Voting start");
		for (r = 0; r < rMAX; r++)	{
			voting[angle][r] = 0;
		}
	}
	////////////////////////////////////////////////////////////////////
	///////////////LookUpTable에Cos,Sin값저장(0~360)////////////////////

	for (i = 35; i<70; i+=5){
		LUT_COS[i] = (double)cos(i);
		LUT_SIN[i] = (double)sin(i);
	}
	for (i = 110; i<ANGLE; i+=5){
		LUT_COS[i] = (double)cos(i);
		LUT_SIN[i] = (double)sin(i);
	}
	////////////////////////////////////////////////////////////////////
	////////////////////////////이미지복사//////////////////////////////
	for (j = 0; j<height; j++){
		printf("Check\n");
		for (i = 0; i<width; i++){
			img_out->imageData[img_out->width*j + i] = img_in->imageData[img_in->width*j + i];
		}
	}
	printf("image copy\n");
	////////////////////////////////////////////////////////////////////
	/////////////////////////////Voting/////////////////////////////////

	for (j = 0; j<height; j++){
		for (i = 0; i<width; i++){
			if (img_out->imageData[img_out->width*j + i] > 80){
				for (angle = 35; angle<70; angle+=5){
					r = (int)(i*LUT_COS[angle] + j*LUT_SIN[angle]);
					if (r >= 0 && r <= rMAX) {
						voting[angle][r]++;
					}
				}
				for (angle = 110; angle<ANGLE; angle+=5){
					r = (int)(i*LUT_COS[angle] + j*LUT_SIN[angle]);
					if (r >= 0 && r <= rMAX) {
						voting[angle][r]++;
					}
				}
			}
		}
	}

	/////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////

	for (r = 0; r <= rMAX; r++){
		for (angle = 35; angle<70; angle+=5){
			if (voting[angle][r]>VOTE){
				for (j = 0; j<height; j++){
					x = (int)((r - j*LUT_SIN[angle]) / LUT_COS[angle]);
					if (x<width && x>0) img_out->imageData[img_out->width*j + x] = 255;
				}
				for (i = 0; i<width; i++){
					y = (int)((r - i*LUT_COS[angle]) / LUT_SIN[angle]);
					if (y<height && y>0) img_out->imageData[img_out->width*y + i] = 255;
				}
			}
		}
		for (angle = 110; angle<ANGLE; angle+=5){
			if (voting[angle][r]>VOTE){
				for (j = 0; j<height; j++){
					x = (int)((r - j*LUT_SIN[angle]) / LUT_COS[angle]);
					if (x<width && x>0) img_out->imageData[img_out->width*j + x] = 255;
				}
				for (i = 0; i<width; i++){
					y = (int)((r - i*LUT_COS[angle]) / LUT_SIN[angle]);
					if (y<height && y>0) img_out->imageData[img_out->width*y + i] = 255;
				}
			}
		}
	}

}

unsigned char** MemAlloc_2D(int width, int height){
	unsigned char** arr;
	int i;

	arr = (unsigned char**)malloc(sizeof(unsigned char*)*height);
	for (i = 0; i<height; i++)
		arr[i] = (unsigned char*)malloc(sizeof(unsigned char*)*width);

	return arr;
}

void MemFree_2D(unsigned char** arr, int height){
	int i;
	for (i = 0; i<height; i++){
		free(arr[i]);
	}
	free(arr);
}

void FileRead(char* filename, unsigned char** img_in, int width, int height){
	FILE* fp_in;
	int i;

	fp_in = fopen(filename, "rb");
	for (i = 0; i<height; i++)	fread(img_in[i], sizeof(unsigned char), width, fp_in);
	fclose(fp_in);
}

void FileWrite(char* filename, unsigned char** img_out, int width, int height){
	FILE* fp_out;
	int i;

	fp_out = fopen(filename, "wb");
	for (i = 0; i<height; i++)	fwrite(img_out[i], sizeof(unsigned char), width, fp_out);
	fclose(fp_out);
}

char** MemAlloc_2D_signed(int width, int height){
	char** arr;
	int i;

	arr = (char**)malloc(sizeof(char*)*height);
	for (i = 0; i<height; i++)
		arr[i] = (char*)malloc(sizeof(char*)*width);

	return arr;
}

void MemFree_2D_signed(char** arr, int height){
	int i;
	for (i = 0; i<height; i++){
		free(arr[i]);
	}
	free(arr);
}
