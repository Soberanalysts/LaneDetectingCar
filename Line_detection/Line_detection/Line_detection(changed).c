
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "tri_function.h"

#define WIDTH 256
#define HEIGHT 256
#define PI 3.141592654
//////////////ForGaussianFiltering//////////////////////
#define GAUSSIANmask 3
////////////////ForSobelFiltering///////////////////////
#define SOBELmask 3
////////////////ForSharpening///////////////////////////
#define Laplacian 3
////////////////ForHoughTransform///////////////////////
#define VOTE 60
#define ANGLE 145

#define rMAX WIDTH+HEIGHT

////////////////Sin, Cos///////////////////////////////
typedef unsigned char BYTE;

unsigned char** MemAlloc_2D(int width, int height);
void MemFree_2D(unsigned char** arr, int height);
char** MemAlloc_2D_signed(int width, int height);
void MemFree_2D_signed(char** arr, int height);
void FileRead(char* filename, unsigned char** img_in, int width, int height);
void FileWrite(char* filename, unsigned char** img_out, int width, int height);
void GaussianFilter(unsigned char** img_in, unsigned char** img_out, unsigned char** img_out_fin, int sigma, int width, int height);
void SobelFilter(unsigned char** img_in, char** img_gradx, char** img_grady, unsigned char** img_out, int width, int height);
void Sharpening(unsigned char** img_in, unsigned** img_out, int width, int height);

void HoughTransform(unsigned char** img_in, unsigned char** img_out, int width, int height);

void main()
{
	BYTE **img_ori, **img_out, **img_gaussian, **img_grad, **img_sobel, **img_sharpening,**img_houghTF, **img_sample;
	char **img_gradx, **img_grady;
	char filename_out[100];

	///////////////////////////////////////////////////////////////////
	///////////////////////시간측정시작////////////////////////////////

	int StartSec, FinishSec;
	StartSec=(int)clock();

	///////////////////////////////////////////////////////////////////
	//////////////////////////메모리할당///////////////////////////////

	img_ori=MemAlloc_2D(WIDTH,HEIGHT);
	img_out=MemAlloc_2D(WIDTH,HEIGHT);
	img_gaussian=MemAlloc_2D(WIDTH,HEIGHT);

	img_gradx=MemAlloc_2D_signed(WIDTH,HEIGHT);
	img_grady=MemAlloc_2D_signed(WIDTH,HEIGHT);
	img_grad=MemAlloc_2D(WIDTH,HEIGHT);
	img_sobel=MemAlloc_2D(WIDTH,HEIGHT);
	img_sharpening = MemAlloc_2D(WIDTH, HEIGHT);

	img_sample=MemAlloc_2D(WIDTH,HEIGHT);

	img_houghTF=MemAlloc_2D(WIDTH,HEIGHT);

	///////////////////////////////////////////////////////////////////
	//////////////////////////이미지읽기///////////////////////////////

	FileRead("HOUSE256.raw",img_ori,WIDTH,HEIGHT);
	FileRead("RoadCannyEdge.raw",img_sample,WIDTH,HEIGHT);

	///////////////////////////////////////////////////////////////////
	////////////////////////캐니엣지검출///////////////////////////////

	GaussianFilter(img_ori, img_out, img_gaussian, 1, WIDTH, HEIGHT);
	FinishSec=(int)clock();
	printf("GaussianFiltering 측정 시간 : %dsec(%dms)\n", (FinishSec-StartSec)/1000, FinishSec-StartSec);

	SobelFilter(img_gaussian, img_gradx, img_grady, img_grad, img_sobel, WIDTH, HEIGHT);
	FinishSec=(int)clock();
	printf("SobelFiltering 측정 시간 : %dsec(%dms)\n", (FinishSec-StartSec)/1000, FinishSec-StartSec);

	Sharpening(img_sobel, img_sharpening, WIDTH, HEIGHT);
	FinishSec = (int)clock();
	printf("Sharpening 측정 시간 : %dsec(%dms)\n", (FinishSec - StartSec) / 1000, FinishSec - StartSec);

	HoughTransform(img_sharpening, img_houghTF, WIDTH, HEIGHT);
	FinishSec=(int)clock();
	printf("HoughTF 측정 시간 : %dsec(%dms)\n", (FinishSec-StartSec)/1000, FinishSec-StartSec);

	///////////////////////////////////////////////////////////////////
	//////////////////////////이미지저장///////////////////////////////

	sprintf(filename_out, "LENA512_gaussian_ver.raw");
	FileWrite(filename_out, img_out, WIDTH, HEIGHT);
	sprintf(filename_out, "EXgaussian_mix.raw");
	FileWrite(filename_out, img_gaussian, WIDTH, HEIGHT);
	sprintf(filename_out, "EXgradx.raw");
	FileWrite(filename_out, img_gradx, WIDTH, HEIGHT);
	sprintf(filename_out, "EXgrady.raw");
	FileWrite(filename_out, img_grady, WIDTH, HEIGHT);
	sprintf(filename_out, "EXgrad.raw");
	FileWrite(filename_out, img_grad, WIDTH, HEIGHT);
	sprintf(filename_out, "EXsobel.raw");
	FileWrite(filename_out, img_sobel, WIDTH, HEIGHT);
	sprintf(filename_out, "EXsharpening.raw");
	FileWrite(filename_out, img_sharpening, WIDTH, HEIGHT);
	sprintf(filename_out, "EXhoughTF.raw");
	FileWrite(filename_out, img_houghTF, WIDTH, HEIGHT);

	//////////////////////////////////////////////////////////////////
	//////////////////////////메모리해제///////////////////////////////

	MemFree_2D(img_ori, HEIGHT);
	MemFree_2D(img_out, HEIGHT);
	MemFree_2D(img_gaussian, HEIGHT);

	MemFree_2D_signed(img_gradx, HEIGHT);
	MemFree_2D_signed(img_grady, HEIGHT);
	MemFree_2D(img_grad, HEIGHT);
	MemFree_2D(img_sobel, HEIGHT);
	MemFree_2D(img_sharpening, HEIGHT);

	MemFree_2D(img_sample, HEIGHT);
	MemFree_2D(img_houghTF, HEIGHT);

	///////////////////////////////////////////////////////////////////
	/////////////////////////시간측정끝////////////////////////////////

	FinishSec=(int)clock();
	printf("Total 측정 시간 : %dsec(%dms)\n", (FinishSec-StartSec)/1000, FinishSec-StartSec);
}

void GaussianFilter(unsigned char** img_in, unsigned char** img_out, unsigned char** img_out_fin, int sigma, int width, int height)
{
	int padding = GAUSSIANmask/2;
	int i, j, m, n;

	///////////////////////////////////////////////////////////////////
	///////////////////////////메모리할당//////////////////////////////
	
	unsigned char** img_in_padding = MemAlloc_2D(width+2*padding, height+2*padding);

	///////////////////////////////////////////////////////////////////
	//////////////////1차원 가우시안 마스크 생성/////////////////////// 

	double temp;

	double gaussian_mask[3][3] = {
		(0.0625, 0.1250, 0.0625),
		(0.1250, 0.2500, 0.1250),
		(0.0625, 0.1250, 0.0625)
	};

	
	////////////////////////////////////////////////////////////////////
	/////////////////////////ImagePadding//////////////////////////////
	
	for(i=0;i<height;i++){
		for(j=0;j<width;j++){
			img_in_padding[i+padding][j+padding] = img_in[i][j];
		}
	}
	for(i=padding;i<height+padding;i++){
		for(j=0;j<padding;j++){
			img_in_padding[i][j] = img_in_padding[i][padding];
			img_in_padding[i][width+padding+j] = img_in_padding[i][width+padding-1];
		}
	}
	for(j=padding;j<width+padding;j++){
		for(i=0;i<padding;i++){
			img_in_padding[i][j] = img_in_padding[padding][j];
			img_in_padding[height+padding+i][j] = img_in_padding[height+padding-1][j];
		}
	}
	for(i=0;i<padding;i++){
		for(j=0;j<padding;j++){
			img_in_padding[i][j] = img_in_padding[padding][padding];
			img_in_padding[i][width+padding+j] = img_in_padding[padding][width+padding-1];
			img_in_padding[height+padding+i][j] = img_in_padding[height+padding-1][padding];
			img_in_padding[height+padding+i][width+padding+j] = img_in_padding[height+padding-1][width+padding-1];
		}
	}
	for (i = 0; i < height; i++){				//2-D Gaussian filtering
		for (j = 0; j < width; j++){
			temp = 0;
			for (m = 0; m < 3; m++){
				for (n = 0; n < 3; n++){
					temp += img_in_padding[i + m][j + n] * gaussian_mask[m][n];
				}
			}
			img_out_fin[i][j] = (unsigned char)floor(temp + 0.5);
		}
	}
		
	////////////////////////////////////////////////////////////////////
	///////////////////////////메모리해제///////////////////////////////
	MemFree_2D(img_in_padding,height+2*padding);
}

void SobelFilter(unsigned char** img_in, char** img_gradx, char** img_grady, unsigned char** img_grad, unsigned char** img_out, int width, int height){

	FILE *fp_info;

	int i,j,row,col;
	int padding = SOBELmask/2;
	int pix_data, pix_data2, pix_data_out;
	double theta_rad;
	double theta_radpi;
	double theta_deg;

	int max, min;

	////////////////////////////////////////////////////////////////////
	/////////////////////////SobelMask생성//////////////////////////////

	double mask_ver[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
	double mask_hor[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};

	///////////////////////////////////////////////////////////////////
	///////////////////////////메모리할당//////////////////////////////

	unsigned char** img_in_padding = MemAlloc_2D(width+2*padding, height+2*padding);

	////////////////////////////////////////////////////////////////////
	/////////////////////////ImagePadding///////////////////////////////

	for(i=0;i<height;i++){
		for(j=0;j<width;j++){
			img_in_padding[i+padding][j+padding] = img_in[i][j];
		}
	}
	for(i=padding;i<height+padding;i++){
		for(j=0;j<padding;j++){
			img_in_padding[i][j] = img_in_padding[i][padding];
			img_in_padding[i][width+padding+j] = img_in_padding[i][width+padding-1];
		}
	}
	for(j=padding;j<width+padding;j++){
		for(i=0;i<padding;i++){
			img_in_padding[i][j] = img_in_padding[padding][j];
			img_in_padding[height+padding+i][j] = img_in_padding[height+padding-1][j];
		}
	}
	for(i=0;i<padding;i++){
		for(j=0;j<padding;j++){
			img_in_padding[i][j] = img_in_padding[padding][padding];
			img_in_padding[i][width+padding+j] = img_in_padding[padding][width+padding-1];
			img_in_padding[height+padding+i][j] = img_in_padding[height+padding-1][padding];
			img_in_padding[height+padding+i][width+padding+j] = img_in_padding[height+padding-1][width+padding-1];
		}
	}

	
	////////////////////////////////////////////////////////////////////
	/////////////////////////SobelMask적용//////////////////////////////

	max=(int)-10e10; min=(int)10e10;
	for(j=0;j<height;j++){
		for(i=0;i<width;i++){
			pix_data=0; pix_data2=0;
			for(row=0;row<SOBELmask;row++){
				for(col=0;col<SOBELmask;col++){
					pix_data+=mask_ver[row][col]*img_in_padding[j+row][i+col];
					pix_data2+=mask_hor[row][col]*img_in_padding[j+row][i+col];
				}
			}
			pix_data=pix_data/4;
			pix_data2=pix_data2/4;
			pix_data_out=sqrt((pow(pix_data,2.0)+pow(pix_data2,2.0)));
						
			img_grady[j][i]=pix_data;
			img_gradx[j][i]=pix_data2;
			img_grad[j][i]=pix_data_out;
			
			if(pix_data_out<min)	min=pix_data_out;
			if(pix_data_out>max)	max=pix_data_out;

			if(pix_data==0) theta_rad=atan((double)pix_data2/0.000001);
			else			theta_rad=atan((double)pix_data2/(double)pix_data);

			theta_radpi=PI/theta_rad;
			theta_deg=theta_rad*180/PI;
			theta_rad=0; theta_deg=0;
		}
	}
	////////////////////////////////////////////////////////////////////
	///////////////////////0~255이미지맵핑//////////////////////////////
	
	
	for(j=0;j<height;j++){
		for(i=0;i<width;i++) img_out[j][i]=img_grad[j][i]*255/(max-min);
	}
}

void Sharpening(unsigned char** img_in, unsigned char** img_out, int width, int height){
	int padding = Laplacian / 2;
	int i, j, m, n;

	///////////////////////////////////////////////////////////////////
	///////////////////////////메모리할당//////////////////////////////

	unsigned char** img_in_padding = MemAlloc_2D(width + 2 * padding, height + 2 * padding);

	///////////////////////////////////////////////////////////////////
	//////////////////1차원 가우시안 마스크 생성/////////////////////// 
	double temp;
	double laplacian_mask[3][3] = { { 0, 1, 0 }, 
									{ 1, -4, 1 }, 
									{ 0, 1, 0}};

	////////////////////////////////////////////////////////////////////
	/////////////////////////ImagePadding//////////////////////////////

	for (i = 0; i<height; i++){
		for (j = 0; j<width; j++){
			img_in_padding[i + padding][j + padding] = img_in[i][j];
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
	for (i = 0; i < height; i++){				//2-D Laplacian filtering
		for (j = 0; j < width; j++){
			temp = 0;
			for (m = 0; m < 3; m++){
				for (n = 0; n < 3; n++){
					temp += img_in_padding[i + m][j + n] * laplacian_mask[m][n];
					//화소값 범위 넘으면 자르기
				}
				if (temp < 0){ temp = 0; }
				else if (temp > 255){ temp = 255; }
			}
			img_out[i][j] = (unsigned char)temp;
		}
	}

	////////////////////////////////////////////////////////////////////
	///////////////////////////메모리해제///////////////////////////////
	MemFree_2D(img_in_padding, height + 2 * padding);
}

void HoughTransform(unsigned char** img_in, unsigned char** img_out, int width, int height){
	
	int voting[ANGLE][rMAX]={0};

	int i,j,angle,r,x,y;
	
	double rad = PI/180.0;
	
	double LUT_COS[ANGLE];
	double LUT_SIN[ANGLE];
	
	memset(LUT_COS,0,sizeof(LUT_COS));
	memset(LUT_SIN,0,sizeof(LUT_SIN));
	
	for (angle = 35; angle < 70; angle+=5){
		for (r = 0; r < rMAX; r++) voting[angle][r] = 0;
	}
	for (angle = 110; angle < ANGLE; angle+=5){
		for (r = 0; r < rMAX; r++) voting[angle][r] = 0;
	}
	////////////////////////////////////////////////////////////////////
	///////////////LookUpTable에Cos,Sin값저장(0~360)////////////////////
	
	for (i = 35; i<70; i+=5){
		LUT_COS[i] = (double)cos(i, 0);
		LUT_SIN[i] = (double)sin(i, 0);
	}
	for (i = 110; i<ANGLE; i+=5){
		LUT_COS[i] = (double)cos(i, 0);
		LUT_SIN[i] = (double)sin(i, 0);
	}
	
	////////////////////////////////////////////////////////////////////
	////////////////////////////이미지복사//////////////////////////////
	for(j=0;j<height;j++){
		for(i=0;i<width;i++){
			img_out[j][i]=img_in[j][i];
		}
	}
	
	////////////////////////////////////////////////////////////////////
	/////////////////////////////Voting/////////////////////////////////
	
	for(j=0;j<height;j++){
		for(i=0;i<width;i++){
			if(img_out[j][i]>80){
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
	
	for(r=0;r<=rMAX;r++){
		for (angle = 35; angle<70; angle+=5){
			if (voting[angle][r]>VOTE){

				for (j = 0; j<height; j++){
					x = (int)((r - j*LUT_SIN[angle]) / LUT_COS[angle]);
					if (x<width && x>0) img_out[j][x] = 255;
				}
				for (i = 0; i<width; i++){
					y = (int)((r - i*LUT_COS[angle]) / LUT_SIN[angle]);
					if (y<height && y>0) img_out[y][i] = 255;
				}
			}
		}
		for (angle = 110; angle<ANGLE; angle+=5){
			if (voting[angle][r]>VOTE){

				for (j = 0; j<height; j++){
					x = (int)((r - j*LUT_SIN[angle]) / LUT_COS[angle]);
					if (x<width && x>0) img_out[j][x] = 255;
				}
				for (i = 0; i<width; i++){
					y = (int)((r - i*LUT_COS[angle]) / LUT_SIN[angle]);
					if (y<height && y>0) img_out[y][i] = 255;
				}
			}
		}
	}
	
}


unsigned char** MemAlloc_2D(int width, int height){
	unsigned char** arr;
	int i;

	arr=(unsigned char**)malloc(sizeof(unsigned char*)*height);
	for(i=0;i<height;i++)
		arr[i]=(unsigned char*)malloc(sizeof(unsigned char*)*width);

	return arr;
}

char** MemAlloc_2D_signed(int width, int height){
	char** arr;
	int i;

	arr=(char**)malloc(sizeof(char*)*height);
	for(i=0;i<height;i++)
		arr[i]=(char*)malloc(sizeof(char*)*width);

	return arr;
}

void MemFree_2D(unsigned char** arr, int height){
	int i;
	for(i=0;i<height;i++){
		free(arr[i]);
	}
	free(arr);
}

void MemFree_2D_signed(char** arr, int height){
	int i;
	for(i=0;i<height;i++){
		free(arr[i]);
	}
	free(arr);
}

void FileRead(char* filename, unsigned char** img_in, int width, int height){
	FILE* fp_in;
	int i;

	fp_in=fopen(filename, "rb");
	for(i=0;i<height;i++)	fread(img_in[i], sizeof(unsigned char), width, fp_in);
	fclose(fp_in);
}

void FileWrite(char* filename, unsigned char** img_out, int width, int height){
	FILE* fp_out;
	int i;

	fp_out=fopen(filename, "wb");
	for(i=0;i<height;i++)	fwrite(img_out[i], sizeof(unsigned char), width, fp_out);
	fclose(fp_out);
}


