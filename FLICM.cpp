// This is a C++ implementation of FLICM algorithm, which is introduced in paper "A Robust Fuzzy Local Information C-Means Clustering Algorithm"
// Author: Qingsheng Wang
// Date: Feb. 19 2021
// Place: Jinan, China
// OpenCV v3+ used for image reading and displaying is required in this code.

#include <opencv2\opencv.hpp>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
	Mat input = imread("test1_noise.tif", 0);
	const int rows = input.rows;
	const int cols = input.cols;
	namedWindow("Input");
	imshow("Input", input);
	// Parameters
	int m = 2, K = 2, max_iter = 100, side = 3;
	double error = 0.001;
 	// Create memory space for array input_ptr to store input image.
	const int half_k = side / 2;
	const int padded_rows = rows + half_k * 2;
	const int padded_cols = cols + half_k * 2;
	double** input_ptr = (double**)malloc(rows * sizeof(double*));
	if (input_ptr == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
  	for (int i = 0; i < rows; i++)
	{
		input_ptr[i] = (double*)malloc(cols * sizeof(double));
		if (input_ptr[i] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	}
  	// Create memory space for padded input image - input_padded
	double** input_padded = (double**)calloc(padded_rows, sizeof(double*));
	if (input_padded == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	for (int i = 0; i < rows + 2 * half_k; i++)
	{
		input_padded[i] = (double*)calloc(padded_cols, sizeof(double));
		if (input_padded[i] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	}
  	// Fill arrays input_ptr and input_padded with pixels in input
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			input_ptr[i][j] = (double)input.at<uchar>(i, j);
			input_padded[i + half_k][j + half_k] = input_ptr[i][j];
		}
	input.release(); // Delete input
  	// Create memory space for membership values U, fuzzy factor G, and padded U U_padded.
	double*** U = (double***)malloc(rows * sizeof(double**));
	if (U == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	double*** G = (double***)malloc(rows * sizeof(double**));
	if (G == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	double*** U_padded = (double***)calloc(padded_rows, sizeof(double**));
	if (U_padded == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	for (int i = 0; i < rows + 2 * half_k; i++)
	{
		U_padded[i] = (double**)calloc(padded_cols, sizeof(double*));
		if (U_padded[i] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
		for (int j = 0; j < cols + 2 * half_k; j++)
		{
			U_padded[i][j] = (double*)calloc(K, sizeof(double));
			if (U_padded[i][j] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
		}
	}
	srand((int)time(0)); // Set random seed
	for (int i = 0; i < rows; i++)
	{
		U[i] = (double**)malloc(cols * sizeof(double*));
		if (U[i] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
		G[i] = (double**)malloc(cols * sizeof(double*));
		if (G[i] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
		for (int j = 0; j < cols; j++)
		{
			U[i][j] = (double*)malloc(K * sizeof(double));
			if (U[i][j] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
			double U_sum = 0;
			for (int l = 0; l < K; l++)
			{
				U[i][j][l] = (double)rand() / RAND_MAX; // Random initialize for memberships U
				U_sum += U[i][j][l];
			}
			for (int l = 0; l < K; l++)
			{
				U[i][j][l] /= U_sum; // Make sure that sum of all memberships of each pixel is 1
				U_padded[i + half_k][j + half_k][l] = U[i][j][l];
			}
			G[i][j] = (double*)malloc(K * sizeof(double));
			if (G[i][j] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
		}
	}
  	// Create memory space for cluster center - center, and objective function - J
	double* center = (double*)malloc(K * sizeof(double));
	if (center == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	double* J = (double*)calloc(max_iter, sizeof(double));
	if (J == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
  	// CLUSTERING!
	for (int iter = 0; iter < max_iter; iter++)
	{
		for (int l = 0; l < K; l++)
	  	{
      			// Compute cluster centers
			double center_up, center_down;
			center_up = center_down = 0;
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					center_up += input_ptr[i][j] * pow(U[i][j][l], m);
					center_down += pow(U[i][j][l], m);
				}
			center[l] = center_up / center_down;
      			// Compute fuzzy factors
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					double d, U_res, G_temp = 0;
					for (int r = -half_k; r <= half_k; r++)
						for (int n = -half_k; n <= half_k; n++)
						{
							if (r == 0 && n == 0)
								continue;
							d = 1.0 / (1.0 + sqrt(r * r + n * n));
							U_res = pow(1 - U_padded[i + half_k + r][j + half_k + n][l], m);
							G_temp = G_temp + d * U_res * pow(input_padded[i + half_k + r][j + half_k + n] - center[l], 2);
						}
					G[i][j][l] = G_temp;
				}
		}
    		// Compute memberships
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
			{
				double U_down = 0.0, d, U_up;
				for (int l = 0; l < K; l++)
				{
					d = pow(input_ptr[i][j] - center[l], 2);
					U_down = U_down + pow(d + G[i][j][l], -1 / (m - 1));
				}
				for (int l = 0; l < K; l++)
				{
					d = pow(input_ptr[i][j] - center[l], 2);
					U_up = pow(d + G[i][j][l], 1 / (m - 1));
					U[i][j][l] = 1 / (U_up * U_down);
					U_padded[i + half_k][j + half_k][l] = U[i][j][l];
          				// Compute objective function
					J[iter] += pow(U[i][j][l], m) * pow(input_ptr[i][j] - center[l], 2) + G[i][j][l];
				}
			}
		printf("Iter %d，J = %.3f\n", iter + 1, J[iter]);
		if (abs(J[iter] - J[iter - 1]) <= error && iter >= 1)
		{
			cout << "Objective function converged\n";
			break;
		}
		else if (iter == max_iter - 1 && iter >= 1)
		{
			cout << "Max iteration reached\n";
			break;
		}
	}
  	// Memory clearing
	free(G);
	free(J);
	free(input_ptr);
	free(input_padded);
	free(U_padded);
	J = NULL;
	input_ptr = input_padded = NULL;
	U_padded = G = NULL;
  	// Display all cluster centers
	cout << "Cluster centers:\n";
	for (int l = 0; l < K; l++)
		cout << center[l] << "\n";
  	// Compute Vpc and Vpe
	double Vpc, Vpe;
	Vpc = Vpe = 0;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int l = 0; l < K; l++)
			{
				Vpc += U[i][j][l] * U[i][j][l];
				Vpe += U[i][j][l] * log(U[i][j][l]);
			}
	Vpc = Vpc / ((double)rows * (double)cols) * 100;
	Vpe = -Vpe / ((double)rows * (double)cols) * 100;
	cout << "Membership partition index Vpc = " << Vpc << "%" << endl;
	cout << "Membership partition entropy Vpe = " << Vpe << "%" << endl;
  	// Create memory space for pixel labels
	short** labels = (short**)calloc(rows, sizeof(short*));
	if (labels == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	for (int i = 0; i < rows; i++)
	{
		labels[i] = (short*)calloc(cols, sizeof(short));
		if (labels[i] == NULL) { cout << "Critical error in memory allocation!\n"; return 0; }
	}
  	// Set each cluster center to be the color of corresponded cluster
	uchar* colors = new uchar[K];
	for (int i = 0; i < K; i++)
		*(colors + i) = (uchar)(center[i]);
	free(center);
	center = NULL;
  	// Display segmentation result
	Mat result = Mat(rows, cols, CV_8U);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			double U_max = 0;
			for (int l = 0; l < K; l++)
				if (U_max < U[i][j][l])
				{
					U_max = U[i][j][l];
					labels[i][j] = (short)l;
				}
			result.at<uchar>(i, j) = *(colors + labels[i][j]);
		}
  	// Memory clearing
	delete[] colors;
	free(labels);
	free(U);
	labels = NULL;
	U = NULL;
  	// Display!
	imshow("Clustering result", result);
	waitKey(0);
	result.release();
	return 0;
}
