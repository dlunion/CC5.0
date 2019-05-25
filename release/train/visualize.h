#pragma once

#include <cc_v5.h>
#include <vector>

using namespace std;
using namespace cc;

vector<int> range(int begin, int end, int step = 1);
vector<int> range(int num);

const int norNorm = 0;
const int localNorm = 1;
const int showFeatureMap = 1;
const int showImage = 2;
const int showMatrix = 3;
void initializeVisualze();
void destoryVisualze();
void visualizeFeatureMap_image(Blob* blob, const string& name, int n, int rows, int cols, int normType = localNorm);
void visualFeaturemap(const Mat& featuremap, Mat& showbgr);
void visualizeFeatureMap_featuremap(Blob* blob, const string& name, int n, const vector<int>& channelIndex, int rows, int cols, int normType = localNorm);
void visualizeFeatureMap(Blob* blob, const string& name, int n, const vector<int>& channelIndex, int rows, int cols, int normType = localNorm, int showType = showFeatureMap);
void postBlob(Blob* blob, const string& name, int n = 0, int rows = 0, int cols = 0, int normType = localNorm, int showType = showFeatureMap);
void postBlob(Blob* blob, const string& name, const vector<int>& channelIndex, int n = 0, int rows = 0, int cols = 0, int normType = localNorm, int showType = showFeatureMap);
void postMatrix(Mat matrix, const string& name);
void setSolver(cc::Solver* solver);
cc::Solver* getSolver();