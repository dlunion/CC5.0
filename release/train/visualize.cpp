

#include "visualize.h"
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <memory>

using namespace cc;
using namespace cv;
using namespace std;

static struct ShowInfo{
	shared_ptr<Blob> blob;
	shared_ptr<Blob> show;
	Mat matrix;
	string name;
	int n;
	int rows;
	int cols;
	int normType;
	int showType;
	vector<int> channelsIndex;
};

static mutex lock_;
static map<string, ShowInfo> showWatcher;
static volatile bool singal_show_ = false;
static thread g_work_thread;
static volatile bool keeprun = false;
static Solver* g_solver = nullptr;

cc::Solver* getSolver(){
	return g_solver;
}

void setSolver(cc::Solver* solver){
	g_solver = solver;
}

vector<int> range(int begin, int end, int step ){
	vector<int> lis;
	for (int i = begin; i < end; i += step){
		lis.push_back(i);
	}
	return lis;
}

vector<int> range(int num){
	return range(0, num);
}

void visualizeFeatureMap_image(Blob* blob, const string& name, int n, int rows, int cols, int normType){
	int channels = blob->channel();
	int num = blob->channel() / 3;
	int fw = blob->width();
	int fh = blob->height();
	float* ptr = blob->mutable_cpu_data() + blob->offset(n);

	if (rows == 0 && cols == 0){
		rows = sqrt((float)num);
		rows = rows < 1 ? 1 : rows;
		cols = ceil(num / (float)rows);
	}
	else if (rows == 0){
		rows = ceil(num / (float)cols);
	}
	else if (cols == 0){
		cols = ceil(num / (float)rows);
	}

	Mat dst = Mat::zeros(fh * rows, fw * cols, CV_32FC3);
	int c = 0;
	double mi = 0, mx = 1;

	for (int y = 0; y < rows; ++y){
		for (int x = 0; x < cols; ++x){
			if (c < channels){
				Mat ms[3];
				for (int i = 0; i < 3; ++i){
					ms[i] = Mat(fh, fw, CV_32F, ptr + c * fw*fh);
					c++;
				}
				Mat roi = dst(Rect(x * fw, y*fh, fw, fh));
				cv::merge(ms, 3, roi);

				if (normType == localNorm){
					double local_mi, local_mx;
					cv::minMaxIdx(roi, &local_mi, &local_mx);
					roi = (roi - local_mi) / (local_mx - local_mi);
				}
			}
			else{
				Mat roi = dst(Rect(x * fw, y*fh, fw, fh));
				line(roi, Point(0, roi.rows*0.5), Point(roi.cols, roi.rows*0.5), Scalar(1), 1);
				line(roi, Point(roi.cols*0.5, 0), Point(roi.cols*0.5, roi.rows), Scalar(1), 1);
			}
		}
	}

	string uname = format("%s c:%d fw:%d fh:%d", name.c_str(), channels, fw, fh);
	namedWindow(uname, CV_WINDOW_NORMAL);
	imshow(uname, dst);
}

void visualFeaturemap(const Mat& featuremap, Mat& showbgr){
	Scalar begin(1, 0, 0);
	Scalar end(0, 0, 1);
	for (int x = 0; x < featuremap.cols; ++x){
		for (int y = 0; y < featuremap.rows; ++y){
			float v = featuremap.at<float>(y, x);
			Vec3f& sv = showbgr.at<Vec3f>(y, x);

			for (int i = 0; i < 3; ++i){
				sv[i] = (end[i] - begin[i]) * v + begin[i];
			}
		}
	}
}

void visualizeFeatureMap_featuremap(Blob* blob, const string& name, int n, const vector<int>& channelIndex, int rows, int cols, int normType){
	int channels = blob->channel();
	int num = channelIndex.size();
	int fw = blob->width();
	int fh = blob->height();
	float* ptr = blob->mutable_cpu_data() + blob->offset(n);

	if (rows == 0 && cols == 0){
		rows = sqrt((float)num);
		rows = rows < 1 ? 1 : rows;
		cols = ceil(num / (float)rows);
	}
	else if (rows == 0){
		rows = ceil(num / (float)cols);
	}
	else if (cols == 0){
		cols = ceil(num / (float)rows);
	}

	Mat dst = Mat::zeros(fh * rows, fw * cols, CV_32F);
	int cInd = 0;
	double mi = 0, mx = 1;

	for (int y = 0; y < rows; ++y){
		for (int x = 0; x < cols; ++x){
			if (cInd < num){
				int c = channelIndex[cInd];
				Mat roi = dst(Rect(x * fw, y*fh, fw, fh));
				Mat fm = Mat(fh, fw, CV_32F, ptr + c * fw*fh);
				cInd++;

				if (normType == localNorm){
					double local_mi, local_mx;
					fm = fm.clone();
					cv::minMaxIdx(fm, &local_mi, &local_mx);
					fm = (fm - local_mi) / (local_mx - local_mi);
				}
				//visualFeaturemap(fm, roi);
				fm.copyTo(roi);
			}
			else{
				Mat roi = dst(Rect(x * fw, y*fh, fw, fh));
				line(roi, Point(0, roi.rows*0.5), Point(roi.cols, roi.rows*0.5), Scalar::all(1), 1);
				line(roi, Point(roi.cols*0.5, 0), Point(roi.cols*0.5, roi.rows), Scalar::all(1), 1);
			}
		}
	}

	string uname = name;// format("%s c:%d[range:%d] fw:%d fh:%d", name.c_str(), channels, num, fw, fh);
	namedWindow(uname, CV_WINDOW_FREERATIO);
	imshow(uname, dst);
}

void visualizeFeatureMap(Blob* blob, const string& name, int n, const vector<int>& channelIndex, int rows, int cols, int normType, int showType){
	if (showType == showFeatureMap){
		visualizeFeatureMap_featuremap(blob, name, n, channelIndex, rows, cols, normType);
	}
	else{
		visualizeFeatureMap_image(blob, name, n, rows, cols, normType);
	}
}

void watcherThread(){

	while (keeprun){
		if (singal_show_){
			singal_show_ = false;
			lock_.lock();
			for (auto& k : showWatcher){
				if (k.second.showType != showMatrix)
					k.second.show->copyFrom(k.second.blob.get(), false, true);
			}

			map<string, ShowInfo> showd = showWatcher;
			lock_.unlock();

			for (auto& k : showd){
				auto& item = k.second;
				if (item.showType == showMatrix){
					string uname = item.name;//format("%s c:%d fw:%d fh:%d", item.name.c_str(), item.matrix.channels(), item.matrix.cols, item.matrix.rows);'

					//cv::normalize(item.matrix, item.matrix, 1.0, 0, NORM_MINMAX);
					namedWindow(uname, CV_WINDOW_FREERATIO);
					imshow(uname, item.matrix);
				}
				else{
					visualizeFeatureMap(item.blob.get(), item.name, item.n, item.channelsIndex, item.rows, item.cols, item.normType, item.showType);
				}
			}
		}

		int key = waitKey(1);
		if (getSolver()){
			Solver* solver = getSolver();
			if (key == 'w'){
				solver->setBaseLearningRate(solver->getBaseLearningRate() * 10);
				printf("baselr: %g\n", solver->getBaseLearningRate());
			}
			else if (key == 's'){
				solver->setBaseLearningRate(solver->getBaseLearningRate() * 0.1);
				printf("baselr: %g\n", solver->getBaseLearningRate());
			}
			else if (key == ' '){
				solver->postSnapshotSignal();
			}
		}
	}
}

void postMatrix(Mat matrix, const string& name){
	lock_.lock();
	if (showWatcher.find(name) == showWatcher.end()){
		ShowInfo& item = showWatcher[name];
	}

	ShowInfo& item = showWatcher[name];
	item.name = name;
	item.normType = norNorm;
	item.showType = showMatrix;
	matrix.copyTo(item.matrix);
	lock_.unlock();
	singal_show_ = true;
}

void postBlob(Blob* blob, const string& name, int n, int rows, int cols, int normType, int showType){
	postBlob(blob, name, range(blob->channel()), n, rows, cols, normType, showType);
}

void postBlob(Blob* blob, const string& name, const vector<int>& channelIndex, int n, int rows, int cols, int normType, int showType){
	if (channelIndex.size() == 0) return;

	lock_.lock();
	if (showWatcher.find(name) == showWatcher.end()){
		ShowInfo& item = showWatcher[name];
		item.blob = newBlob();
		item.show = newBlob();
	}

	ShowInfo& item = showWatcher[name];
	item.blob->copyFrom(blob, false, true);
	item.cols = cols;
	item.name = name;
	item.n = n;
	item.rows = rows;
	item.normType = normType;
	item.showType = showType;
	item.channelsIndex = channelIndex;
	lock_.unlock();
	singal_show_ = true;
}

void initializeVisualze(){
	keeprun = true;
	g_work_thread = thread(watcherThread);
}

void destoryVisualze(){
	keeprun = false;

	if (g_work_thread.joinable())
		g_work_thread.join();
}