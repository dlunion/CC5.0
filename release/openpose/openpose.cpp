


#include <cv.h>
#include <highgui.h>
#include "cc_nb.h"

using namespace cv;
using namespace cc;
using namespace std;

namespace L = cc::layers;

cc::Tensor _conv2d(cc::Tensor& input, int kernel_size, int num_output, const string& name){

	auto x = L::conv2d(input, { kernel_size, kernel_size, num_output }, "same", { 1, 1 }, { 1, 1 }, name);
	L::OConv2D* conv = (L::OConv2D*)x->owner.get();
	conv->bias_initializer.reset(new cc::Initializer());
	conv->kernel_initializer.reset(new cc::Initializer());

	conv->kernel_initializer->type = "gaussian";
	conv->kernel_initializer->stdval = 0.01;
	conv->bias_initializer->type = "constant";
	conv->bias_initializer->value = 0;
	return x;
}

cc::Tensor VGG16(const cc::Tensor& input){

	string vgg16_define[] = { "64", "64", "M", "128", "128", "M", "256", "256", "256", "256", "M", "512", "512" };
	int num_layers = sizeof(vgg16_define) / sizeof(vgg16_define[0]);
	string* usedefine = vgg16_define;

	auto x = input;
	int numpool = 1;
	int numconv = 1;
	int numchildren = 1;
	for (int i = 0; i < num_layers; ++i){
		if (usedefine[i] == "M"){
			x = L::max_pooling2d(x, { 2, 2 }, { 2, 2 }, { 0, 0 }, false, cc::f("pool%d_stage1", numpool++));
			numchildren = 1;
			numconv++;
		}
		else{
			int num_output = atoi(usedefine[i].c_str());
			x = _conv2d(x, 3, num_output, cc::f("conv%d_%d", numconv, numchildren));
			x = L::relu(x, cc::f("relu%d_%d", numconv, numchildren));
			numchildren++;
		}
	}
	return x;
}

cc::Tensor CPMHead(const cc::Tensor& input,
	const string& convnamefmt, const string& relunamefmt, const vector<int>& numoutput, int kernelSize){

	auto x = input;
	for (int i = 1; i <= numoutput.size(); ++i){

		int usekernelsize;
		if (i <= numoutput.size() - 2)
			usekernelsize = kernelSize;
		else
			//last two layer is 1x1
			usekernelsize = 1;

		x = _conv2d(x, usekernelsize, numoutput[i - 1], cc::f(convnamefmt.c_str(), i));
		if (i != numoutput.size())
			x = L::relu(x, cc::f(relunamefmt.c_str(), i));
	}
	return x;
}

cc::Tensor OpenPoseCPMVGG16(const cc::Tensor& input, int num_stage, int l1Output = 38, int l2Output = 19){

	auto x = VGG16(input);
	x = _conv2d(x, 3, 256, cc::f("conv%d_%d_CPM", 4, 3));
	x = L::relu(x, cc::f("relu%d_%d_CPM", 4, 3));

	x = _conv2d(x, 3, 128, cc::f("conv%d_%d_CPM", 4, 4));
	x = L::relu(x, cc::f("relu%d_%d_CPM", 4, 4));

	auto backbone = x;
	auto l1 = CPMHead(backbone, "conv5_%d_CPM_L1", "relu5_%d_CPM_L1", { 128, 128, 128, 512, l1Output }, 3);
	auto l2 = CPMHead(backbone, "conv5_%d_CPM_L2", "relu5_%d_CPM_L2", { 128, 128, 128, 512, l2Output }, 3);
	x = L::concat({ l1, l2, backbone }, 1, "concat_stage2");

	for (int i = 0; i < num_stage; ++i){

		int stage = i + 2;
		auto l1 = CPMHead(x, cc::f("Mconv%%d_stage%d_L1", stage), cc::f("Mrelu%%d_stage%d_L1", stage), { 128, 128, 128, 128, 128, 128, l1Output }, 7);
		auto l2 = CPMHead(x, cc::f("Mconv%%d_stage%d_L2", stage), cc::f("Mrelu%%d_stage%d_L2", stage), { 128, 128, 128, 128, 128, 128, l2Output }, 7);

		if (i < num_stage - 1)
			x = L::concat({ l1, l2, backbone }, 1, cc::f("concat_stage%d", stage + 1));
		else
			x = L::concat({ l1, l2 }, 1, cc::f("concat_stage%d", stage + 1));
	}
	return x;
}

bool saveToFile(const string& path, const string& data){

	FILE* f = fopen(path.c_str(), "wb");
	if (!f) return false;

	fwrite(data.data(), 1, data.size(), f);
	fclose(f);
	return true;
}

Scalar getColor(int label){
	static vector<Scalar> colors;
	if (colors.size() == 0){
		colors.push_back(Scalar(255, 0, 0));
		colors.push_back(Scalar(0, 255, 0));
		colors.push_back(Scalar(0, 0, 255));
		colors.push_back(Scalar(0, 255, 255));
		colors.push_back(Scalar(255, 0, 255));
		colors.push_back(Scalar(128, 0, 255));
		colors.push_back(Scalar(128, 255, 255));
		colors.push_back(Scalar(255, 128, 255));
		colors.push_back(Scalar(128, 255, 128));
		colors.push_back(Scalar(255, 128, 255));
	}
	return colors[label % colors.size()];
}

template<typename _Type>
static _Type getGradientColor(const _Type& begin, const _Type& end, float val){
	return _Type(
		cv::saturate_cast<uchar>(begin[0] + (end[0] - begin[0])*val),
		cv::saturate_cast<uchar>(begin[1] + (end[1] - begin[1])*val),
		cv::saturate_cast<uchar>(begin[2] + (end[2] - begin[2])*val)
		);
}

static void renderMask(Mat& rgbImage, const Mat& mask, Scalar color){

	CV_Assert(mask.size() == rgbImage.size());

	for (int i = 0; i < mask.cols; ++i){
		for (int j = 0; j < mask.rows; ++j){
			float val = mask.at<float>(j, i);

			if (val){
				Vec3b& v = rgbImage.at<Vec3b>(j, i);
				v = getGradientColor(v, Vec3b(color[0], color[1], color[2]), val);
			}
		}
	}
}

template<typename _T>
float l2distance(const _T& a, const _T& b){
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

struct PartPoint{
	Point2f point;
	float score;
	int classes;

	PartPoint(Point2f point, float score, int classes) :
		point(point), score(score), classes(classes){}
};

vector<vector<PartPoint>> peaksNMS(Blob* input, float threshold, int stride, float scalex, float scaley){
	//maxPeaks就是最大人数，+1是为了第一位存个数
	//算法，是每个点，如果大于阈值，同时大于上下左右值的时候，则认为是峰值

	//算法很简单，featuremap的任意一个点，其上下左右和斜上下左右，都小于自身，就认为是要的点
	//然后以该点区域，选择7*7区域，按照得分值和x、y来计算最合适的亚像素坐标

	float innerNMSDistance = 20;
	float outerNMSDistance = 5;
	vector<PartPoint> allparts;
	int w = input->width();
	int h = input->height();
	int plane_offset = w * h;
	const float* ptr = input->cpu_data();

	auto nmsFunction = [](vector<PartPoint>& parts, float thresholdDistance){
		std::sort(parts.begin(), parts.end(), [](PartPoint& a, PartPoint& b){
			return a.score > b.score;
		});

		for (int k = 0; k < parts.size(); ++k){
			for (int l = k + 1; l < parts.size(); ++l){
				if (l2distance(parts[k].point, parts[l].point) < thresholdDistance){
					parts.erase(parts.begin() + l);
					l--;
				}
			}
		}
	};

	for (int n = 0; n < input->num(); ++n){
		for (int c = 0; c < input->channel() - 1; ++c){

			vector<PartPoint> innerPartVector;
			int num_peaks = 0;
			for (int y = 1; y < h - 1; ++y){
				for (int x = 1; x < w - 1; ++x){
					float value = ptr[y*w + x];
					if (value > threshold){
						const float topLeft = ptr[(y - 1)*w + x - 1];
						const float top = ptr[(y - 1)*w + x];
						const float topRight = ptr[(y - 1)*w + x + 1];
						const float left = ptr[y*w + x - 1];
						const float right = ptr[y*w + x + 1];
						const float bottomLeft = ptr[(y + 1)*w + x - 1];
						const float bottom = ptr[(y + 1)*w + x];
						const float bottomRight = ptr[(y + 1)*w + x + 1];

						//gaussian
						if (value > topLeft && value > top && value > topRight
							&& value > left && value > right
							&& value > bottomLeft && value > bottom && value > bottomRight)
						{
							//subpix
							float xAcc = 0;
							float yAcc = 0;
							float scoreAcc = 0;
							for (int kx = -3; kx <= 3; ++kx){
								int ux = x + kx;
								if (ux >= 0 && ux < w){
									for (int ky = -3; ky <= 3; ++ky){
										int uy = y + ky;
										if (uy >= 0 && uy < h){
											float score = ptr[uy * w + ux];
											xAcc += ux * score;
											yAcc += uy * score;
											scoreAcc += score;
										}
									}
								}
							}

							xAcc /= scoreAcc;
							yAcc /= scoreAcc;
							scoreAcc = value;
							innerPartVector.push_back(PartPoint(Point2f((xAcc + 0.5)* stride, (yAcc+0.5) * stride), scoreAcc, c));
							num_peaks++;
						}
					}
				}
			}
			ptr += plane_offset;

			//inner nms
			nmsFunction(innerPartVector, innerNMSDistance);
			allparts.insert(allparts.end(), innerPartVector.begin(), innerPartVector.end());
		}
	}
	nmsFunction(allparts, outerNMSDistance);

	//restore to raw image size
	vector<vector<PartPoint>> parts(input->channel() - 1);
	for (int k = 0; k < allparts.size(); ++k){
		allparts[k].point.x *= scalex;
		allparts[k].point.y *= scaley;
		parts[allparts[k].classes].push_back(allparts[k]);
	}
	return parts;
}

int main(){

	cc::setGPU(0);
	
	auto image = L::input({ 1, 3, 688, 368 }, "image");
	auto poseNetwork = OpenPoseCPMVGG16(image, 5);
	auto net = cc::engine::caffe::buildNet(poseNetwork);
	//auto net = loadNetFromPrototxt("net.prototxt");
	//net->input_blob(0)->reshape(1, 3, 688, 368);
	//net->reshape();

	//这里是生成一个net.prototxt文件，用来查看，实际上我们不需要prototxt文件，因为代码可以生成
	//pose_deploy_linevec.prototxt是官方原版参照的文件
	cc::engine::caffe::buildGraphToFile(poseNetwork, "net.prototxt");

	//我们加载模型
	if (!net->weightsFromFile("pose_iter_440000.caffemodel")){

		//模型异常不崩溃
		printf("load weights fail.\n");
	}

	Mat im = imread("demo.jpg");
	Mat show = im.clone();
	Mat rawShow = show.clone();

	double tick = getTickCount();
	im.convertTo(im, CV_32F, 1 / 256.0, -0.5);

	//如果是opencv2410，或者是跟libcaffe.dll编译使用同一个opencv版本，则可以直接使用setData(0, im)
	//否则需要用下面做法
	//net->input_blob(0)->setData(0, im);
	net->input_blob(0)->setData(0, CVFMat(im));
	net->forward();
	net->output_blob(0)->mutable_cpu_data();
	tick = (getTickCount() - tick) / getTickFrequency() * 1000;
	printf("tick: %.2f ms\n", tick);

	float scalex = im.cols / (float)net->input_blob(0)->width();
	float scaley = im.rows / (float)net->input_blob(0)->height();
	Blob* keypoints = net->blob("Mconv7_stage6_L2");
	auto parts = peaksNMS(keypoints, 0.25, 8, scalex, scaley);

	for (int c = 0; c < parts.size(); ++c){
		for (int p = 0; p < parts[c].size(); ++p)
			circle(rawShow, Point(parts[c][p].point.x, parts[c][p].point.y), 10, getColor(c), 2);
	}

	for (int i = 0; i < keypoints->channel()-1; ++i){

		Mat mask(keypoints->height(), keypoints->width(), CV_32F, keypoints->mutable_cpu_data() + keypoints->offset(0, i));
		resize(mask, mask, show.size(), 0, 0, INTER_CUBIC);
		renderMask(show, mask, getColor(i));
	}
	imshow("openpose demo", show);
	imshow("peak nms", rawShow);
	imwrite("result.jpg", show);
	waitKey();
	return 0;
}