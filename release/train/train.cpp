


#include <cv.h>
#include <highgui.h>
#include "cc_nb.h"

using namespace cv;
using namespace cc;
using namespace std;

namespace L = cc::layers;

#define InputWidth		32
#define InputHeight		32
#define BatchSize		128

int randr(int low, int high){

	if (low > high) std::swap(low, high);
	return rand() % (high - low + 1) + low;
}

float randr(float low, float high){

	if (low > high) std::swap(low, high);
	return rand() / (float)RAND_MAX * (high - low) + low;
}

int loadCifarImageLabel(const string filepath, const vector<string> filename, vector<Mat>& images, vector<int>& labels) {
	int img_w = 32;
	int img_h = 32;
	int img_c = 3;
	string file;
	for (int i = 0; i < filename.size(); i++) {
		file = (filepath + filename[i]);
		FILE* f = fopen(file.c_str(), "rb");
		if (!f) return 0;
		int num = 10000;
		char lab = 0;
		Mat R(img_w, img_h, CV_8UC1);
		Mat G(img_w, img_h, CV_8UC1);
		Mat B(img_w, img_h, CV_8UC1);
		for (int i = 0; i < num; ++i) {
			fread(&lab, 1, 1, f);
			labels.push_back(lab);
			fread(R.data, 1, img_w * img_h, f);
			fread(G.data, 1, img_w * img_h, f);
			fread(B.data, 1, img_w * img_h, f);
			std::vector<cv::Mat> tmp{ B, G, R };
			cv::Mat bgr;
			cv::merge(tmp, bgr);
			images.emplace_back(bgr);
		}
		fclose(f);
	}
	return 0;
}

void mulMat(const Mat& a, int type, const Mat& b, Mat& d){

	static Mat ms[3];
	split(a, ms);
	for (int i = 0; i < 3; ++i){
		ms[i].convertTo(ms[i], CV_32F);
		ms[i] = ms[i].mul(b);
		ms[i].convertTo(ms[i], CV_MAT_DEPTH(type));
	}
	merge(ms, 3, d);
}

void multi_padding_and_random_crop(const vector<Mat>& ims, float scale_min, float scale_max, int padding, vector<Mat>& outs, int type, float mixup){

	int imbatch = ims.size();
	int imheight = ims[0].rows;
	int imwidth = ims[0].cols;

	bool usemixup = mixup != 1;
	bool usescale = !(scale_min == scale_max && scale_min == 1);
	Mat background = Mat::zeros(imheight + padding * 2, imwidth + padding * 2, type);
	Mat mixup_heatmap = Mat::zeros(imheight, imwidth, CV_32F);
	Mat used_image = Mat::zeros(imheight, imwidth, type);
	Mat mix_image = Mat::zeros(imheight, imwidth, type);

	for (int i = 0; i < imbatch; ++i){

		if (usemixup){
			cv::randu(mixup_heatmap, mixup, 1.0);
			///used_image = ims[i].mul(mixup_heatmap);
			mulMat(ims[i], type, mixup_heatmap, used_image);

			int mixind = rand() % imbatch;
			//mix_image = ims[mixind].mul(1 - mixup_heatmap);
			mulMat(ims[mixind], type, 1 - mixup_heatmap, mix_image);
			used_image += mix_image;

			used_image.copyTo(background(Rect(padding, padding, imwidth, imheight)));
		}
		else{
			ims[i].copyTo(background(Rect(padding, padding, imwidth, imheight)));
		}

		if (usescale){
			if (rand() % 5 != 0){
				float acc = rand() / (float)RAND_MAX;
				float scale = acc * (scale_max - scale_min) + scale_min;

				int uw = imwidth * scale;
				int uh = imheight * scale;
				int rx = background.cols == uw ? 0 : rand() % (background.cols - uw);
				int ry = background.rows == uh ? 0 : rand() % (background.rows - uh);
				Rect roi = Rect(rx, ry, uw, uh) & Rect(0, 0, background.cols, background.rows);
				if (roi.area() > 0){
					resize(background(roi), outs[i], Size(imwidth, imheight));
				}
				else{
					printf("error uw(%d), uh(%d), scale(%f), scale_min(%f), scale_max(%f)\n", uw, uh, scale, scale_min, scale_max);
					return;
				}
			}

			if (rand() % 5 != 0){
				float cutout_acc = rand() / (float)RAND_MAX;
				float cutout_scale = cutout_acc * (scale_max - scale_min) + scale_min;

				int cutout_uw = (imwidth * 0.5) * cutout_scale;
				int cutout_uh = (imheight * 0.5) * cutout_scale;
				int cutout_rx = imwidth == cutout_uw ? 0 : rand() % (imwidth - cutout_uw);
				int cutout_ry = imheight == cutout_uh ? 0 : rand() % (imheight - cutout_uh);
				Rect cutout_roi = Rect(cutout_rx, cutout_ry, cutout_uw, cutout_uh) & Rect(0, 0, imwidth, imheight);
				if (cutout_roi.area() > 0){
					outs[i](cutout_roi).setTo(0);
				}
				else{
					printf("error uw(%d), uh(%d), scale(%f), scale_min(%f), scale_max(%f)", cutout_uw, cutout_uh, cutout_scale, scale_min, scale_max);
					return;
				}
			}
		}
		else{
			int rx = padding == 0 ? 0 : rand() % (padding * 2);
			int ry = padding == 0 ? 0 : rand() % (padding * 2);
			background(Rect(rx, ry, imwidth, imheight)).copyTo(outs[i]);
		}

		if (rand() % 2 == 0)
			flip(outs[i], outs[i], (rand() % 3) - 1);
	}
}

class CifarDataLayer : public BaseLayer {
public:
	SETUP_LAYERFUNC(CifarDataLayer);

	void preperData(int phase) {

		vector<string> img_labelfile;

		if (phase == PhaseTest) {
			img_labelfile.push_back("test_batch.bin");
		}
		else {
			img_labelfile.push_back("data_batch_1.bin");
			img_labelfile.push_back("data_batch_2.bin");
			img_labelfile.push_back("data_batch_3.bin");
			img_labelfile.push_back("data_batch_4.bin");
			img_labelfile.push_back("data_batch_5.bin");
		}
		loadCifarImageLabel("CIFAR_data/", img_labelfile, images_, labels_);
		inds_.resize(images_.size());
		cursor_ = 0;

		for (int i = 0; i < inds_.size(); ++i)
			inds_[i] = i;

		std::random_shuffle(inds_.begin(), inds_.end());
	}

	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop) {

		Blob* image = top[0];
		Blob* label = top[1];

		for (int i = 0; i < batch_size_; ++i) {
			int ind = inds_[cursor_++];
			batchLabs_[i] = labels_[ind];
			images_[ind].convertTo(batchImage_[i], CV_32F, 2 / 255.0, -1.0);// , 1 / 255.0, -0.5);

			if (cursor_ == inds_.size()) {
				std::random_shuffle(inds_.begin(), inds_.end());
				cursor_ = 0;
			}
		}

		if (this->phase_ == PhaseTrain){

			multi_padding_and_random_crop(batchImage_, 1, 1, 4, augmentResult_, CV_32FC3, 0.8);
			for (int i = 0; i < batch_size_; ++i) {
				image->setData(i, augmentResult_[i]);
				label->mutable_cpu_data()[i] = batchLabs_[i];
			}
		}
		else{
			for (int i = 0; i < batch_size_; ++i) {
				image->setData(i, batchImage_[i]);
				label->mutable_cpu_data()[i] = batchLabs_[i];
			}
		}
	}

	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) {

		Blob* data = top[0];
		Blob* label = top[1];

		this->phase_ = phase;
		this->batch_size_ = BatchSize;

		data->reshape(batch_size_, 3, InputHeight, InputWidth);
		label->reshape(batch_size_, 1, 1, 1);
		batchImage_.resize(batch_size_);
		batchLabs_.resize(batch_size_);
		augmentResult_.resize(batch_size_);
		preperData(phase);
	}

private:
	vector<Mat> augmentResult_;
	vector<int> batchLabs_;
	vector<Mat> batchImage_;
	vector<int> labels_;
	vector<Mat> images_;
	vector<int> inds_;
	int cursor_;
	int batch_size_;
	int phase_;
};


//第一节单独卷积
cc::Tensor resnet_conv(const cc::Tensor& input, const vector<int>& kernel, const string& name, int stride = 1, bool has_relu = true, bool bias_term = false){

	auto x = L::conv2d(input, kernel, "same", { stride, stride }, { 1, 1 }, name);
	L::OConv2D* layer = (L::OConv2D*)x->owner.get();
	layer->bias_term = bias_term;
	layer->kernel_initializer.reset(new cc::Initializer());
	layer->kernel_initializer->type = "msra";
	layer->bias_initializer.reset();
	layer->bias_mult.reset();

	x = L::batch_norm_only(x, "bn_" + name);
	x = L::scale(x, true, "scale_" + name);
	if (has_relu)
		x = L::relu(x, name + "_relu");
	return x;
}

//残差块卷积模块.构造一个分支卷积模块，左右通用
cc::Tensor resnet_conv_block(const cc::Tensor& input, const vector<int>& kernel, int innum1, string part1, int innum2, string part2, int stride = 1, bool has_relu = true, bool bias_term = false){

	auto x = L::conv2d(input, kernel, "same", { stride, stride }, { 1, 1 }, cc::f("res%d", innum1) + part1 + cc::f("_branch%d%s", innum2, part2.c_str()));
	L::OConv2D* layer = (L::OConv2D*)x->owner.get();
	layer->bias_term = bias_term;
	layer->kernel_initializer.reset(new cc::Initializer());
	layer->kernel_initializer->type = "msra";
	layer->bias_initializer.reset();
	layer->bias_mult.reset();

	x = L::batch_norm_only(x, cc::f("bn%d", innum1) + part1 + cc::f("_branch%d%s", innum2, part2.c_str()));
	x = L::scale(x, true, cc::f("scale%d", innum1) + part1 + cc::f("_branch%d%s", innum2, part2.c_str()));
	if (has_relu)
		x = L::relu(x, cc::f("res%d", innum1) + part1 + cc::f("_branch%d%s_relu", innum2, part2.c_str()));

	return x;
}

//右侧分支模块，由三个卷积模块组成
cc::Tensor resnet_branch2(const cc::Tensor& input, int stride, int innum, int outnum, int stage, string part){

	auto right = input;
	right = resnet_conv_block(right, { 3, 3, innum }, stage, part, 2, "a", stride, true);
	right = resnet_conv_block(right, { 3, 3, outnum }, stage, part, 2, "b", 1, false);
	//right = resnet_conv_block(right, { 3, 3, outnum }, stage, part, 2, "c", 1, false);
	return right;
}

cc::Tensor resnet_block(const cc::Tensor& input, int n, int stride, int innum, int outnum, int numinner){

	auto x = input;
	{
		auto branch1 = resnet_conv_block(x, { 1, 1, outnum }, n, "a", 1, "", stride, false, false); //构建左侧分支
		auto branch2 = resnet_branch2(x, stride, innum, outnum, n, "a");
		auto out = L::add(branch1, branch2, cc::f("res%da", n));
		x = L::relu(out, cc::f("res%da_relu", n));
	};

	string buff[5] = { "b", "c", "d", "e", "f" };
	for (int i = 0; i < numinner; ++i){
		auto branch1 = x;
		auto branch2 = resnet_branch2(x, 1, innum, outnum, n, buff[i]);
		auto out = L::add(branch1, branch2, cc::f("res%d%s", n, buff[i].c_str()));
		x = L::relu(out, cc::f("res%d%s_relu", n, buff[i].c_str()));
	};
	return x;
}

cc::Tensor resnet50(const cc::Tensor& input, int numunit){

	auto x = input;
	{
		//cc::name_scope n("input");
		x = resnet_conv(x, { 7, 7, 64 }, "conv1", 2, true, true);
		x = L::max_pooling2d(x, { 3, 3 }, { 2, 2 }, { 0, 0 }, false, "pool1");
	};

	x = resnet_block(x, 2, 1, 64, 256, 2);
	x = resnet_block(x, 3, 2, 128, 512, 3);
	x = resnet_block(x, 4, 2, 256, 1024, 5);
	x = resnet_block(x, 5, 2, 512, 2048, 2);

	x = L::avg_pooling2d(x, { 1, 1 }, { 1, 1 }, { 0, 0 }, true, "pool5");
	x = L::dense(x, numunit, "fc", true);
	return x;
}

cc::Tensor resnet18(const cc::Tensor& input, int numunit){
	auto x = input;
	{
		//cc::name_scope n("input");
		x = resnet_conv(x, { 3, 3, 16 }, "conv1", 1, true, true);
		//x = L::max_pooling2d(x, { 3, 3 }, { 2, 2 }, { 0, 0 }, false, "pool1");
	};

	x = resnet_block(x, 2, 1, 16, 32, 2);
	x = resnet_block(x, 3, 2, 32, 64, 2);
	x = resnet_block(x, 4, 2, 64, 64, 2);

	x = L::avg_pooling2d(x, { 1, 1 }, { 1, 1 }, { 0, 0 }, true, "pool5");
	x = L::dense(x, numunit, "fc", true);
	L::ODense* denseLayer = (L::ODense*)x->owner.get();
	denseLayer->weight_initializer.reset(new cc::Initializer("msra"));
	denseLayer->bias_initializer.reset(new cc::Initializer("constant", 0));
	return x;
}

bool saveToFile(const string& path, const string& data){

	FILE* f = fopen(path.c_str(), "wb");
	if (!f) return false;

	fwrite(data.data(), 1, data.size(), f);
	fclose(f);
	return true;
}

int main(){

	cc::installRegister();
	INSTALL_LAYER(CifarDataLayer);

	auto data = L::data("CifarDataLayer", {"image", "label"}, "data");
	auto image = data[0];
	auto label = data[1];
	auto x = resnet18(image, 10);
	auto test = cc::metric::classifyAccuracy(x, label, "accuracy");
	auto loss = cc::loss::softmax_cross_entropy(x, label, "loss");
	 
	int trainDatasetSize = 50000;
	int valDatasetSize = 10000;
	int trainEpochs = 150;
	int epochIters = trainDatasetSize / BatchSize;

	cc::engine::caffe::buildGraphToFile(resnet18(L::input({ 1, 3, InputHeight, InputWidth }, "image"), 10), "deploy.prototxt");
	auto op = cc::optimizer::momentumStochasticGradientDescent(cc::learningrate::step(0.1, 0.1, 20*epochIters), 0.9);
	op->test_initialization = false;
	//op->regularization_type = "L2";
	op->test_interval = epochIters;
	op->test_iter = valDatasetSize / BatchSize;
	op->weight_decay = 0.0002f;
	op->average_loss = 1;
	op->max_iter = trainEpochs * epochIters;
	op->display = 10;
	op->device_ids = { 0,1 };
	//op->reload_weights = "saved_iter[26520]_loss[0.340022]_accuracy[0.896635].caffemodel";
	op->minimize({ loss, test });
	//op->minimizeFromFile("net.prototxt");

	saveToFile("solver.prototxt", op->seril());
	cc::engine::caffe::buildGraphToFile({ loss, test }, "train.prototxt");

	//setup test classifier callback
	registerOnTestClassificationFunction([](Solver* solver, float testloss, int index, const char* itemname, float accuracy){
		if (strcmp(itemname, "accuracy") == 0){

			static string lastsavedname;
			static float bestAccuracy = 0, bestLoss = 0;
			static int bestIter = 0;
			static string datarootdir = ".";
			if (accuracy >= bestAccuracy){
				if (!lastsavedname.empty())
					remove(lastsavedname.c_str());

				int iter = solver->iter();
				bestIter = iter;
				bestAccuracy = accuracy;
				bestLoss = testloss;

				string savedname = format("%s/saved_iter[%d]_loss[%f]_accuracy[%f].caffemodel", datarootdir.c_str(), iter, testloss, accuracy);
				solver->net()->saveToCaffemodel(savedname.c_str());
				lastsavedname = savedname;

				if (accuracy >= 1){
					printf("accuracy 提前满足要求，退出.\n");
					solver->postEarlyStopSignal();
				}
			}
		}
	});

	cc::train::caffe::run(op, [](OThreadContextSession* session, int step, float smoothed_loss){
		//do smothing...
	});
	return 0;
}