
#include <opencv.hpp>
#include "caffe/cc/core/cc_v5.h"
#include "caffe/blob.hpp"
#include <vector>
#include "caffe/util/math_functions.hpp"
using namespace std;
using namespace cv;

namespace cc{

#define cvt(p)	((caffe::Blob<float>*)p)
#define ptr		(cvt(this->_native))

	CCAPI CCCALL Blob::Blob(){
		memset(this->_dims, 0, sizeof(this->_dims));
		this->_num_axis = 0;
		this->_native = nullptr;
	}

	CCAPI void CCCALL releaseBlob(Blob* blob){
		if (blob){
			caffe::Blob<float>* p = cvt(blob->getNative());
			if (p) delete p;
		}
	}

	CCAPI std::shared_ptr<Blob> CCCALL newBlob(){
		caffe::Blob<float>* blob = new caffe::Blob<float>();
		return std::shared_ptr<Blob>(blob->ccBlob(), releaseBlob);
	}

	CCAPI std::shared_ptr<Blob> CCCALL newBlobByShape(int num, int channels, int height, int width){
		caffe::Blob<float>* blob = new caffe::Blob<float>(num, channels, height, width);
		return std::shared_ptr<Blob>(blob->ccBlob(), releaseBlob);
	}

	CCAPI std::shared_ptr<Blob> CCCALL newBlobByShapes(int numShape, int* shapes){
		caffe::Blob<float>* blob = new caffe::Blob<float>(vector<int>(shapes, shapes + numShape));
		return std::shared_ptr<Blob>(blob->ccBlob(), releaseBlob);
	}

	CCAPI void CCCALL Blob::updateInfo(){
		this->_dims[0] = this->num();
		this->_dims[1] = this->channel();
		this->_dims[2] = this->height();
		this->_dims[3] = this->width();
		this->_num_axis = this->num_axes();
	}

	CCAPI void CCCALL Blob::transpose(int axis0, int axis1, int axis2, int axis3){

		int muls[] = { count(1), count(2), count(3), 1 };
		//auto t = newBlobByShape(_dims[axis0], _dims[axis1], _dims[axis2], _dims[axis3]);
		BlobData temp;
		temp.reshape(_dims[axis0], _dims[axis1], _dims[axis2], _dims[axis3]);
		float* dataptr = temp.list;
		const float* srcptr = this->cpu_data();
		int tcount1 = temp.count(1);
		int tcount2 = temp.count(2);
		int tcount3 = temp.count(3);

		for (int a0 = 0; a0 < _dims[axis0]; ++a0){
			for (int a1 = 0; a1 < _dims[axis1]; ++a1){
				for (int a2 = 0; a2 < _dims[axis2]; ++a2)
					for (int a3 = 0; a3 < _dims[axis3]; ++a3)
						dataptr[a0 * tcount1 + a1 * tcount2 + a2 * tcount3 + a3] = srcptr[a0 * muls[axis0] + a1 * muls[axis1] + a2 * muls[axis2] + a3 * muls[axis3]];
			}
		}
		this->copyFrom(&temp);
	}

	void blobsetData(Blob* blob, int numIndex, const Mat& data, const Scalar& meanValue, float scale){
		CHECK(!data.empty()) << "data is empty";
		CHECK_EQ(data.channels(), blob->channel()) << "data channel error";

		int w = blob->width();
		int h = blob->height();
		Mat udata = data;
		if (udata.size() != cv::Size(w, h))
			resize(udata, udata, cv::Size(w, h));

		if (CV_MAT_DEPTH(udata.type()) != CV_32F)
			udata.convertTo(udata, CV_32F);

		if (meanValue[0] != 0 || meanValue[1] != 0 || meanValue[2] != 0)
			udata -= meanValue;

		if (scale != 1)
			udata *= scale;

		int channel_size = w*h;
		int num_size = blob->channel() * channel_size;
		float* input_data = blob->mutable_cpu_data() + num_size * numIndex;
		vector<cv::Mat> mats(data.channels());
		for (int i = 0; i < mats.size(); ++i)
			mats[i] = cv::Mat(h, w, CV_32F, input_data + channel_size * i);

		split(udata, mats);
		CHECK_EQ((float*)mats[0].data, input_data) << "error, split pointer fail.";
	}

	void Blob::setData(int numIndex, const unsigned char* imdataptr, int width, int height, int channels, const CCScalar& meanValue, float scale){

		Mat im(height, width, CV_8UC(channels), (uchar*)imdataptr);
		blobsetData(this, numIndex, im, CCScal(meanValue), scale);
	}

	void Blob::setData(int numIndex, const float* imdataptr, int width, int height, int channels, const CCScalar& meanValue, float scale){

		Mat im(height, width, CV_32FC(channels), (uchar*)imdataptr);
		blobsetData(this, numIndex, im, CCScal(meanValue), scale);
	}

	bool Blob::setData(int numIndex, const void* imdataptr, int datalength, int color, const CCScalar& meanValue, float scale){

		Mat im;
		try{
			im = imdecode(Mat(1, datalength, CV_8U, (char*)imdataptr), color);
		}catch (...){}

		if (im.empty())
			return false;

		blobsetData(this, numIndex, im, CCScal(meanValue), scale);
		return true;
	}

#if 0
	void Blob::setData(int numIndex, const Mat& data, const Scalar& meanValue, float scale){
		CHECK(!data.empty()) << "data is empty";
		CHECK_EQ(data.channels(), this->channel()) << "data channel error";

		int w = this->width();
		int h = this->height();
		Mat udata = data;
		if (udata.size() != cv::Size(w, h))
			resize(udata, udata, cv::Size(w, h));

		if (CV_MAT_DEPTH(udata.type()) != CV_32F)
			udata.convertTo(udata, CV_32F);

		if (meanValue[0] != 0 || meanValue[1] != 0 || meanValue[2] != 0)
			udata -= meanValue;

		if (scale != 1)
			udata *= scale;

		int channel_size = w*h;
		int num_size = this->channel() * channel_size;
		float* input_data = this->mutable_cpu_data() + num_size * numIndex;
		vector<cv::Mat> mats(data.channels());
		for (int i = 0; i < mats.size(); ++i)
			mats[i] = cv::Mat(h, w, CV_32F, input_data + channel_size * i);

		split(udata, mats);
		CHECK_EQ((float*)mats[0].data, input_data) << "error, split pointer fail.";
	}
#endif

	int Blob::shape(int index) const {
		return ptr->LegacyShape(index);
	}

	shared_ptr<Blob> Blob::clone(bool clonediff){
		shared_ptr<Blob> newb = newBlob();
		newb->copyFrom(this, false, true);

		if (clonediff)
			newb->copyFrom(this, true, false);
		return newb;
	}

	int Blob::num_axes() const {
		return ptr->num_axes();
	}

	void Blob::set_cpu_data(float* data){
		ptr->set_cpu_data(data);
	}

	void Blob::setNative(void* native){
		this->_native = native;
	}

	const void* Blob::getNative() const{
		return this->_native;
	}

	const float* Blob::cpu_data(bool need_to_cpu) const{
		return ptr->cpu_data(need_to_cpu);
	}

	const float* Blob::gpu_data(bool need_to_gpu) const{
		return ptr->gpu_data(need_to_gpu);
	}

	int Blob::count() const {
		return ptr->count();
	}

	int Blob::count(int start_axis) const {
		return ptr->count(start_axis);
	}

	void Blob::reshape(int num, int channels, int height, int width){
		num = num == -1 ? ptr->num() : num;
		channels = channels == -1 ? ptr->channels() : channels;
		height = height == -1 ? ptr->height() : height;
		width = width == -1 ? ptr->width() : width;
		ptr->Reshape(num, channels, height, width);
	}

	void Blob::setTo(float value){
		
		size_t c = ptr->count();
		if (c > 0)
			caffe::caffe_set(c, value, ptr->mutable_cpu_data());
	}

	int Blob::offset(const int n, const int c, const int h, const int w) const{
		return ptr->offset(n, c, h, w);  
	}

	void Blob::reshape(int numShape, int* shapeDims){
		ptr->Reshape(vector<int>(shapeDims, shapeDims + numShape));
	}

	void Blob::reshapeLike(const Blob* other){
		CHECK(other != nullptr) << "null pointer exception";
		ptr->ReshapeLike(*cvt(other->_native));
	}

	void Blob::copyFrom(const Blob* other, bool copyDiff, bool reshape){
		ptr->CopyFrom(*cvt(other->_native), copyDiff, reshape);
	}

	void Blob::copyDiffFrom(const Blob* other){
		CHECK(other != nullptr) << "null pointer exception";
		CHECK_EQ(this->count(), other->count()) << "shape mismatch";
		if (other->count() > 0){
			memcpy(ptr->mutable_cpu_diff(), other->cpu_diff(), sizeof(float)*other->count());
		}
	}

	void Blob::copyFrom(const BlobData* other){
		CHECK(other != nullptr) << "null pointer exception";
		if (ptr->num() != other->num || ptr->channels() != other->channels || ptr->width() != other->width || ptr->height() != other->height){
			ptr->Reshape(other->num, other->channels, other->height, other->width);
		}

		if (other->count() > 0){
			memcpy(ptr->mutable_cpu_data(), other->list, sizeof(float)*other->count());
		}
	}

	float* Blob::mutable_cpu_data(bool need_to_cpu){
		return ptr->mutable_cpu_data(need_to_cpu);
	}

	float* Blob::mutable_gpu_data(bool need_to_gpu){
		return ptr->mutable_gpu_data(need_to_gpu);
	}

	const float* Blob::cpu_diff() const{
		return ptr->cpu_diff();
	}

	const float* Blob::gpu_diff() const{
		return ptr->gpu_diff();
	}

	float* Blob::mutable_cpu_diff(){
		return ptr->mutable_cpu_diff();
	}

	float* Blob::mutable_gpu_diff(){
		return ptr->mutable_gpu_diff();
	}

	int Blob::height() const {
		return ptr->height();
	}

	int Blob::width() const {
		return ptr->width();
	}

	int Blob::channel() const{
		return ptr->channels();
	}

	int Blob::num() const {
		return ptr->num();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////
	BlobData::BlobData()
		:list(0), num(0), height(0), width(0), channels(0), capacity_count(0)
	{}

	BlobData::~BlobData(){
		release();
	}

	bool BlobData::empty() const{
		return count() == 0;
	}

	int BlobData::count(int start_axis) const{

		int dims[] = {num, channels, height, width};
		start_axis = std::max(0, start_axis);
		start_axis = std::min(3, start_axis);

		int count_dims = 1;
		for (int i = start_axis; i < 4; ++i)
			count_dims *= dims[i];
		return count_dims;
	}

	void BlobData::reshape(int num, int channels, int height, int width){
		this->num = num;
		this->channels = channels;
		this->height = height;
		this->width = width;

		if (this->capacity_count < this->count()){
			if (this->list)
				delete[] this->list;

			this->list = this->count() > 0 ? new float[this->count()] : 0;
			this->capacity_count = this->count();
		}
	}

	void BlobData::copyFrom(const Blob* other){
		reshapeLike(other);
		if (other->count() > 0){
			memcpy(this->list, other->cpu_data(), this->count()*sizeof(float));
		}
	}

	void BlobData::reshapeLike(const Blob* other){
		reshape(other->num(), other->channel(), other->height(), other->width());
	}

	void BlobData::reshapeLike(const BlobData* other){
		reshape(other->num, other->channels, other->height, other->width);
	}

	void BlobData::copyFrom(const BlobData* other){
		reshapeLike(other);
		if (other->count() > 0){
			memcpy(this->list, other->list, this->count()*sizeof(float));
		}
	}

	void BlobData::release(){
		if (list){
			delete[]list;
			list = 0;
		}
	}

}