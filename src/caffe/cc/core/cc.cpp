

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/cc/core/cc_v5.h"
#include <map>
#include <string>


#ifdef WIN32
#include <import-staticlib.h>
#include <Windows.h>
#endif

#include <thread>
#include "caffe/layers/cpp_layer.hpp"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/io/strtod.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/tokenizer.h>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#undef GetMessage

using namespace std;
using namespace cc;
using namespace cv;
using namespace google::protobuf;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::IstreamInputStream;
using google::protobuf::io::GzipInputStream;

struct LayerInfo{
	createLayerFunc creater;
	releaseLayerFunc release;
	BaseLayer* instance;
};

static map<string, LayerInfo> g_custom_layers;
static std::function<cc::OnTestClassification> g_onTestClassification;
static std::function<cc::OnOptimizationStopped> g_onOptimizationStopped;

static LayerInfo* createLayer(const char* type, void* native){
	map<string, LayerInfo>::iterator itr = g_custom_layers.find(type);
	if (itr == g_custom_layers.end()){
		LOG(FATAL) << "unknow custom layer type:" << type << ", no register.";
		return 0;
	}
	LayerInfo* layer = new LayerInfo(itr->second);
	layer->instance = layer->creater();
	layer->instance->setNative(native);
	return layer;
}

static void releaseLayer(LayerInfo* layer){
	if (layer){
		layer->release(layer->instance);
		layer->instance = 0;
		delete layer;
	}
}

cc::BaseLayer* customInstanceGetBaseLayer(CustomLayerInstance instance){
	if (instance)
		return ((LayerInfo*)instance)->instance;
	return nullptr;
}

static CustomLayerInstance CCCALL NewLayerFunction(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop, void* native){
	LayerInfo* layer = createLayer(type, native);
	layer->instance->setup(name, type, param_str, phase, bottom, numBottom, top, numTop);
	return layer;
}

static void CCCALL CustomLayerForward(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop){
	((LayerInfo*)instance)->instance->forward(bottom, numBottom, top, numTop);
}

static void CCCALL CustomLayerBackward(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down){
	((LayerInfo*)instance)->instance->backward(bottom, numBottom, top, numTop, propagate_down);
}

static void CCCALL CustomLayerReshape(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop){
	((LayerInfo*)instance)->instance->reshape(bottom, numBottom, top, numTop);
}

static void CCCALL CustomLayerRelease(CustomLayerInstance instance){
	releaseLayer((LayerInfo*)instance);
}

void onTestClassification(cc::Solver* solver, float testloss, int index, const char* itemname, float itemscore){
	if (g_onTestClassification){
		g_onTestClassification(solver, testloss, index, itemname, itemscore);
	}
}

void onOptimizationStopped(cc::Solver* solver, bool early, int iters, float smoothed_loss){
	
	if (g_onOptimizationStopped){
		g_onOptimizationStopped(solver, early, iters, smoothed_loss);
	}
}

namespace cc{

	CCScalar::CCScalar(){
		val[0] = 0;
		val[1] = 0;
		val[2] = 0;
		val[3] = 0;
	}

	CCScalar::CCScalar(double v0, double v1, double v2, double v3){
		val[0] = v0;
		val[1] = v1;
		val[2] = v2;
		val[3] = v3;
	}

	double CCScalar::operator[](int index) const{
		return val[index];
	}

	double& CCScalar::operator[](int index){
		return val[index];
	}

	CCScalar CCScalar::all(double value){
		return CCScalar(value, value, value, value);
	}

	////////////////////////////////////////////////////////////////////////////////////////
	CCAPI void CCCALL installLayer(const char* type, createLayerFunc creater, releaseLayerFunc release){
		if (g_custom_layers.find(type) != g_custom_layers.end()){
			LOG(FATAL) << "layer " << type << " already register.";
		}
		g_custom_layers[type].creater = creater;
		g_custom_layers[type].release = release;
	}

	CCAPI void CCCALL registerOnTestClassificationFunctionStdFunction(OnTestClassificationStdFunction func){
		g_onTestClassification = func;
	}

	CCAPI void CCCALL registerOnTestClassificationFunction(OnTestClassification func){
		g_onTestClassification = func;
	}

	CCAPI void CCCALL registerOnOptimizationStopped(OnOptimizationStopped func){
		g_onOptimizationStopped = func;
	}

	CCAPI void CCCALL installRegister(){
		registerLayerFunction(NewLayerFunction);
		registerLayerForwardFunction(CustomLayerForward);
		registerLayerBackwardFunction(CustomLayerBackward);
		registerLayerReshapeFunction(CustomLayerReshape);
		registerLayerReleaseFunction(CustomLayerRelease);
	}

	CCAPI bool CCCALL checkDevice(int id){
		return caffe::Caffe::CheckDevice(id);
	}

	CCAPI void CCCALL setGPU(int id){
		if (id == -1){
			caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
		}
		else{
#ifdef CPU_ONLY
			caffe::Caffe::set_mode(caffe::Caffe::Brew::CPU);
#else
			caffe::Caffe::set_mode(caffe::Caffe::Brew::GPU);
			caffe::Caffe::SetDevice(id);
#endif
		}
	}

	void* BaseLayer::getNative(){
		return this->native_;
	}

	void BaseLayer::setNative(void* ptr){
		this->native_ = ptr;
	}

	Layer* BaseLayer::ccLayer(){
		return ((caffe::CPPLayer<float>*)this->native_)->ccLayer();
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	CCAPI void CCCALL releaseBlobData(BlobData* ptr){
		if (ptr) delete ptr;
	}

	//返回GPU设备id号，失败返回-1
	CCAPI int CCCALL getDevice(){
		int device = -1;
		cudaGetDevice(&device);
		return device;
	}

	//当前是否为rootSolver
	CCAPI bool CCCALL rootSolver(){
		return caffe::Caffe::root_solver();
	}

	CCAPI std::shared_ptr<BlobData> CCCALL newBlobData(int num, int channels, int height, int width){
		BlobData* data = new BlobData();
		data->reshape(num, channels, height, width);
		return std::shared_ptr<BlobData>(data, releaseBlobData);
	}

	CCAPI std::shared_ptr<BlobData> CCCALL newBlobDataFromBlobShape(Blob* blob){
		return newBlobData(blob->num(), blob->channel(), blob->height(), blob->width());
	}

	CCAPI void CCCALL copyFromBlob(BlobData* dest, Blob* blob){
		dest->reshape(1, blob->channel(), blob->height(), blob->width());

		if (blob->count()>0)
			memcpy(dest->list, blob->cpu_data(), blob->count()*sizeof(float));
	}

	CCAPI void CCCALL copyOneFromBlob(BlobData* dest, Blob* blob, int numIndex){
		dest->reshape(1, blob->channel(), blob->height(), blob->width());

		int numSize = blob->channel()*blob->height()*blob->width();
		if (blob->count()>0)
			memcpy(dest->list, blob->cpu_data() + numIndex * numSize, numSize*sizeof(float));
	}

	CCAPI void CCCALL disableLogPrintToConsole(){
		static volatile int flag = 0;
		if (flag) return;

		flag = 1;
		google::InitGoogleLogging("cc");
	}

	CCAPI const char* CCCALL getCCVersionString(){
		return VersionStr __TIMESTAMP__;
	}

	CCAPI int CCCALL getCCVersionInt(){
		return VersionInt;
	}
	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	CCAPI int CCCALL argmax(const Blob* classification_blob, int numIndex, float* conf_ptr){
		if (conf_ptr) *conf_ptr = 0;
		if (!classification_blob) return -1;

		int planeSize = classification_blob->height() * classification_blob->width();
		return argmax(classification_blob->cpu_data() + classification_blob->offset(numIndex),
			classification_blob->channel() * planeSize, conf_ptr);
	}

	CCAPI int CCCALL argmax(const float* data_ptr, int num_data, float* conf_ptr){
		if (conf_ptr) *conf_ptr = 0;
		if (!data_ptr || num_data < 1) return -1;
		int label = static_cast<int>(std::max_element(data_ptr, data_ptr + num_data) - data_ptr); 
		if (conf_ptr) *conf_ptr = data_ptr[label];
		return label;
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////	
}