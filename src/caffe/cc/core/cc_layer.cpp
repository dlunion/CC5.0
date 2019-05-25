

#include "caffe/cc/core/cc_v5.h"
#include "caffe/layer.hpp"
#include <google/protobuf/text_format.h>
#include "caffe/util/io.hpp"
#include "caffe/layer_factory.hpp"

#ifdef WIN32
#include <windows.h>
#endif

namespace cc{
	using namespace std;

#define cvt(p)	((caffe::Layer<float>*)p)
#define ptr		(cvt(this->_native))

	using google::protobuf::Message;
	void Layer::setNative(void* native){
		this->_native = native;
	}

	void Layer::setSharedPtrNative(void* native){
		this->_shared_ptr_native = native;
	}

	const char* Layer::name() const{
		return ptr->layer_param_.name().c_str();
	}

	const char* Layer::type() const{
		return ptr->type();
	}

	void Layer::setupLossWeights(int num, float* weights){
		for (int i = 0; i < num; ++i)
			ptr->set_loss(i, weights[i]);
	}

	float Layer::lossWeights(int index){
		return ptr->loss(index);
	}

	void Layer::setLossWeights(int index, float weights){
		ptr->set_loss(index, weights);
	}

	vector<caffe::Blob<float>*> toVec(const Blob** blobs, int num){
		vector<caffe::Blob<float>*> blobs_(num);
		for (int i = 0; i < num; ++i)
			blobs_[i] = ((caffe::Blob<float>*)blobs[i]->getNative());
		return blobs_;
	}

	void Layer::setup(const Blob** bottom, int numbottom, const Blob** top, int numtop){
		ptr->SetUp(toVec(bottom, numbottom), toVec(top, numtop));
	}

	void Layer::reshape(const Blob** bottom, int numbottom, const Blob** top, int numtop){
		ptr->Reshape(toVec(bottom, numbottom), toVec(top, numtop));
	}

	float Layer::forward(const Blob** bottom, int numbottom, const Blob** top, int numtop){
		return ptr->Forward(toVec(bottom, numbottom), toVec(top, numtop));
	}

	void Layer::backward(const Blob** bottom, int numbottom, const bool* propagate_down, int propagates, const Blob** top, int numtop){
		ptr->Backward(toVec(bottom, numbottom), vector<bool>(propagate_down, propagate_down + propagates), toVec(top, numtop));
	}

	int Layer::getNumBottom() const{
		return ptr->layer_param_.bottom_size();
	}

	int Layer::getNumTop() const{
		return ptr->layer_param_.top_size();
	}

	const char* Layer::bottomName(int index) const{
		if (index >= 0 && index < getNumBottom())
			return ptr->layer_param_.bottom(index).c_str();
		else
			return 0;
	}

	const char* Layer::topName(int index) const{
		if (index >= 0 && index < getNumTop())
			return ptr->layer_param_.top(index).c_str();
		else
			return 0;
	}

	Blob* Layer::paramBlob(int index) const{
		return ptr->blobs_[index]->ccBlob();
	}

	int Layer::getNumParamBlob() const{
		return ptr->blobs_.size();
	}

	void* Layer::getNative() const{
		return this->_native;
	}

	void* Layer::getSharedPtrNative() const{
		return this->_shared_ptr_native;
	}

	BaseLayer* Layer::getBaseLayerInstance() const{
		return this->_baselayer_instance;
	}

	void Layer::setBaseLayerInstance(BaseLayer* custom_layer){
		this->_baselayer_instance = custom_layer;
	}
	
	Layer::Layer(){
		this->_native = nullptr;
		this->_shared_ptr_native = nullptr;
		this->_baselayer_instance = nullptr;
	}

	Layer::~Layer(){
		this->_native = nullptr;
		this->_shared_ptr_native = nullptr;
		this->_baselayer_instance = nullptr;
	}

	CCAPI void CCCALL releaseLayer(Layer* layer){
		if (layer){
			boost::shared_ptr<caffe::Layer<float>>* super_layer = (boost::shared_ptr<caffe::Layer<float>>*)layer->getSharedPtrNative();
			if (super_layer)
				delete super_layer;
		}
	}

	CCAPI std::shared_ptr<Layer> CCCALL newLayer(const char* layer_defined, int length) {
		if (length == -1) length = strlen(layer_defined);

		caffe::LayerParameter param;
		if (!caffe::ReadProtoFromTextString(string(layer_defined, layer_defined + length), &param))
			return std::shared_ptr<Layer>();

		boost::shared_ptr<caffe::Layer<float>>* super_layer = new boost::shared_ptr<caffe::Layer<float>>(caffe::LayerRegistry<float>::CreateLayer(param));
		(*super_layer)->ccLayer()->setSharedPtrNative(super_layer);
		return std::shared_ptr<Layer>((*super_layer)->ccLayer(), releaseLayer);
	}

	namespace plugin{
#ifdef WIN32

		HMODULE loadModule(){
			HMODULE module = LoadLibraryA("libnetscope.dll");
			if (!module){
				printf("can not find plugin module: libnetscope.dll\n");
				return nullptr;
			}
			return module;
		}

		typedef HINSTANCE(*__stdcall proOpenNetscope)(const char* netName);
		typedef bool(*__stdcall proPostPrototxt)(const char* netName, const char* userName, const char* content, int length);
		CCAPI void CCCALL openNetscope(const char* netName){

			auto module = loadModule();
			if (module){
				proOpenNetscope func = (proOpenNetscope)GetProcAddress(module, "openNetscope");
				if (func){
					func(netName);
				}
				else{
					printf("can not find function openNetscope in libnetscope.dll\n");
				}
			}
		}

		CCAPI bool CCCALL postPrototxt(const char* netName, const char* userName, const char* content, int length){

			auto module = loadModule();
			if (module){
				proPostPrototxt func = (proPostPrototxt)GetProcAddress(module, "postPrototxt");
				if (func){
					return func(netName, userName, content, length);
				}
				else{
					printf("can not find function postPrototxt in libnetscope.dll\n");
				}
			}
			return false;
		}
#else
		CCAPI void CCCALL openNetscope(const char* netName){
			printf("openNetscope no impl.\n");
		}

		CCAPI bool CCCALL postPrototxt(const char* netName, const char* userName, const char* content, int length){
			printf("postPrototxt no impl.\n");
			return false;
		}
#endif
	};
};