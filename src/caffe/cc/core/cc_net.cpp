
#include "caffe/cc/core/cc_v5.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <math.h>
#include <iostream>
#include "caffe/net.hpp"
#include "caffe/util/signal_handler.h"
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>

#ifdef WIN32
#include <io.h>
#endif


using namespace std;
using namespace cv;

namespace cc{

	using google::protobuf::io::FileInputStream;
	using google::protobuf::io::FileOutputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;
	using google::protobuf::io::ZeroCopyOutputStream;
	using google::protobuf::io::CodedOutputStream;
	using google::protobuf::io::IstreamInputStream;
	using google::protobuf::io::GzipInputStream;
	using google::protobuf::Message;

#define cvt(p)	((caffe::Net<float>*)p)
#define ptr		(cvt(this->_native))

	Blob* Net::blob(const char* name){
		const boost::shared_ptr<caffe::Blob<float> > blob = ptr->blob_by_name(name);
		if (blob.get() == NULL)
			return 0;

		return blob->ccBlob();
	}

	void Net::reshape(){
		ptr->Reshape();
	}

	void Net::setNative(void* native){
		this->_native = native;
	}

	void* Net::getNative(){
		return this->_native;
	}

	void Net::forward(float* loss){
		ptr->Forward(loss);
	}

	bool Net::saveToPrototxt(const char* path, bool write_weights){

		int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (fd == -1)
			return false;

		caffe::NetParameter net_param;
		ptr->ToProto(&net_param);

		if (!write_weights){
			for (int i = 0; i < net_param.layer_size(); ++i)
				net_param.mutable_layer(i)->clear_blobs();

			for (int i = 0; i < net_param.layers_size(); ++i)
				net_param.mutable_layers(i)->clear_blobs();
		}

		FileOutputStream* output = new FileOutputStream(fd);
		bool ok = google::protobuf::TextFormat::Print(net_param, output);
		delete output;
		close(fd);
		return ok;
	}

	bool Net::saveToCaffemodel(const char* path){

		caffe::NetParameter net_param;
		ptr->ToProto(&net_param);

		fstream output(path, ios::out | ios::trunc | ios::binary);
		return net_param.SerializeToOstream(&output);
	}

	size_t Net::memory_used(){
		return ptr->memory_used_;
	}

	void Net::weightsFromFile(const char* file){
		ptr->CopyTrainedLayersFrom(file);
	}

	void Net::shareTrainedLayersWith(const Net* other){
		ptr->ShareTrainedLayersWith(cvt(other->_native));
	}

	void Net::weightsFromData(const void* data, int length){
		ptr->CopyTrainedLayersFromData(data, length);
	}

	int Net::num_layers(){
		return ptr->layers().size();
	}

	Layer* Net::layer(const char* name){
		return ptr->layer_by_name(name)->ccLayer();
	}

	Layer* Net::layer(int index){
		return index < 0 || index >= num_layers() ? 0 : ptr->layers()[index]->ccLayer();
	}

	const char* Net::layer_name(int index){
		Layer* l = layer(index);
		if (l) return l->name();
		return "";
	}

	CCAPI void CCCALL releaseNet(Net* net){
		if (net){
			void* p = net->getNative();
			if (p) delete cvt(p);
		}
	}

	bool Net::has_blob(const char* name){
		return ptr->has_blob(name);
	}

	Blob* Net::blob(int index){
		if (index < 0 || index >= num_blobs())
			return 0;
		return ptr->blobs()[index]->ccBlob();
	}

	bool Net::has_layer(const char* name){
		return ptr->has_layer(name);
	}

	int Net::num_blobs(){
		return ptr->blobs().size();
	}

	int Net::input_blob_indice(int index){
		if (index < 0 || index >= ptr->input_blob_indices().size())
			return -1;

		return ptr->input_blob_indices()[index];
	}

	int Net::output_blob_indice(int index){
		if (index < 0 || index >= ptr->output_blob_indices().size())
			return -1;

		return ptr->output_blob_indices()[index];
	}

	const char* Net::input_blob_name(int index){
		int indice = input_blob_indice(index);
		return blob_name(indice);
	}

	const char* Net::output_blob_name(int index){
		int indice = output_blob_indice(index);
		return blob_name(indice);
	}

	const char* Net::blob_name(int index){
		if (index < 0 || index >= num_blobs())
			return "";

		return ptr->blob_names()[index].c_str();
	}

	int Net::num_input_blobs(){
		return ptr->num_inputs();
	}

	int Net::num_output_blobs(){
		return ptr->num_outputs();
	}

	Blob* Net::input_blob(int index){
		CHECK_GE(index, 0);
		CHECK_LT(index, num_input_blobs());
		return ptr->input_blobs()[index]->ccBlob();
	}

	Blob* Net::output_blob(int index){
		CHECK_GE(index, 0);
		CHECK_LT(index, num_output_blobs());
		return ptr->output_blobs()[index]->ccBlob();
	}

	CCAPI std::shared_ptr<Net> CCCALL loadNetFromPrototxt(const char* net_prototxt, int phase){
		caffe::Net<float>* net = new caffe::Net<float>(net_prototxt, phase == PhaseTrain ? caffe::Phase::TRAIN : caffe::Phase::TEST);
		return std::shared_ptr<Net>(net->ccNet(), releaseNet);
	}

	CCAPI std::shared_ptr<Net> CCCALL loadNetFromPrototxtString(const char* net_prototxt, int length, int phase){
		caffe::Net<float>* net = new caffe::Net<float>(net_prototxt, length < 1 ? strlen(net_prototxt) : length, phase == PhaseTrain ? caffe::Phase::TRAIN : caffe::Phase::TEST);
		return std::shared_ptr<Net>(net->ccNet(), releaseNet);
	}

	CCAPI std::shared_ptr<Net> CCCALL newNetFromParam(const caffe::NetParameter& param){
		caffe::Net<float>* net = new caffe::Net<float>(param, 0);
		return std::shared_ptr<Net>(net->ccNet(), releaseNet);
	}
}