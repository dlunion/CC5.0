
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
#include "caffe/util/math_functions.hpp"

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

	bool Net::weightsFromFile(const char* file){
		return ptr->CopyTrainedLayersFrom(file);
	}

	void Net::shareTrainedLayersWith(const Net* other){
		ptr->ShareTrainedLayersWith(cvt(other->_native));
	}

	bool Net::weightsFromData(const void* data, int length){
		return ptr->CopyTrainedLayersFromData(data, length);
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

	void freeChar(char* p){
		if (p) delete[] p;
	}

	CCData Net::saveCaffemodelToData(){

		caffe::NetParameter net_param;
		ptr->ToProto(&net_param);

		string data = net_param.SerializeAsString();
		CCData out;

		if (!data.empty()){
			out.length = data.size();
			out.data.reset(new char[out.length], freeChar);
			memcpy(out.data.get(), data.data(), out.length);
		}
		return out;
	}

	CCData Net::savePrototxtToData(bool write_weights){

		caffe::NetParameter net_param;
		ptr->ToProto(&net_param);

		if (!write_weights){
			for (int i = 0; i < net_param.layer_size(); ++i)
				net_param.mutable_layer(i)->clear_blobs();

			for (int i = 0; i < net_param.layers_size(); ++i)
				net_param.mutable_layers(i)->clear_blobs();
		}

		CCData out;
		string data;
		bool ok = google::protobuf::TextFormat::PrintToString(net_param, &data);
		if (ok && !data.empty()){
			out.length = data.size();
			out.data.reset(new char[out.length], freeChar);
			memcpy(out.data.get(), data.data(), out.length);
		}
		return out;
	}

	CCAPI std::shared_ptr<Net> CCCALL loadNetFromPrototxt(const char* net_prototxt, int phase){

		if (net_prototxt == nullptr)
			return std::shared_ptr<Net>();

		caffe::Phase p = phase == PhaseTrain ? caffe::Phase::TRAIN : caffe::Phase::TEST;
		caffe::Net<float>* net = caffe::newNetFromParamPrototxtFile(net_prototxt, p);
		if (net == nullptr)
			return std::shared_ptr<Net>();

		return std::shared_ptr<Net>(net->ccNet(), releaseNet);
	}

	CCAPI std::shared_ptr<Net> CCCALL loadNetFromPrototxtString(const char* net_prototxt, int length, int phase){

		if (net_prototxt == nullptr)
			return std::shared_ptr<Net>();

		if (length == -1)
			length = strlen(net_prototxt);

		if (length < 1)
			return std::shared_ptr<Net>();

		caffe::Phase p = phase == PhaseTrain ? caffe::Phase::TRAIN : caffe::Phase::TEST;
		caffe::Net<float>* net = caffe::newNetFromParamPrototxtString(string(net_prototxt, net_prototxt + length), p);
		if (net == nullptr)
			return std::shared_ptr<Net>();

		return std::shared_ptr<Net>(net->ccNet(), releaseNet);
	}

	/*
	def _calculate_fan_in_and_fan_out(tensor):
		dimensions = tensor.dim()
		if dimensions < 2:
			raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

		if dimensions == 2:  # Linear
			fan_in = tensor.size(1)
			fan_out = tensor.size(0)
		else:
			num_input_fmaps = tensor.size(1)
			num_output_fmaps = tensor.size(0)
			receptive_field_size = 1
			if tensor.dim() > 2:
				receptive_field_size = tensor[0][0].numel()
			fan_in = num_input_fmaps * receptive_field_size
			fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out
	*/

	void _calculate_fan_in_and_fan_out(caffe::Blob<float>* tensor, int& fan_in, int& fan_out){

		if (tensor->num_axes() == 2){ //Linear
			fan_in = tensor->shape(1);
			fan_out = tensor->shape(0);
		}
		else{
			int num_input_fmaps = tensor->shape(1);
			int num_output_fmaps = tensor->shape(0);
			int receptive_field_size = 1;
			if (tensor->num_axes() > 2){
				for (int i = 2; i < tensor->num_axes(); ++i)
					receptive_field_size *= tensor->shape(i);
			}
			fan_in = num_input_fmaps * receptive_field_size;
			fan_out = num_output_fmaps * receptive_field_size;
		}
	}

	/*
	def calculate_gain(nonlinearity, param=None):
		r"""Return the recommended gain value for the given nonlinearity function.
		The values are as follows:

		================= ====================================================
		nonlinearity      gain
		================= ====================================================
		Linear / Identity :math:`1`
		Conv{1,2,3}D      :math:`1`
		Sigmoid           :math:`1`
		Tanh              :math:`\frac{5}{3}`
		ReLU              :math:`\sqrt{2}`
		Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
		================= ====================================================

		Args:
			nonlinearity: the non-linear function (`nn.functional` name)
			param: optional parameter for the non-linear function

		Examples:
			>>> gain = nn.init.calculate_gain('leaky_relu')
		"""
		linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
		if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
			return 1
		elif nonlinearity == 'tanh':
			return 5.0 / 3
		elif nonlinearity == 'relu':
			return math.sqrt(2.0)
		elif nonlinearity == 'leaky_relu':
			if param is None:
				negative_slope = 0.01
			elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
				# True/False are instances of int, hence check above
				negative_slope = param
			else:
				raise ValueError("negative_slope {} not a valid number".format(param))
			return math.sqrt(2.0 / (1 + negative_slope ** 2))
		else:
			raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
	*/

	float calculate_gain(const string& nonlinearity, float* param_ = nullptr){

		static map<string, int> linear_fns{
			{"linear", 1 }, { "conv1d", 1 }, { "conv2d", 1 }, { "conv3d", 1 }, { "conv_transpose1d", 1 }, { "conv_transpose2d", 1 }, { "conv_transpose3d", 1 }
		};
		
		if(linear_fns.find(nonlinearity) != linear_fns.end() || nonlinearity == "sigmoid"){
			return 1;
		}
		else if (nonlinearity == "tanh"){
			return 5.0 / 3;
		}
		else if (nonlinearity == "relu"){
			return sqrt(2.0f);
		}
		else if (nonlinearity == "leaky_relu"){

			float negative_slope = 0;
			if (param_ == nullptr){
				negative_slope = 0.01;
			}
			else{
				negative_slope = *param_;
			}
			return sqrt(2.0 / (1 + pow(negative_slope, 2)));
		}
		else{
			//raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
			static char error[100];
			sprintf(error, "Unsupported nonlinearity: %s", nonlinearity.c_str());
			throw error;
		}
	}

	/*
	def _calculate_correct_fan(tensor, mode):
		mode = mode.lower()
		valid_modes = ['fan_in', 'fan_out']
		if mode not in valid_modes:
			raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

		fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
		return fan_in if mode == 'fan_in' else fan_out
	*/
	int _calculate_correct_fan(caffe::Blob<float>* tensor, const string& mode = "fan_in"){

		int fan_in, fan_out;
		_calculate_fan_in_and_fan_out(tensor, fan_in, fan_out);

		if (mode == "fan_in")
			return fan_in;
		else
			return fan_out;
	}

	/*
	def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
		r"""Fills the input `Tensor` with values according to the method
		described in `Delving deep into rectifiers: Surpassing human-level
		performance on ImageNet classification` - He, K. et al. (2015), using a
		uniform distribution. The resulting tensor will have values sampled from
		:math:`\mathcal{U}(-\text{bound}, \text{bound})` where

		.. math::
			\text{bound} = \sqrt{\frac{6}{(1 + a^2) \times \text{fan\_in}}}

		Also known as He initialization.

		Args:
			tensor: an n-dimensional `torch.Tensor`
			a: the negative slope of the rectifier used after this layer (0 for ReLU
				by default)
			mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
				preserves the magnitude of the variance of the weights in the
				forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
				backwards pass.
			nonlinearity: the non-linear function (`nn.functional` name),
				recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

		Examples:
			>>> w = torch.empty(3, 5)
			>>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
		"""
		fan = _calculate_correct_fan(tensor, mode)
		gain = calculate_gain(nonlinearity, a)
		std = gain / math.sqrt(fan)
		bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
		with torch.no_grad():
			return tensor.uniform_(-bound, bound)
	*/

	void kaiming_uniform_(caffe::Blob<float>* tensor, float a = 0, const string& mode = "fan_in", const string& nonlinearity = "leaky_relu"){

		float fan = _calculate_correct_fan(tensor, mode);
		float gain = calculate_gain(nonlinearity, &a);
		float std = gain / sqrt(fan);
		float bound = sqrt(3.0) * std;
		int count = tensor->count();
		caffe::caffe_rng_uniform(count, -bound, +bound, tensor->mutable_cpu_data());
	}

	void kaiming_bias_uniform(caffe::Blob<float>* weight, caffe::Blob<float>* bias){

		/*
			fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			init.uniform_(self.bias, -bound, bound)
		*/

		int fan_in, fan_out;
		_calculate_fan_in_and_fan_out(weight, fan_in, fan_out);
		float bound = 1 / sqrt((float)fan_in);

		int count = bias->count();
		caffe::caffe_rng_uniform(count, -bound, +bound, bias->mutable_cpu_data());
	}

	CCAPI void CCCALL kaimingUniform(Net* net_){

		auto net = cvt(net_->getNative()); 
		for (int i = 0; i < net->layers_.size(); ++i){
			
			auto& layer = net->layers_[i];
			const auto& layer_name = layer->layer_param_.name();
			const auto& layer_type = layer->layer_param_.type();

			if (layer_type == "Convolution" || layer_type == "InnerProduct" || layer_type == "Deconvolution"){
				caffe::Blob<float>* weight = layer->blobs_[0].get();
				caffe::Blob<float>* bias = layer->blobs_.size() > 1 ? layer->blobs_[1].get() : nullptr;

				LOG(INFO) << "KaimingUniform layer \"" << layer_name << "\".weight";
				kaiming_uniform_(weight, sqrt(5.0f));

				if (bias){
					LOG(INFO) << "KaimingUniform layer \"" << layer_name << "\".bias";
					kaiming_bias_uniform(weight, bias);
				}
			}
			else{
				LOG(INFO) << "KaimingUniform No Change layer \"" << layer_name << "\"";
			}
		}
	}
}