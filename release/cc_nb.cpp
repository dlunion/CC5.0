

#ifdef WIN32
#include <Windows.h>
#endif


#include "cc_nb.h"
#include <stdarg.h>
#include <thread>
#include <mutex>
#include <map>
#include <stack>
#include <unordered_map>

using namespace std;
using namespace cc;
using namespace cv;


namespace cc{

	///////////////////////////////////////////////////////////////
	string f(const char* fmt, ...){
		va_list vl;
		va_start(vl, fmt);

		char buffer[10000];
		vsprintf(buffer, fmt, vl);
		return buffer;
	}

	struct OThreadContextSessionImpl : public OThreadContextSession{

		std::function<StepEndCallbackFunctional> step_end_callback_;
		cc::Solver* solver_ = nullptr;
		string solver_pb;
		string net_pb;
		std::thread::id thread_id_;
		stack<string> namescope_stack_;
		map<std::string, int> layers_last_name_number_map_;
		map<std::string, void*> value_store_map_;
		LayerID next_layer_id_ = 1;

		virtual void clean_auto_name_info(){
			next_layer_id_ = 1;
			value_store_map_.clear();
			layers_last_name_number_map_.clear();
		}

		virtual cc::Solver* solver(){
			return solver_;
		}

		static mutex global_lock_;
		static map<std::thread::id, std::shared_ptr<OThreadContextSessionImpl>> global_session_pool_;

		static OThreadContextSessionImpl* this_thread(){
			std::thread::id tid = std::this_thread::get_id();
			std::unique_lock<mutex> l(OThreadContextSessionImpl::global_lock_);

			OThreadContextSessionImpl* session = nullptr;
			if (OThreadContextSessionImpl::global_session_pool_.find(tid) ==
				OThreadContextSessionImpl::global_session_pool_.end()){

				session = new OThreadContextSessionImpl();
				session->thread_id_ = tid;
				OThreadContextSessionImpl::global_session_pool_[tid].reset(session);
			}
			else{
				session = OThreadContextSessionImpl::global_session_pool_[tid].get();
			}
			return session;
		}

		virtual cc::Blob* get_tensor_blob(const char* blob_name){
			if (this->solver_){
				if (this->solver_->net()){
					return this->solver_->net()->blob(blob_name);
				}
			}
			return nullptr;
		}

		//获取存储在session中的值
		virtual void* get(const char* key){
			auto itr = value_store_map_.find(key);
			if (itr == value_store_map_.end())
				return nullptr;
			return itr->second;
		}

		virtual void put(const char* key, void* value){
			value_store_map_[key] = value;
		}

		virtual LayerID next_layer_id(){
			return next_layer_id_++;
		}
	};

	mutex OThreadContextSessionImpl::global_lock_;
	map<std::thread::id, std::shared_ptr<OThreadContextSessionImpl>> OThreadContextSessionImpl::global_session_pool_;

	OThreadContextSession* OThreadContextSession::this_thread(){
		return OThreadContextSessionImpl::this_thread();
	}

	void* OThreadContextSession::this_thread_get(const char* key){
		return OThreadContextSessionImpl::this_thread()->get(key);;
	}

	void OThreadContextSession::this_thread_put(const char* key, void* value){
		OThreadContextSessionImpl::this_thread()->put(key, value);
	}

	cc::Solver* OThreadContextSession::this_thread_solver(){
		return OThreadContextSessionImpl::this_thread()->solver();
	}

	void OThreadContextSession::this_thread_clean_auto_name_info(){
		OThreadContextSessionImpl::this_thread()->clean_auto_name_info();
	}

	//
	//    计算标准的DNN相关输出shape
	//
	//int compute_std_dnn_output_shape(int input_dim, int kernel_dim, int strides, int padding, int dilation){
	//	int kernel_extent = dilation * (kernel_dim - 1) + 1;
	//	return (input_dim + 2 * padding - kernel_extent) / strides + 1;
	//}

	//
	//    获取名称，基于当前上下文中的scope指定名称
	//    返回名称以： scope / name 的形式，若scope为空，则返回name
	//
	string get_name_with_scope(const string& name){
		if (OThreadContextSessionImpl::this_thread()->namescope_stack_.empty())
			return name;

		return OThreadContextSessionImpl::this_thread()->namescope_stack_.top() + "/" + name;
	}

	//
	//    scope的具体实现定义，构造的时候push，析构的时候pop
	//
	name_scope::name_scope(const string& name){
		OThreadContextSessionImpl::this_thread()->namescope_stack_.push(get_name_with_scope(name));
	}
	name_scope::~name_scope(){
		OThreadContextSessionImpl::this_thread()->namescope_stack_.pop();
	}

	string Initializer::seril(){
		string result;

		result += f("type: \"%s\"\n", type.c_str());
		if (type == "constant")
			result += f("value: %g\n", value);
		else
			if (value != 0) result += f("value: %g\n", value);

		if (minval != 0) result += f("min: %g\n", minval);
		if (maxval != 1) result += f("max: %g\n", maxval);
		if (meanval != 0) result += f("mean: %g\n", meanval);
		if (stdval != 1) result += f("std: %g\n", stdval);
		if (sparse != -1) result += f("sparse: %d\n", sparse);
		if (variance_norm != VarianceNorm_FAN_IN) result += f("value: %g\n", variance_norm_string(variance_norm));
		return result;
	}

	Tensor OTensor::getTensorFromName(
		const vector<Tensor>& graph, const string& name){

		unordered_map<OTensor*, bool> path;
		stack<shared_ptr<OTensor>> tensors;

		for (int i = 0; i < graph.size(); ++i)
			tensors.push(graph[i]);

		while (!tensors.empty()){

			shared_ptr<OTensor> t = tensors.top();
			tensors.pop();

			if (path.find(t.get()) != path.end())
				continue;

			if (t->name.compare(name) == 0)
				return t;

			path[t.get()] = true;
			for (int i = 0; i < t->owner->output.size(); ++i)
				tensors.push(t->owner->output[i]);
			
			for (int i = 0; i < t->owner->input.size(); ++i)
				tensors.push(t->owner->input[i]);
		}
		return shared_ptr<OTensor>();
	}

	//string OTensor::shapestr(){
	//	string r;
	//	char buf[100];
	//
	//	for (int i = 0; i < shape.size(); ++i){
	//		sprintf(buf, "%d", shape[i]);
	//
	//		if (i == 0)
	//			r = buf;
	//		else
	//			r = r + "," + buf;
	//	}
	//	return r;
	//}

	string operator+(const char* s1, const string& s2){
		return string(s1) + s2;
	}

	//
	//    序列化参数
	//
	string OOptimizer::seril(){

		string result;
		if (iter_size != 1) result += f("iter_size: %d\n", iter_size);
		if (!test_iter.empty()){
			for (size_t i = 0; i < test_iter.intarraySize(); ++i)
				result += f("test_iter: %d\n", test_iter.intval(i));
		}
		if (test_interval != 0) result += f("test_interval: %d\n", test_interval);
		if (!test_initialization) result += f("test_initialization: %s\n", bool_string(test_initialization));
		if (!base_lr.empty()) result += f("base_lr: %g\n", base_lr.doubleval());
		if (!display.empty()) result += f("display: %d\n", display.intval());
		if (average_loss != 1) result += f("average_loss: %d\n", average_loss);
		if (!max_iter.empty()) result += f("max_iter: %d\n", max_iter.intval());
		if (!lr_policy.empty()) result += f("lr_policy: \"%s\"\n", lr_policy.c_str());
		if (random_seed != -1) result += f("random_seed: %d\n", random_seed);

		if (!gamma.empty()) result += f("gamma: %g\n", gamma.floatval());
		if (!power.empty()) result += f("power: %g\n", power.floatval());
		if (!weight_decay.empty()) result += f("weight_decay: %g\n", weight_decay.floatval());
		if (!stepsize.empty()) result += f("stepsize: %d\n", stepsize.intval());
		for (size_t i = 0; i < stepvalue.size(); ++i)
			result += f("stepvalue: %d\n", stepvalue[i]);

		if (!regularization_type.empty()) result += f("regularization_type: \"%s\"\n", regularization_type.c_str());

		if (snapshot != 0) result += f("snapshot: %d\n", snapshot);
		if (!snapshot_prefix.empty()) result += f("snapshot_prefix: \"%s\"\n", snapshot_prefix.strval().c_str());
		if (snapshot_diff) result += f("snapshot_diff: %s\n", bool_string(snapshot_diff));
		result += f("solver_mode: %s\n", solver_mode_string(solver_mode));
		//if (!device_ids.empty()) result += f("device_id: %d\n", device_ids[0]);
		return result + seril_sub_param();
	}
	
	OLayerOp::OLayerOp(){
		this->layer_id = OThreadContextSession::this_thread()->next_layer_id();
	}

	string OLayerOp::scope_name_or_next_auto_name(const string& name){
		string uname = name;
		if (uname.empty()){
			string caffetypename = caffe_type_name();
			if (caffetypename == "CPP")
				caffetypename = ((cc::layers::OCustom*)this)->cpp_type;

			map<std::string, int>& layer_last_name_number_map = OThreadContextSessionImpl::this_thread()->layers_last_name_number_map_;
			uname = f("%s%d", caffetypename.c_str(), ++layer_last_name_number_map[caffetypename.c_str()]);
		}
		return get_name_with_scope(uname);
	}

	string OLayerOp::serial(){

		string param = serial_param();
		string result = "layer{\n";
		result += f("name: \"%s\"\n", name.c_str());
		result += f("type: \"%s\"\n", caffe_type_name());

		for (int i = 0; i < input.size(); ++i)
			result += f("bottom: \"%s\"\n", input[i]->name.c_str());

		for (int i = 0; i < output.size(); ++i)
			result += f("top: \"%s\"\n", output[i]->name.c_str());

		if (phase){
			result += f(
				"include {\n"
				"phase: %s\n"
				"}\n", phase_string(*phase.get()));
		}

		for (size_t i = 0; i < loss_weight.floatarraySize(); ++i)
			result += f("loss_weight: %g\n", loss_weight.floatval(i));

		for (size_t i = 0; i < propagate_down.size(); ++i)
			result += f("propagate_down: %d\n", propagate_down[i]);

		if (kernel_mult)
			result += kernel_mult->seril() + "\n";

		if (bias_mult)
			result += bias_mult->seril() + "\n";

		if (!param.empty())
			result = result + param + "\n";

		result += "}";
		return result;
	}

	//
	//    指定要优化的对象，图
	//
	void OOptimizer::minimize(const vector<Tensor>& graphs){
		graph_type = GraphType_FromTensor;
		this->graphs = graphs;
	}

	//
	//    指定要优化的对象，图
	//
	void OOptimizer::minimizeFromPrototxt(const string& graphstr){
		graph_type = GraphType_FromPrototxt;
		this->str_graph = graphstr;
	}

	//
	//    指定要优化的对象，图
	//
	void OOptimizer::minimizeFromFile(const string& graphfile){
		graph_type = GraphType_FromFile;
		this->file_graph = graphfile;
	}

	namespace engine{
		namespace caffe{

			GraphInput::GraphInput(const std::vector<Tensor>& input){
				this->graphs = input;
			}

			GraphInput::GraphInput(const Tensor& input){
				this->graphs = { input };
			}

			GraphInput::GraphInput(const std::initializer_list<Tensor>& input){
				this->graphs = input;
			}

			GraphInput::GraphInput(){
			}

			void serial_layer(OLayerOp* layer, vector<OLayerOp*>& layer_order,
				map<OLayerOp*, bool>& layer_state, map<string, OLayerOp*>& output_blob_layer_map){

				if (layer_state[layer])
					return;

				for (int i = 0; i < layer->input.size(); ++i){
					Tensor tensor = layer->input[i];
					string name = tensor->owner->name + "#" + tensor->name;
					OLayerOp* l = output_blob_layer_map[name.c_str()];
					serial_layer(l, layer_order, layer_state, output_blob_layer_map);
				}

				if (!layer_state[layer]){
					layer_state[layer] = true;
					layer_order.push_back(layer);
				}
			};

			//
			//    将计算图编译到caffe支持
			//
			bool buildGraphToFile(const GraphInput& graph, const string& file){
				FILE* f = fopen(file.c_str(), "wb");
				if (!f) return false;

				string pb = buildGraph(graph);
				fwrite(pb.data(), 1, pb.size(), f);
				fclose(f);
				return true;
			}

			//
			//    将计算图编译到caffe支持
			//
			string buildGraph(const GraphInput& graph){

				list<OLayerOp*> stack_;

				//0 is root layer
				for (size_t i = 0; i < graph.graphs.size(); ++i)
					stack_.push_back(graph.graphs[i]->owner.get());

				//bool is serialed
				map<OLayerOp*, bool> all_layer;
				map<string, OLayerOp*> output_blob_layer_map;
				map<string, int> blob_name_map;
				while (!stack_.empty()){

					OLayerOp* layer = stack_.front();
					stack_.pop_front();

					//对每个输入的blob，记录其owner，然后遍历
					//如果该layer已经处理过，则不再对其input做查找，否则d将在递归模块时造成死循环
					if (all_layer.find(layer) != all_layer.end())
						continue;

					all_layer[layer] = false;
					for (int i = 0; i < layer->output.size(); ++i){

						string blobname = layer->name + "#" + layer->output[i]->name;
						output_blob_layer_map[blobname.c_str()] = layer;
						blob_name_map[blobname]++;

						if (blob_name_map[blobname] > 1){
							printf("multi source with layer[%s], blob [%s]\n", layer->name.c_str(), layer->output[i]->name.c_str());
							return "";
						}
					}

					for (int i = 0; i < layer->input.size(); ++i){
						if (layer->input[i]->owner)
							stack_.push_back(layer->input[i]->owner.get());
					}
				}

				vector<OLayerOp*> serial_order;
				for (int i = 0; i < graph.graphs.size(); ++i){
					serial_layer(graph.graphs[i]->owner.get(), serial_order, all_layer, output_blob_layer_map);
				}

				string net_output;
				int space = 0;
				for (int j = 0; j < serial_order.size(); ++j){
					OLayerOp* l = serial_order[j];
					string layer_string = l->serial();
					char* token = strtok((char*)layer_string.c_str(), "\n");

					while (token){
						if (strchr(token, '}'))
							space--;

						for (int k = 0; k < space; ++k)
							net_output += "    ";

						if (strchr(token, '{'))
							space++;

						net_output += f("%s\n", token);
						token = strtok(nullptr, "\n");
					}
				}
				return net_output;
			}


			//
			//    编译一个net，然后可以做inference
			//
			std::shared_ptr<Net> buildNet(const GraphInput& graph, int phase){
				 
				string net_pb = buildGraph(graph);
				return loadNetFromPrototxtString(net_pb.c_str(), net_pb.length(), phase);
			}
		}
	};
	
	//
	//    引擎部分
	//
	namespace train{

		namespace caffe{

			//
			//    train回调函数的转发函数
			//
			void trainStepEndCallbackFunc(cc::Solver* solver, int step, float smoothed_loss, void* userData){
				OThreadContextSessionImpl* session = (OThreadContextSessionImpl*)userData;
				if (session->step_end_callback_){
					session->step_end_callback_(session, step, smoothed_loss);
				}
			}

			//
			//    从文件读取数据
			//
			string readfromfile(const string& file){
				FILE* f = fopen(file.c_str(), "rb");
				if (!f){
					printf("read fail: %s\n", file.c_str());
					return "";
				}
				string out;
				int len = 0;
				fseek(f, 0, SEEK_END);
				len = ftell(f);
				fseek(f, 0, SEEK_SET);
				if (len > 0){
					out.resize(len);
					fread((char*)out.data(), 1, len, f);
				}
				fclose(f);
				return out;
			}

			//
			//    训练任务执行
			//
			void run(const Optimizer& optimizer, const std::function<StepEndCallbackFunctional>& stepEndCallback, const std::function<PreTrainCallbackFunctional>& preTrainCallback){

				string net_pb;
				switch (optimizer->graph_type){
				case GraphType_FromTensor:
					net_pb = engine::caffe::buildGraph(optimizer->graphs);
					break;

				case GraphType_FromFile:
					net_pb = readfromfile(optimizer->file_graph);
					break;

				case GraphType_FromPrototxt:
					net_pb = optimizer->str_graph;
					break;

				case GraphType_None:
					printf("no set graph_type for optimizer.\n");
					return;
				}

				string solver_pb = optimizer->seril();
				if (optimizer->device_ids.size() < 2){
					if (optimizer->device_ids.empty()){

						//set CPU
						printf("train use CPU.\n");
						setGPU(-1);
					}
					else{
						printf("train use GPU: %d\n", optimizer->device_ids[0]);
						setGPU(optimizer->device_ids[0]);
					}
				}
				std::shared_ptr<cc::Solver> solver = cc::loadSolverFromPrototxtString(solver_pb.c_str(), net_pb.c_str());

				if (!optimizer->reload_weights.empty()){
					printf("load weights from: %s\n", optimizer->reload_weights.c_str());
					solver->net()->weightsFromFile(optimizer->reload_weights.c_str());
				}

				OThreadContextSessionImpl* session = OThreadContextSessionImpl::this_thread();
				session->net_pb = net_pb;
				session->solver_pb = solver_pb;
				session->solver_ = solver.get();
				session->step_end_callback_ = stepEndCallback;

				//if we have a valid callback function
				if (stepEndCallback)
					solver->setSetpEndCallback(trainStepEndCallbackFunc, session);
				
				if (preTrainCallback)
					preTrainCallback(session);

				if (!optimizer->device_ids.empty()){
					solver->solve(optimizer->device_ids.size(), optimizer->device_ids.data());
				}
				else{
					solver->solve();
				}

				OThreadContextSessionImpl::this_thread()->solver_ = nullptr;
			}
		};
	};

	namespace layers{

		string OSplit::serial_param(){
			string part;
			if (axis != 1) part = f("axis: %d\n", axis);
			for (int i = 0; i < slice_point.size(); ++i)
				part += f("slice_point: %d\n", slice_point[i]);

			if (part.empty()) return "";
			return "slice_param {\n" + part + "}";
		}

		string OConcat::serial_param(){
			string part;
			if (axis != 1) part = f("axis: %d\n", axis);
			if (part.empty()) return "";
			return "concat_param {\n" + part + "}";
		}

		string OTranspose::serial_param(){
			string part;
			part = f("axis: %d\n", axis);

			for (int i = 0; i < order.size(); ++i)
				part += f("order: %d\n", order[i]);
			
			if (order.empty()) return "";
			return "permute_param {\n" + part + "}";
		}

		string OCrop::serial_param(){
			string part;

			part = f("axis: %d\n", axis);
			for (int i = 0; i < offset.size(); ++i)
				part += f("offset: %d\n", offset[i]);

			if (part.empty()) return "";
			return "crop_param {\n" + part + "}";
		}

		string OReshape::serial_param(){
			string part = "";
			if (axis != 0) part += f("axis: %d\n", axis);

			if (!new_dims.empty()){
				part += "shape {\n";
				for (int i = 0; i < new_dims.size(); ++i)
					part += f("dim: %d\n", new_dims[i]);
				part += "}\n";
			}
			return "reshape_param {\n" + part + "}";
		}

		string OAdd::serial_param(){
			string part = "operation: SUM\n";
			if (!stable_prod_grad) part += f("stable_prod_grad: %s\n", bool_string(stable_prod_grad));

			for (int i = 0; i < coeff.size(); ++i)
				part += f("coeff: %g\n", coeff[i]);
			return "eltwise_param {\n" + part + "}";
		}

		string OProduct::serial_param(){
			string part = "operation: PROD\n";
			if (!stable_prod_grad) part += f("stable_prod_grad: %s\n", bool_string(stable_prod_grad));
			return "eltwise_param {\n" + part + "}";
		}

		string OSigmoid::serial_param(){
			return "";
		}

		string OSoftmax::serial_param(){
			string part;
			if (axis != 1) part += f("axis: %d\n", axis);
			if (hard_ratio != 1.0f) part += f("hard_ratio: %g\n", hard_ratio);
			if (!hard_mining_label.empty()) part += f("hard_mining_label: %d\n", hard_mining_label.intval());
			if (!cutting_point.empty()) part += f("cutting_point: %g\n", cutting_point.floatval());
			if (normalize_type != "Softmax") part += f("normalize_type: %g\n", normalize_type.c_str());
			for (size_t i = 0; i < class_weight.floatarraySize(); ++i)
				part += f("class_weight: %g\n", class_weight.floatval(i));
			if (part.empty()) return "";
			return "softmax_param {\n" + part + "\n}";
		}

		string OMax::serial_param(){
			string part = "operation: MAX\n";
			return "eltwise_param {\n" + part + "}";
		}

		string ODropout::serial_param(){
			string part = f("dropout_ratio: %g\n", 1 - keep_prob);
			return "dropout_param {\n" + part + "}";
		}

		string OScale::serial_param(){

			string part;
			part = f("bias_term: %s\n", bool_string(bias_term));
			if (axis != 1) part += f("axis: %d\n", axis);
			if (num_axes != 1) part += f("num_axes: %d\n", num_axes);
			if (bias_filler) part += bias_filler->seril();
			if (part.empty()) return "";

			return "scale_param {\n" + part + "}";
		}

		string OBatchNorm::serial_param(){

			string part;
			//part += f("use_global_stats: %s\n", bool_string(true));
			if (moving_average_fraction != 0.999f) part += f("moving_average_fraction: %g\n", moving_average_fraction);
			if (eps != 1e-5f) part += f("eps: %g\n", eps);
			if (part.empty()) return "";

			return "batch_norm_param {\n" + part + "}";
		}

		string OPooling2D::serial_param(){

			string result = f(
				"pooling_param {\n"
				"pool: %s\n",
				pool_method_string(method));

			if (global_pooling){
				result += f("global_pooling: %s\n", bool_string(global_pooling));
			}
			else{
				//卷积核的定义
				if (kernel[0] != kernel[1]){
					result += f(
						"kernel_h: %d\n"
						"kernel_w: %d\n"
						, kernel[0], kernel[1]);
				}
				else{
					result += f("kernel_size: %d\n", kernel[0]);
				}
			}

			if (padding_size[0] != padding_size[1]){
				result += f(
					"pad_w: %d\n"
					"pad_h: %d\n"
					, padding_size[0], padding_size[1]);
			}
			else{
				if (padding_size[0] != 0)
					result += f("pad: %d\n", padding_size[0]);
			}

			if (strides[0] != strides[1]){
				result += f(
					"stride: %d\n"
					"stride: %d\n"
					, strides[0], strides[1]);
			}
			else{
				if (strides[0] != 1)
					result += f("stride: %d\n", strides[0]);
			}
			result += "}";
			return result;
		}

		string ODense::serial_param(){
			string part = f("num_output: %d\n", units);
			if (!bias_term) part += f("bias_term: %s\n", bool_string(bias_term));

			if (weight_initializer)
				part = part + "weight_filler {\n" + weight_initializer->seril() + "\n}\n";

			if (bias_initializer)
				part = part + "bias_filler {\n" + bias_initializer->seril() + "\n}\n";

			if (axis != -1) part += f("axis: %d\n", axis);
			if (transpose) part += f("transpose: %s\n", bool_string(transpose));
			return "inner_product_param {\n" + part + "}";
		}

		string OROIPooling::serial_param(){
			string part;
			part = part + f("pooled_w: %d\n", pooled_w);
			part = part + f("pooled_h: %d\n", pooled_h);
			part = part + f("spatial_scale: %f\n", spatial_scale);
			return "roi_pooling_param {\n" + part + "}";
		}

		string OConv2D::serial_param(){

			string result = f(
				"convolution_param {\n"
				"num_output: %d\n",
				kernel[2]);
			if (!bias_term) result += f("bias_term: %s\n", bool_string(bias_term));

			//卷积核的定义
			if (kernel[0] != kernel[1]){
				result += f(
					"kernel_size: %d\n"
					"kernel_size: %d\n"
					, kernel[0], kernel[1]);
			}
			else{
				result += f("kernel_size: %d\n", kernel[0]);
			}

			if (padding_size[0] != padding_size[1]){
				result += f(
					"pad: %d\n"
					"pad: %d\n"
					, padding_size[0], padding_size[1]);
			}
			else{
				if (padding_size[0] != 0)
					result += f("pad: %d\n", padding_size[0]);
			}

			if (strides[0] != strides[1]){
				result += f(
					"stride: %d\n"
					"stride: %d\n"
					, strides[0], strides[1]);
			}
			else{
				if (strides[0] != 1)
					result += f("stride: %d\n", strides[0]);
			}

			if (dilations[0] != dilations[1]){
				result += f(
					"dilation: %d\n"
					"dilation: %d\n"
					, dilations[0], dilations[1]);
			}
			else{
				if (dilations[0] != 1)
					result += f("dilation: %d\n", dilations[0]);
			}

			if (kernel_initializer){
				result += f("weight_filler {\n%s}\n",
					kernel_initializer->seril().c_str());
			}

			if (bias_initializer){
				result += f("bias_filler {\n%s}\n",
					bias_initializer->seril().c_str());
			}

			result += "}";
			return result;
		}

		string OIm2Col::serial_param(){

			string result = f(
				"convolution_param {\n"
				"num_output: %d\n",
				kernel[2]);

			//卷积核的定义
			if (kernel[0] != kernel[1]){
				result += f(
					"kernel_size: %d\n"
					"kernel_size: %d\n"
					, kernel[0], kernel[1]);
			}
			else{
				result += f("kernel_size: %d\n", kernel[0]);
			}

			if (padding_size[0] != padding_size[1]){
				result += f(
					"pad: %d\n"
					"pad: %d\n"
					, padding_size[0], padding_size[1]);
			}
			else{
				if (padding_size[0] != 0)
					result += f("pad: %d\n", padding_size[0]);
			}

			if (strides[0] != strides[1]){
				result += f(
					"stride: %d\n"
					"stride: %d\n"
					, strides[0], strides[1]);
			}
			else{
				if (strides[0] != 1)
					result += f("stride: %d\n", strides[0]);
			}

			if (dilations[0] != dilations[1]){
				result += f(
					"dilation: %d\n"
					"dilation: %d\n"
					, dilations[0], dilations[1]);
			}
			else{
				if (dilations[0] != 1)
					result += f("dilation: %d\n", dilations[0]);
			}
			result += "}";
			return result;
		}

		string ODeconv2D::serial_param(){

			string result = f(
				"convolution_param {\n"
				"num_output: %d\n",
				kernel[2]);
			if (!bias_term) result += f("bias_term: %s\n", bool_string(bias_term));

			//卷积核的定义
			if (kernel[0] != kernel[1]){
				result += f(
					"kernel_size: %d\n"
					"kernel_size: %d\n"
					, kernel[0], kernel[1]);
			}
			else{
				result += f("kernel_size: %d\n", kernel[0]);
			}

			if (padding_size[0] != padding_size[1]){
				result += f(
					"pad: %d\n"
					"pad: %d\n"
					, padding_size[0], padding_size[1]);
			}
			else{
				if (padding_size[0] != 0)
					result += f("pad: %d\n", padding_size[0]);
			}

			if (strides[0] != strides[1]){
				result += f(
					"stride: %d\n"
					"stride: %d\n"
					, strides[0], strides[1]);
			}
			else{
				if (strides[0] != 1)
					result += f("stride: %d\n", strides[0]);
			}

			if (dilations[0] != dilations[1]){
				result += f(
					"dilation: %d\n"
					"dilation: %d\n"
					, dilations[0], dilations[1]);
			}
			else{
				if (dilations[0] != 1)
					result += f("dilation: %d\n", dilations[0]);
			}

			if (kernel_initializer){
				result += f("weight_filler {\n%s}\n",
					kernel_initializer->seril().c_str());
			}

			if (bias_initializer){
				result += f("bias_filler {\n%s}\n",
					bias_initializer->seril().c_str());
			}

			result += "}";
			return result;
		}

		///////////////////////////////////////////////////////////////////////////////

		//
		//    数据输入层的定义
		//
		string OInput::serial_param(){
			string part;
			for (int i = 0; i < dims.size(); ++i)
				part += f("dim: %d\n", dims[i]);
			return "input_param {\nshape {\n" + part + "}\n}";
		}

		Tensor input(const vector<int>& dims, const string& name){
			OInput* pinput = new OInput();
			pinput->dims = dims;

			LayerOp layer(pinput);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->output[0] = blob;
			return layer->output[0];
		}

		//
		//    数据层的定义
		//

		string OCustom::serial_param(){
			string result = "cpp_param {\n";
			if (!cpp_param_str.empty()) result += f("param_str: \"%s\"\n", cpp_param_str.c_str());
			result += f("type: \"%s\"\n", cpp_type.c_str());
			result += "}";
			return result;
		}

		vector<Tensor> data(const string& cpp_type, const vector<string>& output, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(output.size());

			for (int i = 0; i < output.size(); ++i){

				Tensor blob(new OTensor());
				blob->name = output[i];
				blob->owner = layer;
				layer->output[i] = blob;
			}
			return layer->output;
		}

		//
		//    自定义层1
		//
		vector<Tensor> custom(const string& cpp_type, const vector<Tensor>& input, const vector<string>& output, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input = input;
			layer->output.resize(output.size());

			for (int i = 0; i < output.size(); ++i){

				Tensor blob(new OTensor());
				blob->name = layer->name + "/" + output[i];
				blob->owner = layer;
				layer->output[i] = blob;
			}
			return layer->output;
		}

		//
		//    自定义层2
		//
		Tensor custom(const string& cpp_type, const vector<Tensor>& input, const string& output, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input = input;
			layer->output.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name + "/" + output;
			blob->owner = layer;
			layer->output[0] = blob;
			return blob;
		}

		//
		//    自定义层3
		//
		Tensor custom(const string& cpp_type, const Tensor& input, const string& name, const string& param_str){

			OCustom* pdata = new OCustom();
			pdata->cpp_type = cpp_type;
			pdata->cpp_param_str = param_str;

			LayerOp layer(pdata);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input = { input };
			layer->output.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->output[0] = blob;
			return blob;
		}

		//
		//    Im2Col的定义
		//    x:        tensor
		//              需要卷积的tensor
		//
		//    kernel:   3-d array
		//              卷积核的大小，这里是2维，指定为height, width, output
		//
		//    padding:    "valid"or "same"
		//              指定padding的实现方式，valid即卷积后尺寸，无padding，same即卷积后尺寸和x一致
		//
		//    strides:  2-d array, height, width
		//              指定步长
		//
		//    dilations: 2-d array, height, width
		//              卷积的膨胀尺寸
		//
		//    name:     指定卷积层名称
		//              默认是为空，即自动生成的名称
		//
		Tensor im2col(const Tensor&  x, const vector<int>& kernel, const string& padding,
			const vector<int>& strides, const vector<int>& dilations, const string& name){

			OIm2Col* conv = new OIm2Col();
			conv->kernel = kernel;
			conv->padding = padding;
			conv->strides = strides;
			conv->padding_size.resize(2);
			conv->dilations = dilations;

			LayerOp layer(conv);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			//shape:  n, c, h, w
			//kernel: h, w, output
			if (padding == "valid"){
				conv->padding_size[0] = 0;
				conv->padding_size[1] = 0;
			}
			else if (padding == "same"){
				conv->padding_size[0] = (dilations[0] * (kernel[0] - 1) + 1) / 2;
				conv->padding_size[1] = (dilations[1] * (kernel[1] - 1) + 1) / 2;
			}

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		//
		//    卷积层的定义
		//    x:        tensor
		//              需要卷积的tensor
		//
		//    kernel:   3-d array
		//              卷积核的大小，这里是2维，指定为height, width, output
		//
		//    padding:    "valid"or "same"
		//              指定padding的实现方式，valid即卷积后尺寸，无padding，same即卷积后尺寸和x一致
		//
		//    strides:  2-d array, height, width
		//              指定步长
		//
		//    dilations: 2-d array, height, width
		//              卷积的膨胀尺寸
		//
		//    name:     指定卷积层名称
		//              默认是为空，即自动生成的名称
		//
		Tensor conv2d(const Tensor&  x, const vector<int>& kernel, const string& padding,
			const vector<int>& strides, const vector<int>& dilations, const string& name){

			OConv2D* conv = new OConv2D();
			conv->kernel = kernel;
			conv->padding = padding;
			conv->strides = strides;
			conv->padding_size.resize(2);
			conv->dilations = dilations;


			//我们一般默认卷积的权重初始化方式会是gaussian
			conv->kernel_initializer.reset(new Initializer());
			conv->bias_initializer.reset(new Initializer());

			conv->kernel_initializer->type = "gaussian";
			conv->kernel_initializer->stdval = 0.01;
			conv->bias_initializer->type = "constant";
			conv->bias_initializer->value = 0;
		

			LayerOp layer(conv);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			layer->kernel_mult.reset(new ParamSpecMult());
			layer->bias_mult.reset(new ParamSpecMult());

			layer->kernel_mult->decay_mult = 0;
			layer->kernel_mult->lr_mult = 1;

			layer->bias_mult->decay_mult = 0;
			layer->bias_mult->lr_mult = 2;

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			//shape:  n, c, h, w
			//kernel: h, w, output

			if (padding == "valid"){
				conv->padding_size[0] = 0;
				conv->padding_size[1] = 0;
			}
			else if (padding == "same"){
				conv->padding_size[0] = (dilations[0] * (kernel[0] - 1) + 1) / 2;
				conv->padding_size[1] = (dilations[1] * (kernel[1] - 1) + 1) / 2;
			}

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		//
		//    反卷积层的定义
		//    x:        tensor
		//              需要卷积的tensor
		//
		//    kernel:   3-d array
		//              卷积核的大小，这里是2维，指定为height, width, output
		//
		//    padding:    "valid"or "same"
		//              指定padding的实现方式，valid即卷积后尺寸，无padding，same即卷积后尺寸和x一致
		//
		//    strides:  2-d array, height, width
		//              指定步长
		//
		//    dilations: 2-d array, height, width
		//              卷积的膨胀尺寸
		//
		//    name:     指定卷积层名称
		//              默认是为空，即自动生成的名称
		//
		Tensor deconv2d(const Tensor&  x, const vector<int>& kernel, const string& padding,
			const vector<int>& strides, const vector<int>& dilations, const string& name){

			ODeconv2D* conv = new ODeconv2D();
			conv->kernel = kernel;
			conv->padding = padding;
			conv->strides = strides;
			conv->padding_size.resize(2);
			conv->dilations = dilations;

			//不能够内部分配，否则出错
			//我们一般默认卷积的权重初始化方式会是xavier
			conv->kernel_initializer.reset(new Initializer());
			conv->bias_initializer.reset(new Initializer());

			conv->kernel_initializer->type = "gaussian";
			conv->kernel_initializer->stdval = 0.01;
			conv->bias_initializer->type = "constant";
			conv->bias_initializer->value = 0;

			LayerOp layer(conv);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			layer->kernel_mult.reset(new ParamSpecMult());
			layer->bias_mult.reset(new ParamSpecMult());

			layer->kernel_mult->decay_mult = 0;
			layer->kernel_mult->lr_mult = 1;

			layer->bias_mult->decay_mult = 0;
			layer->bias_mult->lr_mult = 2;

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			//shape:  n, c, h, w
			//kernel: h, w, output

			if (padding == "valid"){
				conv->padding_size[0] = 0;
				conv->padding_size[1] = 0;
			}
			else if (padding == "same"){
				conv->padding_size[0] = (dilations[0] * (kernel[0] - 1) + 1) / 2;
				conv->padding_size[1] = (dilations[1] * (kernel[1] - 1) + 1) / 2;
			}

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor transpose(const Tensor&  x, vector<int> order, const string& name, bool inplace){

			OTranspose* r = new OTranspose();
			r->order = order;

			LayerOp layer(r);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor reshape(const Tensor&  x, vector<int> new_dims, const string& name){

			OReshape* r = new OReshape();
			r->new_dims = new_dims;

			LayerOp layer(r);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor concat(const vector<Tensor>& tensors, int axis, const string& name){

			OConcat* c = new OConcat();
			c->axis = axis;

			LayerOp layer(c);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(tensors.size());

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			layer->input = tensors;
			layer->output[0] = blob;
			return blob;
		}

		Tensor max_pooling2d(const Tensor&  x, const vector<int>& kernel, const vector<int>& strides, const vector<int>& padding, bool global_pooling, const string& name){
			OPooling2D* pool = new OPooling2D();
			pool->kernel = kernel;
			pool->strides = strides;
			pool->method = PoolMethod_MAX;
			pool->global_pooling = global_pooling;
			pool->padding_size = padding;

			LayerOp layer(pool);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor avg_pooling2d(const Tensor&  x, const vector<int>& kernel, const vector<int>& strides, const vector<int>& padding, bool global_pooling, const string& name){

			OPooling2D* pool = new OPooling2D();
			pool->kernel = kernel;
			pool->strides = strides;
			pool->method = PoolMethod_AVE;
			pool->global_pooling = global_pooling;
			pool->padding_size = padding;

			if (global_pooling){
				pool->kernel = { 0, 0 };
				pool->strides = { 1, 1 };
				pool->padding_size = { 0, 0 };
			}

			LayerOp layer(pool);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor dense(const Tensor&  x, int units, const string& name, bool bias_term){

			ODense* d = new ODense();
			d->units = units;
			d->bias_term = bias_term;

			//不能内部构造，否则外边没法修改
			//我们一般默认卷积的权重初始化方式会是xavier
			d->weight_initializer.reset(new Initializer());
			d->bias_initializer.reset(new Initializer());

			d->weight_initializer->stdval = 0.01;
			d->weight_initializer->type = "gaussian";
			d->bias_initializer->type = "constant";
			d->bias_initializer->value = 0.0f;

			LayerOp layer(d);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			layer->kernel_mult.reset(new ParamSpecMult());
			layer->bias_mult.reset(new ParamSpecMult());

			layer->kernel_mult->decay_mult = 0;
			layer->kernel_mult->lr_mult = 2;

			layer->bias_mult->decay_mult = 0;
			layer->bias_mult->lr_mult = 1;

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;

			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor add(const Tensor&  a, const Tensor&  b, const string& name){
			LayerOp layer(new OAdd());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(2);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->input[0] = a;
			layer->input[1] = b;
			layer->output[0] = blob;
			return blob;
		}

		Tensor maxop(const Tensor&  a, const Tensor&  b, const string& name){
			LayerOp layer(new OMax());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(2);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->input[0] = a;
			layer->input[1] = b;
			layer->output[0] = blob;
			return blob;
		}

		Tensor softmax(const Tensor&  x, const string& name, bool inplace){
			LayerOp layer(new OSoftmax());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor relu(const Tensor&  x, const string& name, bool inplace){
			LayerOp layer(new OReLU());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor sigmoid(const Tensor&  x, const string& name, bool inplace){
			LayerOp layer(new OSigmoid());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor batch_norm_only(const Tensor&  x, const string& name, bool inplace){
			OBatchNorm* bn = new OBatchNorm();

			LayerOp layer(new OBatchNorm());
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor roi_pooling(const Tensor&  feature_map, const Tensor&  rois, int pooled_w, int pooled_h, float spatial_scale, const string& name){
			OROIPooling* pool = new OROIPooling();
			pool->pooled_w = pooled_w;
			pool->pooled_h = pooled_h;
			pool->spatial_scale = spatial_scale;
			pool->name = pool->scope_name_or_next_auto_name(name);

			LayerOp layer(pool);
			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->input = { feature_map, rois };
			layer->output = { blob };
			return blob;
		}

		Tensor scale(const Tensor&  x, bool bias_term, const string& name, bool inplace){
			OScale* scale = new OScale();
			scale->bias_term = bias_term;

			LayerOp layer(scale);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor batch_norm(const Tensor&  x, bool bias_term, const string& name, bool inplace){
			Tensor o;
			o = batch_norm_only(x, name.empty() ? "" : name + "/bn", inplace);
			o = scale(o, bias_term, name.empty() ? "" : name + "/scale", inplace);
			return o;
		}

		Tensor dropout(const Tensor&  x, float keep_prob, const string& name, bool inplace){

			ODropout* drop = new ODropout();
			drop->keep_prob = keep_prob;

			LayerOp layer(drop);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);
			layer->input.resize(1);

			Tensor blob(new OTensor());
			blob->name = inplace ? x->name : layer->name;
			blob->owner = layer;
			layer->input[0] = x;
			layer->output[0] = blob;
			return blob;
		}

		Tensor crop(const vector<Tensor>& x, int axis, const string& name, const vector<int>& offset){

			OCrop* c = new OCrop();
			c->axis = axis;
			c->offset = offset;

			LayerOp layer(c);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->output.resize(1);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->input = x;
			layer->output[0] = blob;
			return blob;
		}
	}

	// Return the current learning rate. The currently implemented learning rate
	// policies are as follows:
	//    - fixed: always return base_lr.
	//    - step: return base_lr * gamma ^ (floor(iter / step))
	//    - exp: return base_lr * gamma ^ iter
	//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
	//    - multistep: similar to step but it allows non uniform steps defined by
	//      stepvalue
	//    - poly: the effective learning rate follows a polynomial decay, to be
	//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
	//    - sigmoid: the effective learning rate follows a sigmod decay
	//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
	//
	// where base_lr, max_iter, gamma, step, stepvalue and power are defined
	// in the solver parameter protocol buffer, and iter is the current iteration.
	namespace learningrate{

		//
		//    fixed学习率策略  always return base_lr.
		//
		LearningRatePolicy fixed(double base_lr){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "fixed";
			return lrp;
		}

		//
		//    step学习率策略  return base_lr * gamma ^ (floor(iter / step_size))
		//
		LearningRatePolicy step(double base_lr, float gamma, int step_size){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "step";
			lrp->gamma = gamma;
			lrp->stepsize = step_size;
			return lrp;
		}

		//
		//    exp学习率策略  return base_lr * gamma ^ iter
		//
		LearningRatePolicy exp(double base_lr, float gamma){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "exp";
			lrp->gamma = gamma;
			return lrp;
		}

		//
		//    inv学习率策略  return base_lr * (1 + gamma * iter) ^ (- power)
		//
		LearningRatePolicy inv(double base_lr, float gamma, float power){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "inv";
			lrp->gamma = gamma;
			lrp->power = power;
			return lrp;
		}

		//
		//    multistep学习率策略  similar to step but it allows non uniform steps defined by
		//      stepvalue
		//
		LearningRatePolicy multistep(double base_lr, float gamma, const vector<int>& stepvalue){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "multistep";
			lrp->gamma = gamma;
			lrp->stepvalue = stepvalue;
			return lrp;
		}

		//
		//    poly学习率策略  the effective learning rate follows a polynomial decay, to be
		//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
		//
		LearningRatePolicy poly(double base_lr, float power){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "poly";
			lrp->power = power;
			return lrp;
		}

		//
		//    sigmoid学习率策略  the effective learning rate follows a sigmod decay
		//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
		//
		LearningRatePolicy sigmoid(double base_lr, float gamma, int stepsize){
			LearningRatePolicy lrp(new OLearningRatePolicy());
			lrp->base_lr = base_lr;
			lrp->policy = "sigmoid";
			lrp->gamma = gamma;
			lrp->stepsize = stepsize;
			return lrp;
		}
	};

	namespace optimizer{

		string AdaptiveMomentEstimation::seril_sub_param(){
			string result;
			result += f("momentum: %g\n", beta1);
			result += f("momentum2: %g\n", beta2);
			result += f("delta: %g\n", delta);
			return result + "solver_type: ADAM";
		}

		Optimizer stochasticGradientDescent(LearningRatePolicy lr){
			StochasticGradientDescent* sgd = new StochasticGradientDescent();
			sgd->setlr(lr);

			Optimizer op(sgd);
			return op;
		}

		Optimizer momentumStochasticGradientDescent(LearningRatePolicy lr, float momentum){
			StochasticGradientDescent* sgd = new StochasticGradientDescent();
			sgd->setlr(lr);
			sgd->momentum = momentum;

			Optimizer op(sgd);
			return op;
		}

		Optimizer adaptiveMomentEstimation(LearningRatePolicy lr, float beta1, float beta2, float delta){
			AdaptiveMomentEstimation* adam = new AdaptiveMomentEstimation();
			adam->setlr(lr);
			adam->beta1 = beta1;
			adam->beta2 = beta2;
			adam->delta = delta;

			Optimizer op(adam);
			return op;
		}
	};

	//
	//     loss的定义
	//
	namespace loss{

		string OSoftmaxCrossEntropy::serial_param(){
			string part;
			if (axis != 1) part += f("axis: %d\n", axis);
			if (hard_ratio != 1.0f) part += f("hard_ratio: %g\n", hard_ratio);
			if (!hard_mining_label.empty()) part += f("hard_mining_label: %d\n", hard_mining_label.intval());
			if (!cutting_point.empty()) part += f("cutting_point: %g\n", cutting_point.floatval());
			if (normalize_type != "Softmax") part += f("normalize_type: %s\n", normalize_type.c_str());
			for (size_t i = 0; i < class_weight.floatarraySize(); ++i)
				part += f("class_weight: %g\n", class_weight.floatval(i));

			string softmax_param = part.empty() ? "" : "softmax_param {\n" + part + "\n}";
			string loss_param;
			if (!ignore_label.empty()) loss_param += f("ignore_label: %d\n", ignore_label.intval());
			if (normalize) loss_param += f("normalize: %s\n", bool_string(normalize));
			if (!loss_param.empty()) softmax_param += f("\nloss_param{\n%s}", loss_param.c_str());
			return softmax_param;
		}

		string OSigmoidCrossEntropy::serial_param(){

			string loss_param;
			if (!ignore_label.empty()) loss_param += f("ignore_label: %d\n", ignore_label.intval());
			if (normalize) loss_param += f("normalize: %s\n", bool_string(normalize));
			if (!loss_param.empty()) loss_param = f("loss_param{\n%s}", loss_param.c_str());
			return loss_param;
		}

		string OEuclideanLoss::serial_param(){

			string loss_param;
			if (normalize) loss_param += f("normalize: %s\n", bool_string(normalize));
			if (!loss_param.empty()) loss_param = f("loss_param{\n%s}", loss_param.c_str());
			return loss_param;
		}

		string OSmoothL1::serial_param(){
			if (sigma.empty())
				return "";
			return f("smooth_l1_loss_param {\nsigma: %g\n}", sigma.floatval());
		}

		//
		//    具体交叉熵损失的函数定义
		//
		Tensor softmax_cross_entropy(const Tensor&  x, const Tensor&  y, const string& name, Tensor* loss_weight, bool normalize, DynamicValue ignore_label){

			OSoftmaxCrossEntropy* loss_ = new OSoftmaxCrossEntropy();
			loss_->ignore_label = ignore_label;
			loss_->normalize = normalize;

			LayerOp layer(loss_);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.push_back(x);
			layer->input.push_back(y);
			if (loss_weight)
				layer->input.push_back(*loss_weight);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->output.push_back(blob);
			return blob;
		}

		//
		//    具体交叉熵损失的函数定义
		//
		Tensor smooth_l1(const Tensor&  x, const Tensor&  y, float sigma, const string& name, const vector<Tensor>& loss_weights){

			OSmoothL1* loss_ = new OSmoothL1();
			loss_->sigma = sigma;

			LayerOp layer(loss_);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.push_back(x);
			layer->input.push_back(y);
			for (int i = 0; i < loss_weights.size(); ++i)
				layer->input.push_back(loss_weights[i]);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->output.push_back(blob);
			return blob;
		}

		//
		//    具体sigmoid交叉熵损失的函数定义
		//
		Tensor sigmoid_cross_entropy(const Tensor& x, const Tensor& y, const string& name, bool normalize, DynamicValue ignore_label){

			OSigmoidCrossEntropy* loss_ = new OSigmoidCrossEntropy();
			loss_->ignore_label = ignore_label;
			loss_->normalize = normalize;

			LayerOp layer(loss_);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.push_back(x);
			layer->input.push_back(y);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->output.push_back(blob);
			return blob;
		}

		//
		//    具体交叉熵损失的函数定义
		//
		Tensor euclidean(const Tensor& x, const Tensor& y, Tensor* loss_weight, const string& name, bool normalize){

			OEuclideanLoss* loss_ = new OEuclideanLoss();
			loss_->normalize = normalize;

			LayerOp layer(loss_);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.push_back(x);
			layer->input.push_back(y);

			if (loss_weight)
				layer->input.push_back(*loss_weight);

			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->output.push_back(blob);
			return blob;
		}
	};

	//
	//    关于评测方法
	//
	namespace metric{

		string OClassifyAccuracy::serial_param(){
			string part;
			if (top_k != 1) part += f("top_k: %d\n", top_k);
			if (axis != 1) part += f("axis: %d\n", axis);
			if (!ignore_label.empty()) part += f("ignore_label: %d\n", ignore_label.intval());
			if (part.empty()) return "";
			return f("accuracy_param {\n%s\n}", part.c_str());
		}

		//
		//    accuracy
		//
		Tensor classifyAccuracy(const Tensor&  x, const Tensor&  y, const string& name){

			OClassifyAccuracy* acc = new OClassifyAccuracy();
			LayerOp layer(acc);
			layer->name = layer->scope_name_or_next_auto_name(name);
			layer->input.resize(2);
			layer->output.resize(1);
			layer->phase.reset(new Phase(Phase_TEST));

			layer->input[0] = x;
			layer->input[1] = y;
			Tensor blob(new OTensor());
			blob->name = layer->name;
			blob->owner = layer;
			layer->output[0] = blob;
			return blob;
		}
	}
}