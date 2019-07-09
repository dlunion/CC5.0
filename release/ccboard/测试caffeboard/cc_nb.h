

#ifndef CC_NETBUILD_H
#define CC_NETBUILD_H
#include "cc_v5.h"
#include <opencv2/opencv.hpp>
#include <initializer_list>
#include <list>
#include <mutex>
#include <thread>
#include <memory>
#include <string>
#include <vector>

using std::string;
using std::vector;

namespace cc{

	struct OLayerOp;
	struct OTensor;
	struct OOptimizer;
	struct OLearningRatePolicy;
	struct OThreadContextSession;

	//
	//    层和张量，实际上是个智能指针
	//
	typedef std::shared_ptr<OLayerOp> LayerOp;
	typedef std::shared_ptr<OTensor> Tensor;
	typedef std::shared_ptr<OOptimizer> Optimizer;
	typedef std::shared_ptr<OThreadContextSession> Session;
	typedef std::shared_ptr<OLearningRatePolicy> LearningRatePolicy;
	typedef int LayerID;

	//
	//    字符串的format，事实上后面的拼接很多都用到它了
	//
	string f(const char* fmt, ...);

	//
	//    权重初始化时，协方差正则化方法
	//
	enum VarianceNorm{
		VarianceNorm_FAN_IN = 0,
		VarianceNorm_FAN_OUT = 1,
		VarianceNorm_AVERAGE = 2
	};

	inline const char* variance_norm_string(VarianceNorm norm){
		switch (norm){
		case VarianceNorm_FAN_IN: return "FAN_IN";
		case VarianceNorm_FAN_OUT: return "FAN_OUT";
		case VarianceNorm_AVERAGE: return "AVERAGE";
		default: return "ErrorVarianceNorm";
		}
	}

	//
	//    Phase定义
	//
	enum Phase{
		Phase_TRAIN = 0,
		Phase_TEST = 1
	};

	inline const char* phase_string(Phase phase){
		switch (phase){
		case Phase_TRAIN: return "TRAIN";
		case Phase_TEST: return "TEST";
		default: return "ErrorPhase";
		}
	}

	inline const char* bool_string(bool val){
		return val ? "true" : "false";
	}

	//
	//   动态类型的定义
	//
	enum DynamicValue_Type{
		DynamicValue_Type_Empty = 0,
		DynamicValue_Type_Int = 1,
		DynamicValue_Type_Float = 2,
		DynamicValue_Type_Double = 3,
		DynamicValue_Type_String = 4,
		DynamicValue_Type_Bool = 5,
		DynamicValue_Type_IntArray = 6,
		DynamicValue_Type_FloatArray = 7,
		DynamicValue_Type_DoubleArray = 8
	};

	//
	//   动态类型的值定义
	//
	struct DynamicValue_v{
		union{
			int intV;
			float floatV;
			bool boolV;
			double doubleV;
		};

		vector<int> intarrayv;
		vector<float> floatarrayv;
		vector<float> doublearrayv;
		string stringv;
		DynamicValue_Type type;
	};

	//
	//   动态类型的类定义，该类可以支持int、float、bool等类型，还支持intarray, floatarray等
	//   通过构造、和操作符重载完成支持，重要的是还支持空值，即不赋值给他时的默认状态为空
	//
	class DynamicValue{
	public:
		DynamicValue(){
			value.intV = 0;
			value.type = DynamicValue_Type_Empty;
		}

		bool empty() const{
			return value.type == DynamicValue_Type_Empty;
		}

		DynamicValue(int v){
			value.intV = v;
			value.type = DynamicValue_Type_Int;
		}

		DynamicValue(float v){
			value.floatV = v;
			value.type = DynamicValue_Type_Float;
		}

		DynamicValue(double v){
			value.doubleV = v;
			value.type = DynamicValue_Type_Double;
		}

		DynamicValue(const char* v){
			value.stringv = v;
			value.type = DynamicValue_Type_String;
		}

		DynamicValue(const string& v){
			value.stringv = v;
			value.type = DynamicValue_Type_String;
		}

		DynamicValue(bool v){
			value.boolV = v;
			value.type = DynamicValue_Type_Bool;
		}

		DynamicValue(const vector<int>& intarray){
			value.intarrayv = intarray;
			value.type = DynamicValue_Type_IntArray;
		}

		DynamicValue(const vector<float>& floatarray){
			value.floatarrayv = floatarray;
			value.type = DynamicValue_Type_FloatArray;
		}

		DynamicValue(const DynamicValue& other){
			value = other.value;
		}

		DynamicValue& operator +=(const char* other){
			value.stringv = value.stringv + other;
			value.type = DynamicValue_Type_String;
			return *this;
		}

		DynamicValue& operator =(const char* other){
			value.stringv = other;
			value.type = DynamicValue_Type_String;
			return *this;
		}

		DynamicValue& operator =(const string& other){
			value.stringv = other;
			value.type = DynamicValue_Type_String;
			return *this;
		}

		DynamicValue& operator =(const DynamicValue& other){
			value = other.value;
			return *this;
		}

		void checktype(DynamicValue_Type dtype) const{
			if (value.type != dtype){
				throw f("error type %d, except: %d\n", value.type, dtype);
			}
		}

		bool isarray() const{
			return (value.type == DynamicValue_Type_IntArray ||
				value.type == DynamicValue_Type_FloatArray ||
				value.type == DynamicValue_Type_DoubleArray);
		}

		bool isnumerical() const{
			return isarray() || isconst();
		}

		bool isconst() const{
			return (
				value.type == DynamicValue_Type_Int ||
				value.type == DynamicValue_Type_Float ||
				value.type == DynamicValue_Type_Double);
		}

		bool isstr() const{
			return value.type == DynamicValue_Type_String;
		}

		size_t size() const{
			if (isconst()) return 1;
			if (!isarray()) return 0;

			if (value.type == DynamicValue_Type_IntArray) return value.intarrayv.size();
			if (value.type == DynamicValue_Type_FloatArray) return value.floatarrayv.size();
			if (value.type == DynamicValue_Type_DoubleArray) return value.doublearrayv.size();
			return 0;
		}

		int intval(size_t index = 0) const{
			if (value.type == DynamicValue_Type_Int)
				return value.intV;

			if (value.type == DynamicValue_Type_IntArray)
				return value.intarrayv[index];

			if (value.type == DynamicValue_Type_Float)
				return floatval(index);

			if (value.type == DynamicValue_Type_Double)
				return doubleval(index);
			return 0;
		}

		const string& strval() const{
			return operator const string&();
		}

		float floatval(size_t index = 0) const{
			if (value.type == DynamicValue_Type_Float)
				return value.floatV;

			if (value.type == DynamicValue_Type_FloatArray)
				return value.floatarrayv[index];

			if (value.type == DynamicValue_Type_Int)
				return intval(index);

			if (value.type == DynamicValue_Type_Double)
				return doubleval(index);
			return 0;
		}

		float doubleval(size_t index = 0) const{
			if (value.type == DynamicValue_Type_Double)
				return value.doubleV;

			if (value.type == DynamicValue_Type_DoubleArray)
				return value.doublearrayv[index];

			if (value.type == DynamicValue_Type_Int)
				return intval(index);

			if (value.type == DynamicValue_Type_Float)
				return floatval(index);
			return 0;
		}

		operator const string&() const{
			checktype(DynamicValue_Type_String);
			return value.stringv;
		}

		operator unsigned int() const{
			return operator int();
		}

		operator int() const{
			return intval();
		}

		operator float() const{
			return floatval();
		}

		operator double() const{
			return doubleval();
		}

		operator bool() const{
			checktype(DynamicValue_Type_Bool);
			return value.boolV;
		}

		operator const char*() const{
			checktype(DynamicValue_Type_String);
			return value.stringv.c_str();
		}
	private:
		DynamicValue_v value;
	};

#define Empty	DynamicValue()

	struct OThreadContextSession;
	typedef void(StepEndCallbackFunctional)(OThreadContextSession* session, int step, float smoothed_loss);
	typedef void(PreStepEndCallbackFunctional)(OThreadContextSession* session, int step, float smoothed_loss);
	typedef void(OverStepEndCallbackFunctional)(OThreadContextSession* session, int step, float smoothed_loss);
	typedef void(PreTrainCallbackFunctional)(OThreadContextSession* session);

	//
	//    基于线程上下文的Session
	//
	struct OThreadContextSession{
		virtual cc::Blob* get_tensor_blob(const char* blob_name) = 0;

		//获取存储在session中的值
		virtual void* get(const char* key) = 0;
		virtual void put(const char* key, void* value) = 0;
		virtual LayerID next_layer_id() = 0;
		virtual void clean_auto_name_info() = 0;

		//单例模式获取实例指针
		static OThreadContextSession* this_thread();

		//获取求解器
		virtual cc::Solver* solver() = 0;


		static void* this_thread_get(const char* key);
		static void this_thread_put(const char* key, void* value);
		static cc::Solver* this_thread_solver();
		static void this_thread_clean_auto_name_info();
		static void set_pre_step_end_callback(const std::function<PreStepEndCallbackFunctional>& callback);
		static void set_over_step_end_callback(const std::function<OverStepEndCallbackFunctional>& callback);
		static void add_pre_train_callback(const std::function<PreTrainCallbackFunctional>& callback);
		static void add_step_end_callback(const std::function<StepEndCallbackFunctional>& callback);
	};

	//
	//    获取名称，基于当前上下文中的scope指定名称
	//    返回名称以： scope / name 的形式，若scope为空，则返回name
	//
	string get_name_with_scope(const string& name);

	//
	//    scope的具体实现定义，构造的时候push，析构的时候pop
	//
	class name_scope{
	public:
		name_scope(const string& name);
		virtual ~name_scope();
	};

	//
	//    关于权重初始化的定义
	//
	struct Initializer{
		string type = "constant";
		float value = 0;
		float minval = 0;
		float maxval = 1;
		float meanval = 0;
		float stdval = 1;
		int sparse = -1;
		VarianceNorm variance_norm = VarianceNorm_FAN_IN;

		Initializer(){}
		Initializer(const string& type, float stdval = 0):type(type), stdval(stdval){}
		Initializer(float value, const string& type = "constant"):value(value), type(type){}

		string seril();
	};

	//
	//    张量的基本定义，输出是基于张量的
	//
	struct OTensor{
		LayerOp owner;
		string name;

		//format: nchw
		//string shapestr();

		static Tensor getTensorFromName(
			const std::vector<Tensor>& graph, const std::string& name);
	};

	//
	//    学习率策略定义
	//
	struct OLearningRatePolicy{

		double base_lr;
		string policy;
		DynamicValue gamma;
		DynamicValue power;
		DynamicValue stepsize;
		vector<int> stepvalue;
	};

	//
	//    优化器的定义
	//
	enum SolverMode {
		CPU = 0,
		GPU = 1
	};

	//
	//    求解方式的字符串表达
	//
	inline const char* solver_mode_string(SolverMode mode){
		switch (mode){
		case CPU:  return "CPU";
		case GPU:  return "GPU";
		default:   return "unknow solver mode";
		}
	}

	enum GraphType{
		GraphType_None,
		GraphType_FromTensor,
		GraphType_FromFile,
		GraphType_FromPrototxt
	};

	struct GraphInput{

		vector<Tensor> graphs;
		GraphInput(const std::vector<Tensor>& input);
		GraphInput(const std::initializer_list<Tensor>& input);
		GraphInput(const Tensor& input);
		GraphInput();
	};

	//
	//    优化器定义的基类
	//
	struct OOptimizer{

		int iter_size = 1;
		DynamicValue test_iter;
		int test_interval = 0;
		bool test_initialization = true;
		DynamicValue base_lr;
		DynamicValue display;
		int average_loss = 1;
		DynamicValue max_iter;
		int random_seed = -1;

		string lr_policy;
		DynamicValue gamma;
		DynamicValue power;
		DynamicValue weight_decay;
		string regularization_type;
		DynamicValue stepsize;
		vector<int> stepvalue;

		string reload_weights;

		int snapshot = 0;
		DynamicValue snapshot_prefix;
		bool snapshot_diff = false;
		SolverMode solver_mode = GPU;
		vector<int> device_ids;
		DynamicValue one_epoch_iter_size;

		GraphType graph_type = GraphType_None;
		GraphInput graphs;
		string str_graph;
		string file_graph;

		//
		//    根据学习率策略来赋值
		//
		virtual void setlr(LearningRatePolicy lr){
			this->base_lr = lr->base_lr;
			this->lr_policy = lr->policy;
			this->gamma = lr->gamma;
			this->power = lr->power;
			this->stepsize = lr->stepsize;
			this->stepvalue = lr->stepvalue;
		}

		//
		//    指定要优化的对象，图
		//
		virtual void minimize(const GraphInput& graphs);

		//
		//    指定要优化的对象，图
		//
		virtual void minimizeFromPrototxt(const string& graphstr);

		//
		//    指定要优化的对象，图
		//
		virtual void minimizeFromFile(const string& graphfile);

		//
		//    子类来实现的参数序列化
		//
		virtual string seril_sub_param() = 0;

		//
		//    序列化参数
		//
		virtual string seril();
	};

	//
	//    关于偏置和权重学习率的乘项
	//
	struct ParamSpecMult{
		float lr_mult = 1.0f;
		float decay_mult = 0;

		ParamSpecMult(){}
		ParamSpecMult(float lr, float decay) :lr_mult(lr), decay_mult(decay){}
		string seril(){
			string part;
			part += f("lr_mult: %g\n", lr_mult);
			part += f("decay_mult: %g\n", decay_mult);
			return "param{\n" + part + "}";
		}
	};

	//
	//    层的基本定义
	//
	struct OLayerOp{
		LayerID layer_id = 0;
		vector<Tensor> input;
		vector<Tensor> output;

		string name;
		std::shared_ptr<Phase> phase;
		std::shared_ptr<ParamSpecMult> kernel_mult;
		std::shared_ptr<ParamSpecMult> bias_mult;
		DynamicValue loss_weight;
		vector<int> propagate_down;

		OLayerOp();
		virtual string scope_name_or_next_auto_name(const string& name);
		virtual string serial();
		virtual string serial_param() = 0;
		virtual const char* caffe_type_name() = 0;
	};

	//所有层的定义
	namespace layers{
		//
		//    输入层定义
		//
		struct OInput : public OLayerOp{

			vector<int> dims;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Input"; }
		};

		//
		//    数据层定义
		//
		struct OCustom : public OLayerOp{

			string cpp_type;
			string cpp_param_str;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "CPP"; }
		};

		//
		//    ReLU层定义
		//
		struct OReLU : public OLayerOp{

			virtual string serial_param(){ return ""; }
			virtual const char* caffe_type_name(){ return "ReLU"; }
		};
		 
		//
		//    ReLU层定义
		//
		struct OBatchNorm : public OLayerOp{

			//bool use_global_stats = true;
			float moving_average_fraction = 0.999f;
			float eps = 1e-5f;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "BatchNorm"; }
		};

		//
		//    ReLU层定义
		//
		struct OScale : public OLayerOp{

			int axis = 1;
			int num_axes = 1;
			bool bias_term = false;
			std::shared_ptr<Initializer> bias_filler;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Scale"; }
		};

		//
		//    Dropout层定义
		//
		struct ODropout : public OLayerOp{

			float keep_prob = 0.5f;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Dropout"; }
		};

		//
		//    ReLU层定义
		//
		struct OMax : public OLayerOp{

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Eltwise"; }
		};

		//
		//    Sigmoid层定义
		//
		struct OSigmoid : public OLayerOp{

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Sigmoid"; }
		};

		//
		//    Softmax层定义
		//
		struct OSoftmax : public OLayerOp{
			int axis = 1;
			float hard_ratio = 1.0f;
			DynamicValue class_weight;
			DynamicValue hard_mining_label;
			DynamicValue cutting_point;
			string normalize_type = "Softmax";

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Softmax"; }
		};

		//
		//    ReLU层定义
		//
		struct OProduct : public OLayerOp{

			bool stable_prod_grad = true;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Eltwise"; }
		};

		//
		//    ReLU层定义
		//
		struct OAdd : public OLayerOp{

			vector<float> coeff;
			bool stable_prod_grad = true;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Eltwise"; }
		};

		//
		//    reshape操作
		//
		struct OReshape : public OLayerOp{

			vector<int> new_dims;
			int axis = 0;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Reshape"; }
		};

		//
		//    transpose操作
		//
		struct OTranspose : public OLayerOp{

			vector<int> order;
			int axis = 0;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Permute"; }
		};

		//
		//    crop操作
		//
		struct OCrop : public OLayerOp{

			int axis = 0;
			vector<int> offset;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Crop"; }
		};

		//
		//    concat操作
		//
		struct OConcat : public OLayerOp{

			int axis = 1;
			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Concat"; }
		};

		//
		//    split操作
		//
		struct OSplit : public OLayerOp{

			int axis = 1;
			vector<int> slice_point;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Slice"; }
		};

		//
		//    Im2Col
		//
		struct OIm2Col : public OLayerOp{

			vector<int> kernel;
			vector<int> strides;
			vector<int> dilations;
			vector<int> padding_size;
			string padding;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Im2col"; }
		};

		//
		//    卷积层的定义
		//
		struct OConv2D : public OLayerOp{

			vector<int> kernel;
			vector<int> strides;
			vector<int> dilations;
			vector<int> padding_size;
			string padding;
			bool bias_term = true;
			std::shared_ptr<Initializer> kernel_initializer;
			std::shared_ptr<Initializer> bias_initializer;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Convolution"; }
		};

		//
		//    可变卷积层的定义
		//
		struct ODeformableConv : public OLayerOp{

			vector<int> kernel;
			vector<int> strides;
			vector<int> dilations;
			vector<int> padding_size;
			int deformable_group = 4;
			string padding;
			bool bias_term = true;
			std::shared_ptr<Initializer> kernel_initializer;
			std::shared_ptr<Initializer> bias_initializer;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "DeformableConvolution"; }
		};

		//
		//    反卷积层的定义
		//
		struct ODeconv2D : public OLayerOp{

			vector<int> kernel;
			vector<int> strides;
			vector<int> dilations;
			vector<int> padding_size;
			string padding;
			bool bias_term = true;
			std::shared_ptr<Initializer> kernel_initializer;
			std::shared_ptr<Initializer> bias_initializer;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Deconvolution"; }
		};

		//
		//    ROIPooling
		//
		struct OROIPooling : public OLayerOp{

			int pooled_h = 0;
			int pooled_w = 0;
			float spatial_scale = 1;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "ROIPooling"; }
		};

		//
		//    Dense层的定义
		//
		struct ODense : public OLayerOp{

			int units;
			bool bias_term = true;
			int axis = -1;
			bool transpose = false;
			std::shared_ptr<Initializer> weight_initializer;
			std::shared_ptr<Initializer> bias_initializer;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "InnerProduct"; }
		};

		enum PoolMethod{
			PoolMethod_MAX = 0,
			PoolMethod_AVE = 1,
			PoolMethod_STOCHASTIC = 2
		};

		inline const char* pool_method_string(PoolMethod method){
			switch (method){
			case PoolMethod_MAX: return "MAX";
			case PoolMethod_AVE: return "AVE";
			case PoolMethod_STOCHASTIC: return "STOCHASTIC";
			default: return "ErrorPoolMethod";
			}
		}

		//
		//    maxpooling的定义
		//
		struct OPooling2D : public OLayerOp{

			vector<int> kernel;
			vector<int> strides;
			vector<int> padding_size;
			PoolMethod method = PoolMethod_MAX;
			bool global_pooling = false;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Pooling"; }
		};

		///////////////////////////////////////////////////////////////////////////////

		//
		//    数据输入层的定义
		//
		Tensor input(const vector<int>& dims, const string& name = "");

		//
		//    数据层的定义
		//
		vector<Tensor> data(const string& cpp_type, const vector<string>& output, const string& name = "", const string& param_str = "");

		//
		//    自定义层1
		//
		vector<Tensor> custom(const string& cpp_type, const vector<Tensor>& input, const vector<string>& output, const string& name = "", const string& param_str = "");

		//
		//    自定义层2
		//
		Tensor custom(const string& cpp_type, const vector<Tensor>& input, const string& output, const string& name = "", const string& param_str = "");

		//
		//    自定义层3
		//
		Tensor custom(const string& cpp_type, const Tensor& input, const string& name = "", const string& param_str = "");

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
		Tensor im2col(const Tensor& x, const vector<int>& kernel, const string& padding = "same",
			const vector<int>& strides = { 1, 1 }, const vector<int>& dilations = { 1, 1 }, const string& name = "");
		Tensor conv2d(const Tensor& x, const vector<int>& kernel, const string& padding = "same",
			const vector<int>& strides = { 1, 1 }, const vector<int>& dilations = { 1, 1 }, const string& name = "");
		Tensor deformableConv(const Tensor& x, const vector<int>& kernel, const vector<int>& padding = { 2, 2 },
			const vector<int>& strides = { 1, 1 }, const vector<int>& dilations = { 2, 2 }, const string& name = "");
		Tensor deconv2d(const Tensor& x, const vector<int>& kernel, const string& padding = "same",
			const vector<int>& strides = { 1, 1 }, const vector<int>& dilations = { 1, 1 }, const string& name = "");
		Tensor transpose(const Tensor& x, vector<int> order, const string& name = "", bool inplace = true);
		Tensor reshape(const Tensor& x, vector<int> new_dims, const string& name = "");
		Tensor concat(const vector<Tensor>& tensors, int axis = 1, const string& name = "");
		Tensor max_pooling2d(const Tensor& x, const vector<int>& kernel, const vector<int>& strides = { 1, 1 }, const vector<int>& padding = { 0, 0 }, bool global_pooling = false, const string& name = "");
		Tensor avg_pooling2d(const Tensor& x, const vector<int>& kernel, const vector<int>& strides = { 1, 1 }, const vector<int>& padding = { 0, 0 }, bool global_pooling = false, const string& name = "");
		Tensor dense(const Tensor& x, int units, const string& name = "", bool bias_term = true);
		Tensor add(const Tensor& a, const Tensor& b, const string& name = "");
		Tensor maxop(const Tensor& a, const Tensor& b, const string& name = "");
		Tensor softmax(const Tensor& x, const string& name = "", bool inplace = true);
		Tensor sigmoid(const Tensor& x, const string& name = "", bool inplace = true);
		Tensor relu(const Tensor& x, const string& name = "", bool inplace = true);
		Tensor batch_norm_only(const Tensor& x, const string& name = "", bool inplace = true);
		Tensor roi_pooling(const Tensor& feature_map, const Tensor& rois, int pooled_w, int pooled_h, float spatial_scale, const string& name = "");
		Tensor scale(const Tensor& x, bool bias_term = true, const string& name = "", bool inplace = true);
		Tensor batch_norm(const Tensor& x, bool bias_term = true, const string& name = "", bool inplace = true);
		Tensor dropout(const Tensor& x, float keep_prob = 0.5, const string& name = "", bool inplace = true);

		// bottom[0] supplies the data
		// bottom[1] supplies the size
		Tensor crop(const vector<Tensor>& x, int axis = 2, const string& name = "", const vector<int>& offset = vector<int>());
	};

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
		LearningRatePolicy fixed(double base_lr);

		//
		//    step学习率策略  return base_lr * gamma ^ (floor(iter / step_size))
		//
		LearningRatePolicy step(double base_lr, float gamma, int step_size);

		//
		//    exp学习率策略  return base_lr * gamma ^ iter
		//
		LearningRatePolicy exp(double base_lr, float gamma);

		//
		//    inv学习率策略  return base_lr * (1 + gamma * iter) ^ (- power)
		//
		LearningRatePolicy inv(double base_lr, float gamma, float power);

		//
		//    pose学习率策略  return iter <= 1300 ? base_lr * (1 + gamma * iter) ^ (- power) : rate * (iter / 100 + 1) * 0.1
		//
		LearningRatePolicy pose(double base_lr, float gamma, float power);

		//
		//    multistep学习率策略  similar to step but it allows non uniform steps defined by
		//      stepvalue
		//
		LearningRatePolicy multistep(double base_lr, float gamma, const vector<int>& stepvalue);

		//
		//    poly学习率策略  the effective learning rate follows a polynomial decay, to be
		//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
		//
		LearningRatePolicy poly(double base_lr, float power);

		//
		//    sigmoid学习率策略  the effective learning rate follows a sigmod decay
		//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
		//
		LearningRatePolicy sigmoid(double base_lr, float gamma, int stepsize);
	};

	namespace optimizer{

		//
		//    SGD优化器
		//
		struct StochasticGradientDescent : public OOptimizer{
			DynamicValue momentum;

			virtual string seril_sub_param(){
				string result;
				if (!momentum.empty()) result += f("momentum: %g\n", momentum.floatval());
				return result + "solver_type: SGD";
			}
		};

		//
		//    Adam优化器
		//
		struct AdaptiveMomentEstimation : public OOptimizer{
			float beta1 = 0;        //momentum
			float beta2 = 0.999f;   //momentum2
			float delta = 1e-8f;

			virtual string seril_sub_param();
		};

		Optimizer stochasticGradientDescent(LearningRatePolicy lr);
		Optimizer momentumStochasticGradientDescent(LearningRatePolicy lr, float momentum);
		Optimizer adaptiveMomentEstimation(LearningRatePolicy lr, float beta1, float beta2, float delta);
	};

	//
	//     loss的定义
	//
	namespace loss{

		//
		//    关于softmax的交叉熵损失
		//
		struct OSoftmaxCrossEntropy : public OLayerOp{
			int axis = 1;
			float hard_ratio = 1.0f;
			DynamicValue class_weight;
			DynamicValue hard_mining_label;
			DynamicValue cutting_point;
			string normalize_type = "Softmax";
			DynamicValue ignore_label;
			bool normalize = false;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "SoftmaxWithLoss"; }
		};

		//
		//    关于softmax的交叉熵损失
		//
		struct OSigmoidCrossEntropy : public OLayerOp{
			int axis = 1;
			float hard_ratio = 1.0f;
			DynamicValue class_weight;
			DynamicValue hard_mining_label;
			DynamicValue cutting_point;
			DynamicValue ignore_label;
			bool normalize = false;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "SigmoidCrossEntropyLoss"; }
		};

		//
		//    关于softmax的交叉熵损失
		//
		struct OEuclideanLoss : public OLayerOp{
			
			bool normalize = false;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "EuclideanLoss"; }
		};

		//
		//     Smooth L1 loss
		//
		struct OSmoothL1 : public OLayerOp{
			DynamicValue sigma;     // = 1.0f

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "SmoothL1Loss"; }
		};

		//
		//    具体softmax交叉熵损失的函数定义
		//
		Tensor softmax_cross_entropy(const Tensor& x, const Tensor& y, const string& name = "", Tensor* loss_weight = nullptr, bool normalize = false, DynamicValue ignore_label = Empty);

		//
		//    具体交叉熵损失的函数定义
		//
		Tensor smooth_l1(const Tensor& x, const Tensor& y, float sigma = 1, const string& name = "", const vector<Tensor>& loss_weights = vector<Tensor>());

		//
		//    具体sigmoid交叉熵损失的函数定义
		//
		Tensor sigmoid_cross_entropy(const Tensor& x, const Tensor& y, const string& name = "", bool normalize = false, DynamicValue ignore_label = Empty);

		//
		//    具体交叉熵损失的函数定义
		//
		Tensor euclidean(const Tensor& x, const Tensor& y, Tensor* loss_weight = nullptr, const string& name = "", bool normalize = false);
	};

	//
	//    关于评测方法
	//
	namespace metric{

		//
		//     Accuracy
		//
		struct OClassifyAccuracy : public OLayerOp{
			int top_k = 1;     // = 1.0f
			int axis = 1;
			DynamicValue ignore_label;

			virtual string serial_param();
			virtual const char* caffe_type_name(){ return "Accuracy"; }
		};

		//
		//    accuracy
		//
		Tensor classifyAccuracy(const Tensor& x, const Tensor& y, const string& name = "");
	}

	namespace engine{

		namespace caffe{
			
			//
			//    将计算图编译到caffe支持
			//
			string buildGraph(const GraphInput& graph, const string& name = "");

			//
			//    将计算图编译到caffe支持
			//
			bool buildGraphToFile(const GraphInput& graph, const string& file);


			//
			//    编译一个net，然后可以做inference
			//
			std::shared_ptr<Net> buildNet(const GraphInput& graph, int phase = PhaseTest);
		};
	}

	//
	//    引擎部分
	//
	namespace train{

		namespace caffe{

			//
			//    训练任务执行
			//
			void run(const Optimizer& optimizer, const std::function<StepEndCallbackFunctional>& stepEndCallback = nullptr, const std::function<PreTrainCallbackFunctional>& preTrainCallback = nullptr);
		};
	};
};

#endif //CC_NETBUILD_H