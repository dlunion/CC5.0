

#ifndef CC_H
#define CC_H

#include <opencv2/opencv.hpp>
#include <initializer_list>
#include <list>
#include <mutex>
#include <thread>
#include <memory>
#include <functional>

#ifdef WIN32
	#ifdef EXPORT_CC_DLL
		#define CCAPI __declspec(dllexport)  
	#else
		#define CCAPI //__declspec(dllimport)  
		#pragma comment(lib, "libcaffe5.0.lib")
	#endif
	#define CCCALL __stdcall
#else
	#define CCAPI
	#define CCCALL
#endif

#define VersionStr		"CC5.0"
#define VersionInt		0x0500


namespace cc{

	using cv::Mat;
	using cv::Scalar;


	//
	//    phase define
	//
	#define  PhaseTrain			0
	#define  PhaseTest			1


	//
	//    CC's Tensor data
	//
	class Blob;
	struct CCAPI BlobData{
		float* list;
		int num;
		int channels;
		int height;
		int width;
		int capacity_count;

		BlobData();
		virtual ~BlobData();
		bool empty() const;
		int count() const;
		void reshape(int num, int channels, int height, int width);
		void reshapeLike(const BlobData* other);
		void copyFrom(const BlobData* other);
		void copyFrom(const Blob* other);
		void reshapeLike(const Blob* other);
		void release();
	};


	//
	//    Caffe's Tensor
	//
	class CCAPI Blob{
	public:
		Blob();
		void setNative(void* native);
		const void* getNative() const;
		int shape(int index) const;
		int num_axes() const;
		int count() const;
		int count(int start_axis) const;
		int height() const;
		int width() const;
		int channel() const;
		int num() const;
		void set_cpu_data(float* data);

		const float* cpu_data() const;
		const float* gpu_data() const;
		float* mutable_cpu_data();
		float* mutable_gpu_data();
		inline float* cpu_ptr(int n = 0, int c = 0, int h = 0, int w = 0){ return mutable_cpu_data() + offset(n, c, h, w); }
		float* gpu_ptr(int n = 0, int c = 0, int h = 0, int w = 0){ return mutable_gpu_data() + offset(n, c, h, w); }
		float& cpu_at(int n = 0, int c = 0, int h = 0, int w = 0){ return *(mutable_cpu_data() + offset(n, c, h, w)); }
		float& gpu_at(int n = 0, int c = 0, int h = 0, int w = 0){ return *(mutable_gpu_data() + offset(n, c, h, w)); }

		const float* cpu_diff() const;
		const float* gpu_diff() const;
		float* mutable_cpu_diff();
		float* mutable_gpu_diff();
		float* cpu_diff_ptr(int n = 0, int c = 0, int h = 0, int w = 0){ return mutable_cpu_diff() + offset(n, c, h, w); }
		float* gpu_diff_ptr(int n = 0, int c = 0, int h = 0, int w = 0){ return mutable_gpu_diff() + offset(n, c, h, w); }
		float& cpu_diff_at(int n = 0, int c = 0, int h = 0, int w = 0){ return *(mutable_cpu_diff() + offset(n, c, h, w)); }
		float& gpu_diff_at(int n = 0, int c = 0, int h = 0, int w = 0){ return *(mutable_gpu_diff() + offset(n, c, h, w)); }
		void setTo(float value);

		void reshape(int num = 1, int channels = 1, int height = 1, int width = 1);
		void reshape(int numShape, int* shapeDims);
		void reshapeLike(const Blob* other);
		void copyFrom(const Blob* other, bool copyDiff = false, bool reshape = false);
		void copyFrom(const BlobData* other);
		void copyDiffFrom(const Blob* other);
		void setData(int numIndex, const uchar* imdataptr, cv::Size imsize, int channels = 3, const Scalar& meanValue = Scalar(), float scale = 1.0f);
		void setData(int numIndex, const float* imdataptr, cv::Size imsize, int channels = 3, const Scalar& meanValue = Scalar(), float scale = 1.0f);
		bool setData(int numIndex, const void* imdataptr, int datalength, int color = 1, const Scalar& meanValue = Scalar(), float scale = 1.0f);
		void setData(int numIndex, const Mat& data, const Scalar& meanValue = Scalar(), float scale = 1.0f);
		std::shared_ptr<Blob> transpose(int axis0, int axis1, int axis2, int axis3);
		int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const;
		void updateInfo();

	private:
		int _dims[4];
		int _num_axis;
		void* _native;
	};


	//
	//    Custom layer
	//
	class CCAPI BaseLayer;


	//
	//    Caffe's Layer
	//
	class CCAPI Layer{
	public:
		Layer();
		virtual ~Layer();	
		void setupLossWeights(int num, float* weights);
		float lossWeights(int index);
		void setLossWeights(int index, float weights);
		const char* type() const;
		const char* name() const;
		int getNumBottom() const;
		int getNumTop() const;
		const char* bottomName(int index) const;
		const char* topName(int index) const;
		Blob* paramBlob(int index) const;
		int getNumParamBlob() const;
		BaseLayer* getBaseLayerInstance() const;
		void setBaseLayerInstance(BaseLayer* baselayer);

		void setup(const Blob** bottom, int numbottom, const Blob** top, int numtop);
		void reshape(const Blob** bottom, int numbottom, const Blob** top, int numtop);
		float forward(const Blob** bottom, int numbottom, const Blob** top, int numtop);
		void backward(const Blob** bottom, int numbottom, const bool* propagate_down, int propagates, const Blob** top, int numtop);

		//Called at caffe::Layer<DType> construction
		void setNative(void* native);
		void* getNative() const;
		void setSharedPtrNative(void* native);
		void* getSharedPtrNative() const;

	private:

		//nullptr if not custom layer, otherwish is valid.
		BaseLayer* _baselayer_instance;

		//caffe::Layer<float> instance
		void* _native;

		//boost::shared_ptr<caffe::Layer<float>>*
		void* _shared_ptr_native;
	};


	//
	//    Caffe's Net
	//
	class CCAPI Net{
	public:
		void setNative(void* native);
		void* getNative();

		Blob* blob(const char* name);
		Blob* blob(int index);
		void forward(float* loss = 0);
		void reshape();
		bool weightsFromFile(const char* file);
		bool weightsFromData(const void* data, int length);
		void shareTrainedLayersWith(const Net* other);
		bool has_blob(const char* name);
		bool has_layer(const char* name);
		int num_input_blobs();
		int num_output_blobs();
		int num_blobs();
		const char* blob_name(int index);
		const char* layer_name(int index);
		Blob* input_blob(int index);
		Blob* output_blob(int index);
		int input_blob_indice(int index);
		int output_blob_indice(int index);
		const char* input_blob_name(int index);
		const char* output_blob_name(int index);
		int num_layers();
		Layer* layer(const char* name);
		Layer* layer(int index);
		size_t memory_used();
		bool saveToCaffemodel(const char* path);
		bool saveToPrototxt(const char* path, bool write_weights = false);

	private:
		void* _native;
	};
	 

	//
	//    Caffe's Solver
	//
	class CCAPI Solver;

	//Callback function called at the end of each step
	typedef void(*TrainStepEndCallback)(Solver* solver, int step, float smoothed_loss, void* userdata);

	class CCAPI Solver{
	public:
		Solver();
		virtual ~Solver();

		void setNative(void* native);
		void step(int iters);
		Net* net();
		int num_test_net();
		Net* test_net(int index = 0);
		void* getNative();
		int iter();
		float smooth_loss();
		void restore(const char* solvestate_file);
		void snapshot(const char* caffemodel_savepath = 0, bool save_solver_state = true);
		int max_iter();
		void solve(int numGPU = 0, int* gpuid = nullptr);
		void installActionSignalOperator();
		void setBaseLearningRate(float rate);
		float getBaseLearningRate();
		void postSnapshotSignal();
		void postEarlyStopSignal();
		void testAll();
		void setSetpEndCallback(TrainStepEndCallback callback, void* userdata = nullptr);
		TrainStepEndCallback getStepEndCallback();
		void* getStepEndCallbackUserData();
		
	private:
		TrainStepEndCallback stepEndCallback_;
		void* stepEndCallbackUserData_;
		void* signalHandler_;
		void* _native;
	};

	CCAPI std::shared_ptr<Blob> CCCALL newBlob();
	CCAPI std::shared_ptr<Blob> CCCALL newBlobByShape(int num = 1, int channels = 1, int height = 1, int width = 1);
	CCAPI std::shared_ptr<Blob> CCCALL newBlobByShapes(int num_shape, int* shapes);
	CCAPI std::shared_ptr<Layer> CCCALL newLayer(const char* layer_defined, int length = -1);

	CCAPI std::shared_ptr<Solver> CCCALL loadSolverFromPrototxt(const char* solver_prototxt, const char* netstring = 0);
	CCAPI std::shared_ptr<Solver> CCCALL loadSolverFromPrototxtString(const char* solver_prototxt_string, const char* netstring = 0);

	CCAPI std::shared_ptr<Net> CCCALL loadNetFromPrototxt(const char* net_prototxt, int phase = PhaseTest);
	CCAPI std::shared_ptr<Net> CCCALL loadNetFromPrototxtString(const char* net_prototxt, int length = -1, int phase = PhaseTest);

	CCAPI void CCCALL setGPU(int id);
	CCAPI bool CCCALL checkDevice(int id);


	//
	//    Abstract Custom Layer
	//
	class CCAPI BaseLayer{
	public:
		virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop) = 0;
		virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop) = 0;
		virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down){};
		virtual void reshape(Blob** bottom, int numBottom, Blob** top, int numTop){};
		virtual const char* type() = 0;
		virtual ~BaseLayer(){}
		void* getNative();
		void setNative(void* ptr);
		Layer* ccLayer();

	private:
		void* native_;
	};


	//
	//    Custom registration layer
	//
	typedef void* CustomLayerInstance;
	typedef BaseLayer* (*createLayerFunc)();
	typedef void(*releaseLayerFunc)(BaseLayer* layer);
	typedef CustomLayerInstance(CCCALL *newLayerFunction)(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop, void* native);
	typedef void(CCCALL *customLayerForward)(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop);
	typedef void(CCCALL *customLayerBackward)(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down);
	typedef void(CCCALL *customLayerReshape)(CustomLayerInstance instance, Blob** bottom, int numBottom, Blob** top, int numTop);
	typedef void(CCCALL *customLayerRelease)(CustomLayerInstance instance);
	typedef void(OnTestClassification)(Solver* solver, float testloss, int index, const char* itemname, float itemscore);
	typedef void(OnOptimizationStopped)(Solver* solver, bool early, int iters, float smoothed_loss);
	

	CCAPI void CCCALL registerOnTestClassificationFunction(OnTestClassification func);
	CCAPI void CCCALL registerOnOptimizationStopped(OnOptimizationStopped func);
	CCAPI void CCCALL registerLayerFunction(newLayerFunction newlayerFunc);
	CCAPI void CCCALL registerLayerForwardFunction(customLayerForward forward);
	CCAPI void CCCALL registerLayerBackwardFunction(customLayerBackward backward);
	CCAPI void CCCALL registerLayerReshapeFunction(customLayerReshape reshape);
	CCAPI void CCCALL registerLayerReleaseFunction(customLayerRelease release);
	CCAPI void CCCALL installRegister();
	CCAPI void CCCALL installLayer(const char* type, createLayerFunc func, releaseLayerFunc release);


#define INSTALL_LAYER(classes)    {installLayer(#classes, classes::creater, classes::release);};
#define SETUP_LAYERFUNC(classes)  static BaseLayer* creater(){return new classes();} static void release(BaseLayer* layer){if (layer) delete layer; };  virtual const char* type(){return #classes;}

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	CCAPI int CCCALL argmax(const Blob* classification_blob, int numIndex = 0, float* conf_ptr = 0);
	CCAPI int CCCALL argmax(const float* data_ptr, int num_data, float* conf_ptr);
	CCAPI std::shared_ptr<BlobData> CCCALL newBlobData(int num, int channels, int height, int width);
	CCAPI std::shared_ptr<BlobData> CCCALL newBlobDataFromBlobShape(const Blob* blob);
	CCAPI void CCCALL copyFromBlob(BlobData* dest, const Blob* blob);
	CCAPI void CCCALL copyOneFromBlob(BlobData* dest, const Blob* blob, int numIndex);
	CCAPI void CCCALL releaseBlobData(BlobData* ptr);
	CCAPI void CCCALL disableLogPrintToConsole();
	CCAPI const char* CCCALL getCCVersionString();
	CCAPI int CCCALL getCCVersionInt();


	template<typename _DType>
	class ThreadSafetyQueue{
	public:

		//maxLimit最大限制，如果超过最大限制的时候会等待
		ThreadSafetyQueue(int maxLimit){
			this->maxLimit_ = maxLimit;
		}

		//加入数据到队列
		void push(const _DType& in){

			if (maxLimit_){
				while (frameList_.size() >= maxLimit_ && !eof_)
					std::this_thread::sleep_for(std::chrono::milliseconds(1));

				if (eof_) return;
			}

			std::unique_lock<std::mutex> l(lock_);
			frameList_.push_back(in);
		}

		bool empty(){
			return frameList_.empty();
		}

		bool pull(_DType& out){
			std::unique_lock<std::mutex> l(lock_);

			if (frameList_.empty())
				return false;

			out = frameList_.front();
			frameList_.pop_front();
			return true;
		}

		void setEOF(){
			eof_ = true;
		}

		bool eof(){
			return eof_;
		}

	private:
		int maxLimit_;
		std::mutex lock_;
		std::list<_DType> frameList_;
		volatile bool eof_ = false;
	};

	namespace plugin{

		CCAPI void CCCALL openNetscope(const char* netName);
		CCAPI bool CCCALL postPrototxt(const char* netName, const char* userName, const char* content, int length = -1);
	};
};

#endif //CC_H