
#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/dll/shared_library.hpp>
#include <boost/python/call_method.hpp>
#include <boost/python/class.hpp>
#include <boost/utility.hpp>
#include <boost/core/noncopyable.hpp>
#include <boost/thread.hpp>
#include <boost/locale.hpp>

#include <condition_variable>
#include <mutex>
#include <queue>
#include <cv.h>
#include <thread>
#include <string>
#include <map>
#include "numpy/ndarrayobject.h"

using namespace cv;
using namespace std;

#define ToPythonUTF(s)		boost::locale::conv::to_utf<char>((s), "GBK")

class PyEnsureGIL
{
public:
	PyEnsureGIL() : _state(PyGILState_Ensure()) {}
	~PyEnsureGIL()
	{
		PyGILState_Release(_state);
	}
private:
	PyGILState_STATE _state;
};

class PyAllowThreads
{
public:
	PyAllowThreads() : _state(PyEval_SaveThread()) {}
	~PyAllowThreads()
	{
		PyEval_RestoreThread(_state);
	}
private:
	PyThreadState* _state;
};

static PyObject* opencv_error = 0;

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
(0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0") * sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
	return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
	return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator : public MatAllocator
{
public:
	NumpyAllocator() {}
	~NumpyAllocator() {}

	void allocate(int dims, const int* sizes, int type, int*& refcount,
		uchar*& datastart, uchar*& data, size_t* step)
	{
		PyEnsureGIL gil;

		int depth = CV_MAT_DEPTH(type);
		int cn = CV_MAT_CN(type);
		const int f = (int)(sizeof(size_t) / 8);
		int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
			depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
			depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
			depth == CV_64F ? NPY_DOUBLE : f * NPY_ULONGLONG + (f ^ 1)*NPY_UINT;
		int i;
		npy_intp _sizes[CV_MAX_DIM + 1];
		for (i = 0; i < dims; i++)
			_sizes[i] = sizes[i];
		if (cn > 1)
		{
			/*if( _sizes[dims-1] == 1 )
			_sizes[dims-1] = cn;
			else*/
			_sizes[dims++] = cn;
		}
		PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
		if (!o)
			CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
		refcount = refcountFromPyObject(o);
		npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
		for (i = 0; i < dims - (cn > 1); i++)
			step[i] = (size_t)_strides[i];
		datastart = data = (uchar*)PyArray_DATA((PyArrayObject*)o);
	}

	void deallocate(int* refcount, uchar*, uchar*)
	{
		PyEnsureGIL gil;
		if (!refcount)
			return;
		PyObject* o = pyObjectFromRefcount(refcount);
		Py_INCREF(o);
		Py_DECREF(o);
	}
};

NumpyAllocator g_numpyAllocator;

static boost::python::handle<> pyopencv_from(const Mat& m)
{
	if (!m.data)
		return boost::python::handle<>();

	Mat temp, *p = (Mat*)&m;
	if (!p->refcount || p->allocator != &g_numpyAllocator)
	{
		temp.allocator = &g_numpyAllocator;
		ERRWRAP2(m.copyTo(temp));
		p = &temp;
	}
	p->addref();
	return boost::python::handle<>(pyObjectFromRefcount(p->refcount));
}

class Job;
struct Environment {
	void(*monster_postBlob)(Environment* env, const char* name, const Mat& blob);
	void(*monster_postChartValue)(Environment* env, const char* chartName, const char* itemName, int iter, float value);
	void(*monster_postImage)(Environment* env, const char* name, const Mat& image);
	void(*monster_postString)(Environment* env, const char* name, const char* str);
	void(*monster_postStatus)(Environment* env, const char* name, const char* str);
	void(*monster_postBegin)(Environment* env);
	void(*monster_postEnd)(Environment* env);
	std::shared_ptr<char>(*monster_systemConfig)(Environment* env, const char* configJSON);
	int(*local_train)();
	void(*local_stopTrain)();
	std::shared_ptr<char>(*local_notify)(const char* eventType, const char* message);

	Job* context;
};

shared_ptr<char> toSharedString(const string& s) {

	if (s.empty())
		return shared_ptr<char>();

	shared_ptr<char> output;
	char* str = (char*)malloc(s.size() + 1);
	str[s.size()] = 0;
	memcpy(str, s.c_str(), s.size());
	output.reset(str, free);
	return output;
}

string sharedStringExtract(const shared_ptr<char>& s) {

	if (!s) return "";
	return string(s.get());
}

class Base
{
public:
	virtual string getName() const { return "Base"; }
};

enum PostInfoType {
	PostInfoType_Blob,
	PostInfoType_ChartValue,
	PostInfoType_Image,
	PostInfoType_String,
	PostInfoType_Status
};

struct ChartItem {
	float value;
	int iter;

	ChartItem(){}
	ChartItem(int iter, float value) {
		this->iter = iter;
		this->value = value;
	}
};

struct PostInfo {

	PostInfoType type = PostInfoType_String;
	int64 update_time = 0;
	int64 find_time = 0;

	//shared_ptr<cc::Blob> blob;
	string str_value;
	vector<string> str_array;

	//chart数据完整存在c++中，每次python获取的时候，返回插值后的结果，保证数据量小
	map<string, vector<ChartItem> > chartItems;
	Mat image;
};

#define _this	(env->context)

struct ThreadContext {
	volatile bool postTransactional = false;
};

struct Cmd {
	string name;
	string configJSON;
	shared_ptr<string> output;
	shared_ptr<std::condition_variable> ccv;
};

static vector<ChartItem> interpolation(const vector<ChartItem>& data, int bins) {

	vector<ChartItem> output;
	if (data.empty()) return output;

	//去掉一个数据和一个bin的目的是把最后一个数据加进去（不被插值）
	size_t size = data.size() - 1;
	bins = bins - 1;

	if (size < bins || bins < 1)
		return data;

	int step = size / bins;
	for (int i = 0; i < bins; ++i) {
		int begin = i * step;
		int end = std::min(begin + step, (int)size);

		double meanValue = 0;
		for (int j = begin; j < end; ++j) {
			auto& item = data[j];
			meanValue += item.value;
		}

		int num = (end - begin) + 1;
		if (num > 0) 
			meanValue /= num;

		int centerInd = (begin + end) / 2;
		int meanIter = data[centerInd].iter;
		output.push_back(ChartItem(meanIter, meanValue));
	}
	output.push_back(data.back());
	return output;
}

class Job : public Base{

public:
	Job(PyObject* self) : self_(self){

		env_.context = this;
		env_.monster_postBlob = postBlob;
		env_.monster_postChartValue = postChartValue;
		env_.monster_postImage = postImage;
		env_.monster_postString = postString;
		env_.monster_postStatus = postStatus;
		env_.monster_postBegin = postBegin;
		env_.monster_postEnd = postEnd;
		env_.monster_systemConfig = systemConfig;
	}
	
	static std::shared_ptr<char> systemConfig(Environment* env, const char* configJSON) {
		//必须在python上下文里面执行，否则crash
		//return boost::python::call_method<const char*>(_this->self_, "registerNotify", boost::python::str(configJSON));
		std::unique_lock<mutex> lk(queue_mtx_);
		Cmd cmd;
		cmd.name = "systemConfig";
		cmd.configJSON = configJSON;
		cmd.ccv.reset(new condition_variable());
		cmd.output.reset(new string());
		cmdqueue_.push(cmd);
		cmd.ccv->wait(lk);
		return toSharedString(*cmd.output.get());
	}

	static void postImage(Environment* env, const char* name, const Mat& image) {
		std::unique_lock<mutex> l(_this->postlock_);

		auto& store_blob = _this->postmap_[name];
		image.copyTo(store_blob.image);
		store_blob.type = PostInfoType_Image;
		store_blob.update_time = getTickCount();
	}

	static void postChartValue(Environment* env, const char* name, const char* itemName, int iter, float value) {
		std::unique_lock<mutex> l(_this->postlock_);

		auto& store_blob = _this->postmap_[name];
		store_blob.type = PostInfoType_ChartValue;
		store_blob.update_time = getTickCount();
		store_blob.chartItems[itemName].push_back(ChartItem(iter, value));
	}

	static void postStatus(Environment* env, const char* name, const char* str) {
		std::unique_lock<mutex> l(_this->postlock_);

		auto& store_blob = _this->postmap_[name];
		store_blob.str_value = str;
		store_blob.type = PostInfoType_Status;
		store_blob.update_time = getTickCount();
	}

	static void postString(Environment* env, const char* name, const char* str) {
		std::unique_lock<mutex> l(_this->postlock_);

		auto& store_blob = _this->postmap_[name];
		store_blob.str_array.push_back(str);
		store_blob.type = PostInfoType_String;
		store_blob.update_time = getTickCount();
	}

	static void postBlob(Environment* env, const char* name, const Mat& blob){
		std::unique_lock<mutex> l(_this->postlock_);
		
		auto& store_blob = _this->postmap_[name];
		store_blob.type = PostInfoType_Blob;
		blob.copyTo(store_blob.image);
		store_blob.update_time = getTickCount();
	}

	static void postBegin(Environment* env) {
		postTransactional_ = true;
	}

	static void postEnd(Environment* env) {
		postTransactional_ = false;
	}

	bool hasAnyUpdate() {

		if (postTransactional_)
			return false;

		std::unique_lock<mutex> l(postlock_);
		for (auto& item : postmap_) {
			if (item.second.find_time < item.second.update_time)
				return true;
		}
		return false;
	}

	string notify(const string& eventType, const string& message) {

		if (env_.local_notify) 
			return sharedStringExtract(env_.local_notify(eventType.c_str(), message.c_str()));
		return "";
	}

	boost::python::dict getAllStringUpdate(){

		std::unique_lock<mutex> l(postlock_);
		boost::python::dict dic;

		for (auto& mapitem : postmap_) {
			auto& item = mapitem.second;
			boost::python::dict subitem;

			if (item.type == PostInfoType_String) {
				auto name = ToPythonUTF(mapitem.first);
				boost::python::list array_value;

				for (int strind = 0; strind < item.str_array.size(); ++strind)
					array_value.append(ToPythonUTF(item.str_array[strind]));

				dic[name] = array_value;
			}
		}
		return dic;
	}

	boost::python::dict getUpdate() {

		//如果在事务中，不允许获取更新，目的是保证一致性
		boost::python::dict dic;
		if (postTransactional_)
			return dic;

		std::unique_lock<mutex> l(postlock_);
		for (auto& mapitem : postmap_) {

			auto& item = mapitem.second;
			if (item.find_time < item.update_time) {
				boost::python::dict subitem;
				auto name = ToPythonUTF(mapitem.first);

				//状态更新不需要等待事务结束
				if (item.type == PostInfoType_Status) {
					subitem["value"] = ToPythonUTF(item.str_value);
					subitem["type"] = "status";
					dic[name] = subitem;
					item.find_time = getTickCount();
				}
				else if (item.type == PostInfoType_Blob) {
					subitem["blob"] = pyopencv_from(item.image);
					subitem["type"] = "blob";
					subitem["rows"] = item.image.rows;
					subitem["cols"] = item.image.cols;
					subitem["channels"] = item.image.channels();
					dic[name] = subitem;
					item.find_time = getTickCount();
				}
				else if (item.type == PostInfoType_ChartValue) {
					subitem["type"] = "chartValue";
					
					boost::python::dict dctsub;
					for (auto& k : item.chartItems) {
						string itemname = k.first;
						auto& chartitemArray = k.second;
						auto chartitemArrayInterp = interpolation(chartitemArray, 200);

						boost::python::list local_array;
						for (int indchartItem = 0; indchartItem < chartitemArrayInterp.size(); ++indchartItem) {
							auto& a = chartitemArrayInterp[indchartItem];
							boost::python::list value_item;
							value_item.append(a.iter);
							value_item.append(a.value);
							local_array.append(value_item);
						}
						dctsub[ToPythonUTF(itemname)] = local_array;
					}
					subitem["items"] = dctsub;
					dic[name] = subitem;
					item.find_time = getTickCount();
				}
				else if (item.type == PostInfoType_String) {
					//subitem["value"] = ;
					boost::python::list array_value;

					int start = max(0, (int)item.str_array.size() - 200);
					for (int strind = start; strind < item.str_array.size(); ++strind)
						array_value.append(ToPythonUTF(item.str_array[strind]));

					subitem["value"] = array_value;
					subitem["type"] = "string";
					dic[name] = subitem;
					item.find_time = getTickCount();
				}
				else if (item.type == PostInfoType_Image) {
					subitem["image"] = pyopencv_from(item.image);
					subitem["type"] = "image";
					subitem["rows"] = item.image.rows;
					subitem["cols"] = item.image.cols;
					subitem["channels"] = item.image.channels();
					dic[name] = subitem;
					item.find_time = getTickCount();
				}
			}
		}
		return dic;
	}

	//需要在python线程中调用的函数，通过handler来调用
	void hanlderCall() {
		
		if (cmdqueue_.empty()) return;

		std::unique_lock<mutex> lk(queue_mtx_);
		if (cmdqueue_.empty()) return;

		auto cmd = cmdqueue_.front();
		cmdqueue_.pop();
		*cmd.output.get() = 
			boost::python::call_method<string>(
				self_, 
				cmd.name.c_str(), 
				ToPythonUTF(cmd.configJSON)
			);
		cmd.ccv->notify_one();
	}

	bool loadJob(const string& execute_library_name){

		lib_.reset(new boost::dll::shared_library(execute_library_name));

		if (lib_->is_loaded()){
			if (!lib_->has("iopInit")){
				printf("not found function iopInit from %s.\n", execute_library_name.c_str());
				return false;
			}

			auto& symbol = lib_->get<void (Environment*)>("iopInit");
			symbol(&env_);
			return true;
		}
		else{
			lib_.reset();
			printf("load %s fail.\n", execute_library_name.c_str());
		}
		return false;
	}

	void releaseMainThread() {
		if (main_t_ && main_t_->joinable()) {
			main_t_->join();
		}
		main_t_.reset();
	}

	void stopTrain() {

		if (env_.local_stopTrain){
			env_.local_stopTrain();
		}
		releaseMainThread();
	}

	void train() {

		releaseMainThread();
		if (env_.local_train) {
			main_t_.reset(new thread(env_.local_train));
		}
	}

private:
	static std::queue<Cmd> cmdqueue_;
	static std::mutex queue_mtx_;
	map<string, PostInfo> postmap_;
	mutex postlock_;
	shared_ptr<thread> main_t_;
	static volatile bool postTransactional_;

	boost::shared_ptr<boost::dll::shared_library> lib_;
	PyObject* const self_;
	Environment env_;
};

std::queue<Cmd> Job::cmdqueue_;
std::mutex Job::queue_mtx_;
volatile bool Job::postTransactional_ = false;

BOOST_PYTHON_MODULE(VisualizerMiddleware)
{
	using namespace boost::python;

	{if (_import_array() < 0) { PyErr_Print(); PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import"); return; } };
	class_<Base, Job, boost::noncopyable>("Job")
		.def("loadJob", &Job::loadJob)
		.def("train", &Job::train)
		.def("getUpdate", &Job::getUpdate)
		.def("hasAnyUpdate", &Job::hasAnyUpdate)
		.def("notify", &Job::notify)
		.def("hanlderCall", &Job::hanlderCall)
		.def("stopTrain", &Job::stopTrain)
		.def("getAllStringUpdate", &Job::getAllStringUpdate);
}