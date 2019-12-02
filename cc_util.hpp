
#ifndef DETECT_UTILS_HPP
#define DETECT_UTILS_HPP

#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif

#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace ccutil{

#define ORG_Center			-1000000
#define Assert(op)		ccutil::__assert_func((op), __FILE__, __LINE__, __FUNCTION__, #op)
#define INFO(...)	ccutil::__log_func(__FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)

	using std::string;
	using std::vector;
	using std::map;

	struct Timer{
		double tick = 0;

		Timer();
		void begin();
		double end();	//ms
	};

	struct GenNumber{
		volatile int next_ = 0;

		GenNumber(){}
		GenNumber(int start) :next_(start){}
		int next();
	};

	struct BBox{

		float score = 0; float x = 0; float y = 0; float r = 0; float b = 0;
		string filename; string classname; int label = 0;

		BBox();
		BBox(const cv::Rect& other);
		BBox(float x, float y, float r, float b, float score = 0, const string& filename = "", const string& classname = "");
		float width() const;
		float height() const;
		float area() const;
		float iouOf(const BBox& b) const;
		float iouMinOf(const BBox& b) const;
		BBox mergeOf(const BBox& b) const;
		BBox expand(float ratio, const cv::Size& limit = cv::Size()) const;
		cv::Rect box() const;
		operator cv::Rect() const;
		cv::Point tl() const;
		cv::Point rb() const;
		BBox offset(const cv::Point& position) const;
		BBox transfrom(cv::Size sourceSize, cv::Size dstSize);
	};

	inline vector<int> range(int begin, int end, int step = 1){
		vector<int> out;
		for (int i = begin; i != end; i += step)
			out.push_back(i);
		return std::move(out);
	}

	inline vector<int> range(int end){
		vector<int> out;
		for (int i = 0; i != end; ++i)
			out.push_back(i);
		return std::move(out);
	}

	template<typename _T>
	struct __SelectElement{
		_T& ref_;
		__SelectElement(_T& v) :ref_(v){}
		
		typename _T::const_reference operator[](int pos) const{
			if (pos < 0)
				pos = (int)ref_.size() + pos;
			return ref_[pos];
		}

		typename _T::reference operator[](int pos){
			if (pos < 0)
				pos = (int)ref_.size() + pos;
			return ref_[pos];
		}

		_T operator()(int end){
			if (end < 0)
				end = (int)ref_.size() + end + 1;

			_T out;
			for (int index = 0; index != end; ++index)
				out.push_back(ref_[index]);
			return std::move(out);
		}

		_T operator()(int begin, int end, int step = 1){
			if (end < 0)
				end = (int)ref_.size() + end + 1;

			_T out;
			for (int index = begin; index != end; index += step)
				out.push_back(ref_[index]);
			return std::move(out);
		}

		vector<typename _T::value_type> operator[](const vector<int>& inds){
			_T out;
			for (auto& index : inds)
				out.push_back(ref_[index]);
			return std::move(out);
		}
	};

	//vector<int> a = { 1, 2, 3 };
	//auto val = S(a)[-1];            ->  val = 3
	//auto list = S(a)(-1);			  ->  list = {1, 2},   0:-1    -----)   0:2(exclude 2)
	//auto list = S(a)(0, 3)		  ->  list = {1, 2, 3} 0:3     -----)   0:3(exclude 3)
	//auto list = S(a)(0, 3, 1)		  ->  list = {1, 2, 3} 0:3     -----)   0:3(exclude 3)

	//string text = "liuanqi____abc";
	//auto val = S(text)(0, -3);      ->  val = "liuanqi____"
	template<typename _T>
	__SelectElement<_T> S(_T& val){ return __SelectElement<_T>(val); }


	////////////////////////////////////////////////////////////////////////////////////////////////
	string dateNow();
	string timeNow();
	void setLoggerSaveDirectory(const string& loggerDirectory);
	void __assert_func(bool condition, const char* file, int line, const char* function, const char* code);
	void __log_func(const char* file, int line, const char* function, const char* fmt, ...);
	///////////////////////////////////////////////////////////////////////////////////////////////

	string format(const char* fmt, ...);

	//bbox nms
	vector<BBox> nms(vector<BBox>& objs, float iou_threshold);
	vector<BBox> nmsMinIoU(vector<BBox>& objs, float iou_threshold);

	//bbox softnms with linear method
	vector<BBox> softnms(vector<BBox>& B, float iou_threshold);

	//string operator
	vector<string> split(const string& str, const string& spstr);
	vector<int> splitInt(const string& str, const string& spstr);
	vector<float> splitFloat(const string& str, const string& spstr);

	//str="abcdef"   begin="ab"   end="f"   return "cde"
	string middle(const string& str, const string& begin, const string& end);

	//load bbox from std xml file
	vector<BBox> loadxmlFromData(const string& data, int* width, int* height, const string& filter);
	vector<BBox> loadxml(const string& file, int* width = nullptr, int* height = nullptr, const string& filter = "");
	bool savexml(const string& file, int width, int height, const vector<BBox>& objs);
	bool xmlEmpty(const string& file);
	bool xmlHasObject(const string& file, const string& classes);

	vector<string> loadList(const string& listfile);
	bool saveList(const string& file, const vector<string>& list);

	//name,label,x,y,r,b
	//ӳ����������key = name, value = label,x,y,r,b
	map<string, string> loadListMap(const string& listfile);
	cv::Mat loadMatrix(FILE* file);
	bool saveMatrix(FILE* file, const cv::Mat& m);
	cv::Mat loadMatrix(const string& file);
	bool saveMatrix(const string& file, const cv::Mat& m);

	//voc
	string vocxml(const string& vocjpg);
	string vocjpg(const string& vocxml);

	//file operator
	//read file return all data
	string loadfile(const string& file);
	size_t fileSize(const string& file);
	bool savefile(const string& file, const string& data, bool mk_dirs = true);
	bool savefile(const string& file, const void* data, size_t length, bool mk_dirs = true);

	//file exists
	bool exists(const string& path);

	//filename
	//  c:/a/abc.xml
	//  include_suffix = true :   return abc.xml
	//  include_suffix = false :  return abc
	string fileName(const string& path, bool include_suffix = false);

	//  c:/abcdef   ->  return c:/
	//  c:/abcddd/  ->  return c:/abcddd/
	string directory(const string& path);

	//with
	bool beginsWith(const string& str, const string& with);
	bool endsWith(const string& str, const string& with);

	//replace suffix
	//abc.txt, xml           ->  abc.xml
	string repsuffix(const string& path, const string& newSuffix);
	string repstrFast(const string& str, const string& token, const string& value);
	string repstr(const string& str, const string& token, const string& value);

	string md5(const void* data, int length);
	string md5(const string& data);

	//find 
	vector<string> findFiles(const string& directory, const string& filter = "*", bool findDirectory = false, bool includeSubDirectory = false);
	vector<string> findFilesAndCacheList(const string& directory, const string& filter = "*", bool findDirectory = false, bool includeSubDirectory = false);

	//  a == A   ->   ignore_case=true   ->  return true
	bool alphabetEqual(char a, char b, bool ignore_case);

	//   abcdefg.pnga          *.png      > false
	//   abcdefg.png           *.png      > true
	//   abcdefg.png          a?cdefg.png > true
	bool patternMatch(const char* str, const char* matcher, bool igrnoe_case = true);

	vector<string> batchRepSuffix(const vector<string>& filelist, const string& newSuffix);

	template<typename _T, typename _Op>
	void each(_T& list, const _Op& op){ for (auto& item : list) op(item); }

	template<typename _T, typename _Op>
	void each_index(_T& list, const _Op& op){ for (int i = 0; i < list.size(); ++i) op(i, list[i]); }

#define cceach(_lst, _op)	do{ for (int _index = 0; _index < _lst.size(); ++_index) { auto& _item = _lst[_index]; {_op;}} }while(0);

	template<typename _T>
	void shuffle(_T& var){ std::random_shuffle(var.begin(), var.end()); }

	template<typename _T>
	void sameSizeArray(_T& a, _T& b){
		if (a.size() > b.size())
			a.erase(a.begin() + b.size(), a.end());
		else if (a.size() < b.size())
			b.erase(b.begin() + a.size(), b.end());
	}
	
	//make dir and subdirs
	bool mkdirs(const string& path);
	bool mkdir(const string& path);
	bool copyTo(const string& src, const string& dst);
	bool moveTo(const string& src, const string& dst);
	bool rmtree(const string& directory, bool ignore_fail = false);
	bool remove(const string& file);
	FILE* fopen_mkdirs(const string& path, const string& mode);

	void setRandomSeed(int seed);

	//���������صĲ�����high��[low, high)
	float randrf(float low, float high);
	cv::Rect randbox(cv::Size size, cv::Size limit);

	//�������صģ�����high��[low, high]
	int randr(int low, int high);
	int randr(int high);
	int randr_exclude(int mi, int mx, int exclude);

	//���صĲ�����end��[low, end)
	vector<int> seque(int begin, int end);
	vector<int> seque(int end);
	vector<int> shuffleSeque(int begin, int end);
	vector<int> shuffleSeque(int end);

	template<typename _T>
	_T& randitem(vector<_T>& arr){
		int n = randr(0, (int)arr.size() - 1);
		return arr[n];
	}

	template<typename _T>
	const _T& randitem(const vector<_T>& arr){
		int n = randr(0, (int)arr.size() - 1);
		return arr[n];
	}

	template<typename _T>
	void repeat(vector<_T>& list, int count, bool requirement_count_matched = false){

		int oldsize = list.size();
		int n = count - list.size();
		for (int i = 0; i < n; ++i)
			list.push_back(list[i % oldsize]);

		if (requirement_count_matched){
			if (list.size() > count){
				shuffle(list);
				list.erase(list.begin() + count, list.end());
			}
		}
	}

	vector<cv::Scalar> randColors(int size);
	cv::Scalar randColor(int label, int size = 80);
	const vector<string>& vocLabels();
	int vocLabel(const string& name);

	template<typename _TArray>
	inline _TArray& appendArray(_TArray& array, _TArray& other){
		array.insert(array.end(), other.begin(), other.end());
		return array;
	}

	template<typename _T>
	float l2distance(const _T& a, const _T& b){
		return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
	}

	//����32λ�Ĵ�д��ĸ��uuid
	string uuid();

	class BinIO{
	public:
		enum Head{
			FileIO = 0,
			MemoryRead = 1,
			MemoryWrite = 2
		};

		BinIO() { openMemoryWrite(); }
		BinIO(const void* ptr, int memoryLength = -1) { openMemoryRead(ptr, memoryLength); }
		BinIO(const string& file, const string& mode, bool mkparents = false);
		virtual ~BinIO();
		bool opened();
		bool openFile(const string& file, const string& mode, bool mkparents = false);
		bool openMemoryRead(const void* ptr, int memoryLength = -1);
		void openMemoryWrite();
		const string& writedMemory() { return memoryWrite_; }
		void close();
		int write(const void* pdata, size_t length);
		int writeData(const string& data);
		int read(void* pdata, size_t length);
		string readData(int numBytes);
		int readInt();
		float readFloat();

		BinIO& operator >> (string& value);
		BinIO& operator << (const string& value);
		BinIO& operator << (const char* value);

		BinIO& operator >> (cv::Mat& value);
		BinIO& operator << (const cv::Mat& value);

		template<typename _T>
		BinIO& operator >> (vector<_T>& value){
			int length = 0;
			(*this) >> length;

			value.resize(length);
			read(value.data(), length * sizeof(_T));
			return *this;
		}

		template<typename _T>
		BinIO& operator << (const vector<_T>& value){
			(*this) << (int)value.size();
			write(value.data(), sizeof(_T) * value.size());
			return *this;
		}

		template<typename _T>
		BinIO& operator >> (_T& value){
			int rlen = read(&value, sizeof(_T));
			Assert(rlen == sizeof(value));
			return *this;
		}

		template<typename _T>
		BinIO& operator << (const _T& value){
			int wlen = write(&value, sizeof(_T));
			Assert(wlen == sizeof(value));
			return *this;
		}

	private:
		FILE* f_ = nullptr;
		string memoryWrite_;
		const char* memoryRead_ = nullptr;
		int memoryCursor_ = 0;
		int memoryLength_ = -1;
		Head flag_ = MemoryWrite;
	};

	//////////////////////////////////////////////////////////////////////////
#if defined(U_OS_WINDOWS)
	void drawText(cv::Mat& dst, const std::string& str, cv::Point org, cv::Scalar color, int fontSize = 12, bool bold = false, bool italic = false, bool underline = false);
#endif


	///////////////////////////////////////////////////////////////////////
	class FileCache{

	public:
		//Ϊ0ʱ����cache��Ϊ-1ʱ�����ж�cache
		FileCache(int maxCacheSize = -1);
		string loadfile(const string& file);
		vector<BBox> loadxml(const string& file, int* width, int* height, const string& filter);
		cv::Mat loadimage(const string& file, int color = 1);

	private:
		bool hitFile(const string& file);

	private:
		std::mutex lock_;
		int maxCacheSize_ = 0;
		map<string, string> hits_;
		vector<string> cacheNames_;
	};
};

#endif //DETECT_UTILS_HPP