
#include "cc_util.hpp"
#include <fstream>
#include <stack>
#include <algorithm>
#include <mutex>
#include <memory>

#if defined(U_OS_WINDOWS)
#	define HAS_UUID
#	include <Windows.h>
#	include <Shlwapi.h>
#	pragma comment(lib, "shlwapi.lib")
#   pragma comment(lib, "ole32.lib")
#   pragma comment(lib, "gdi32.lib")
#	undef min
#	undef max
#endif

#if defined(U_OS_LINUX)
#	include <sys/io.h>
#	include <dirent.h>
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <unistd.h>
#   include <stdarg.h>
#if defined(HAS_UUID)
//sudo apt-get install uuid-dev
#   include <uuid/uuid.h>
#endif

#	define strtok_s  strtok_r
#endif

namespace ccutil{

	using namespace std;

	static shared_ptr<cv::RNG>& getRandom(){

		static shared_ptr<cv::RNG> g_random;
		static volatile bool g_isinited = false;

		if (g_isinited)
			return g_random;

		g_isinited = true;
		g_random.reset(new cv::RNG(25));
		return g_random;
	}

	Timer::Timer(){
		begin();
	}

	void Timer::begin(){
		tick = cv::getTickCount();
	}

	double Timer::end(){	//ms

		double fee = (cv::getTickCount() - tick) / cv::getTickFrequency() * 1000;

		begin();
		return fee;
	}

	int GenNumber::next(){
		return next_++;
	}

	BBox::BBox(const cv::Rect& other){
		x = other.x;
		y = other.y;
		r = other.x + other.width - 1;
		b = other.y + other.height - 1;
		score = 0;
	}

	BBox BBox::offset(const cv::Point& position) const{

		BBox r(*this);
		r.x += position.x;
		r.y += position.y;
		r.r += position.x;
		r.b += position.y;
		return r;
	}

	cv::Point BBox::tl() const{
		return cv::Point(x, y);
	}

	cv::Point BBox::rb() const{
		return cv::Point(r, b);
	}

	float BBox::width() const{
		return (r - x) + 1;
	}

	float BBox::height() const{
		return (b - y) + 1;
	}

	float BBox::area() const{
		return width() * height();
	}

	BBox::BBox(){
	}

	BBox::BBox(float x, float y, float r, float b, float score, const string& filename, const string& classname) :
		x(x), y(y), r(r), b(b), score(score), filename(filename), classname(classname){
	}

	BBox::operator cv::Rect() const{
		return box();
	}

	cv::Rect BBox::box() const{
		return cv::Rect(x, y, width(), height());
	}

	BBox BBox::transfrom(cv::Size sourceSize, cv::Size dstSize){

		auto& a = *this;
		BBox out;
		out.x = a.x / (float)sourceSize.width * dstSize.width;
		out.y = a.y / (float)sourceSize.height * dstSize.height;
		out.r = a.r / (float)sourceSize.width * dstSize.width;
		out.b = a.b / (float)sourceSize.height * dstSize.height;
		return out;
	}

	BBox BBox::mergeOf(const BBox& b) const{
		auto& a = *this;
		BBox out;
		out.x = min(a.x, b.x);
		out.y = min(a.y, b.y);
		out.r = max(a.r, b.r);
		out.b = max(a.b, b.b);
		return out;
	}

	BBox BBox::expand(float ratio, const cv::Size& limit) const{

		BBox expandbox;
		expandbox.x = (int)(this->x - this->width() * ratio);
		expandbox.y = (int)(this->y - this->height() * ratio);
		expandbox.r = (int)(this->r + this->width() * ratio);
		expandbox.b = (int)(this->b + this->height() * ratio);

		if (limit.area() > 0)
			expandbox = expandbox.box() & cv::Rect(0, 0, limit.width, limit.height);
		return expandbox;
	}

	float BBox::iouMinOf(const BBox& b) const{
		auto& a = *this;
		float xmax = max(a.x, b.x);
		float ymax = max(a.y, b.y);
		float xmin = min(a.r, b.r);
		float ymin = min(a.b, b.b);
		float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
		float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
		float iou = uw * uh;
		return iou / min(a.area(), b.area());
	}

	float BBox::iouOf(const BBox& b) const{

		auto& a = *this;
		float xmax = max(a.x, b.x);
		float ymax = max(a.y, b.y);
		float xmin = min(a.r, b.r);
		float ymin = min(a.b, b.b);
		float uw = (xmin - xmax + 1 > 0) ? (xmin - xmax + 1) : 0;
		float uh = (ymin - ymax + 1 > 0) ? (ymin - ymax + 1) : 0;
		float iou = uw * uh;
		return iou / (a.area() + b.area() - iou);
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	bool alphabetEqual(char a, char b, bool ignore_case){
		if (ignore_case){
			a = a > 'a' && a < 'z' ? a - 'a' + 'A' : a;
			b = b > 'a' && b < 'z' ? b - 'a' + 'A' : b;
		}
		return a == b;
	}

	static bool patternMatchBody(const char* str, const char* matcher, bool igrnoe_case){
		//   abcdefg.pnga          *.png      > false
		//   abcdefg.png           *.png      > true
		//   abcdefg.png          a?cdefg.png > true

		if (!matcher || !*matcher || !str || !*str) return false;

		const char* ptr_matcher = matcher;
		while (*str){
			if (*ptr_matcher == '?'){
				ptr_matcher++;
			}
			else if (*ptr_matcher == '*'){
				if (*(ptr_matcher + 1)){
					if (patternMatchBody(str, ptr_matcher + 1, igrnoe_case))
						return true;
				}
				else{
					return true;
				}
			}
			else if (!alphabetEqual(*ptr_matcher, *str, igrnoe_case)){
				return false;
			}
			else{
				if (*ptr_matcher)
					ptr_matcher++;
				else
					return false;
			}
			str++;
		}

		while (*ptr_matcher){
			if (*ptr_matcher != '*')
				return false;
			ptr_matcher++;
		}
		return true;
	}

	bool patternMatch(const char* str, const char* matcher, bool igrnoe_case){
		//   abcdefg.pnga          *.png      > false
		//   abcdefg.png           *.png      > true
		//   abcdefg.png          a?cdefg.png > true

		if (!matcher || !*matcher || !str || !*str) return false;

		char filter[500];
		strcpy(filter, matcher);

		vector<const char*> arr;
		char* ptr_str = filter;
		char* ptr_prev_str = ptr_str;
		while (*ptr_str){
			if (*ptr_str == ';'){
				*ptr_str = 0;
				arr.push_back(ptr_prev_str);
				ptr_prev_str = ptr_str + 1;
			}
			ptr_str++;
		}

		if (*ptr_prev_str)
			arr.push_back(ptr_prev_str);

		for (int i = 0; i < arr.size(); ++i){
			if (patternMatchBody(str, arr[i], igrnoe_case))
				return true;
		}
		return false;
	}

	vector<string> split(const string& str, const std::string& spstr){
		char* s = (char*)str.c_str();
		char* context = nullptr;
		char* token = strtok_s(s, spstr.c_str(), &context);
		vector<string> out;
		while (token){
			out.push_back(token);
			token = strtok_s(0, spstr.c_str(), &context);
		}
		return out;
	}

	vector<int> splitInt(const string& str, const string& spstr){
		char* s = (char*)str.c_str();
		char* context = nullptr;
		char* token = strtok_s(s, spstr.c_str(), &context);
		vector<int> out;
		while (token){
			out.push_back(atoi(token));
			token = strtok_s(0, spstr.c_str(), &context);
		}
		return out;
	}

	vector<float> splitFloat(const string& str, const string& spstr){
		char* s = (char*)str.c_str();
		char* context = nullptr;
		char* token = strtok_s(s, spstr.c_str(), &context);
		vector<float> out;
		while (token){
			out.push_back(atof(token));
			token = strtok_s(0, spstr.c_str(), &context);
		}
		return out;
	}

	vector<string> findFilesAndCacheList(const string& directory, const string& filter, bool findDirectory, bool includeSubDirectory){

		string path = directory;
		if (path.empty()) return vector<string>();
#ifdef U_OS_WINDOWS
		if (path.back() == '/' || path.back() == '\\'){
#endif

#ifdef U_OS_LINUX
			if (path.back() == '/'){
#endif
			path.pop_back();
		};

		string dirname = fileName(path);
		string findEncode = md5(directory + ";" + filter + ";" + (findDirectory ? "yes" : "no") + ";" + (includeSubDirectory ? "yes" : "no"));
		string cacheFile = dirname + "_" + findEncode + ".list.txt";

		vector<string> files;
		if (!ccutil::exists(cacheFile)){
			files = ccutil::findFiles(directory, filter, findDirectory, includeSubDirectory);
			ccutil::shuffle(files);
			ccutil::saveList(cacheFile, files);
		}
		else{
			files = ccutil::loadList(cacheFile);
		}
		return files;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef U_OS_WINDOWS
	vector<string> findFiles(const string& directory, const string& filter, bool findDirectory, bool includeSubDirectory){
		
		string realpath = directory;
		if (realpath.empty())
			realpath = "./";

		char backchar = realpath.back();
		if (backchar != '\\' && backchar != '/')
			realpath += "/";

		vector<string> out;
		_WIN32_FIND_DATAA find_data;
		stack<string> ps;
		ps.push(realpath);

		while (!ps.empty())
		{
			string search_path = ps.top();
			ps.pop();

			HANDLE hFind = FindFirstFileA((search_path + "*").c_str(), &find_data);
			if (hFind != INVALID_HANDLE_VALUE){
				do{
					if (strcmp(find_data.cFileName, ".") == 0 || strcmp(find_data.cFileName, "..") == 0)
						continue;

					if (!findDirectory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != FILE_ATTRIBUTE_DIRECTORY ||
						findDirectory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY){
						if (PathMatchSpecA(find_data.cFileName, filter.c_str()))
							out.push_back(search_path + find_data.cFileName);
					}

					if (includeSubDirectory && (find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == FILE_ATTRIBUTE_DIRECTORY)
						ps.push(search_path + find_data.cFileName + "/");

				} while (FindNextFileA(hFind, &find_data));
				FindClose(hFind);
			}
		}
		return out;
	}
#endif

#ifdef U_OS_LINUX
	vector<string> findFiles(const string& directory, const string& filter, bool findDirectory, bool includeSubDirectory)
	{
		string realpath = directory;
		if (realpath.empty())
			realpath = "./";

		char backchar = realpath.back();
		if (backchar != '\\' && backchar != '/')
			realpath += "/";

		struct dirent* fileinfo;
		DIR* handle;
		stack<string> ps;
		vector<string> out;
		ps.push(realpath);

		while (!ps.empty())
		{
			string search_path = ps.top();
			ps.pop();

			handle = opendir(search_path.c_str());
			if (handle != 0)
			{
				while (fileinfo = readdir(handle))
				{
					struct stat file_stat;
					if (strcmp(fileinfo->d_name, ".") == 0 || strcmp(fileinfo->d_name, "..") == 0)
						continue;

					if (lstat((search_path + fileinfo->d_name).c_str(), &file_stat) < 0)
						continue;

					if (!findDirectory && !S_ISDIR(file_stat.st_mode) ||
						findDirectory && S_ISDIR(file_stat.st_mode))
					{
						if (patternMatch(fileinfo->d_name, filter.c_str()))
							out.push_back(search_path + fileinfo->d_name);
					}

					if (includeSubDirectory && S_ISDIR(file_stat.st_mode))
						ps.push(search_path + fileinfo->d_name + "/");
				}
				closedir(handle);
			}
		}
		return out;
	}
#endif
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	string loadfile(const string& file){

		ifstream in(file, ios::in | ios::binary);
		if (!in.is_open())
			return "";

		in.seekg(0, ios::end);
		size_t length = in.tellg();

		string data;
		if (length > 0){
			in.seekg(0, ios::beg);
			data.resize(length);

			in.read(&data[0], length);
		}
		in.close();
		return data;
	}

	size_t fileSize(const string& file){

#if defined(U_OS_LINUX)
		struct stat st;
		stat(file.c_str(), &st);
		return st.st_size;
#elif defined(U_OS_WINDOWS)
		WIN32_FIND_DATAA find_data;
		HANDLE hFind = FindFirstFileA(file.c_str(), &find_data);
		if (hFind == INVALID_HANDLE_VALUE)
			return 0;

		FindClose(hFind);
		return (uint64_t)find_data.nFileSizeLow | ((uint64_t)find_data.nFileSizeHigh << 32);
#endif
	}

	bool savefile(const string& file, const string& data, bool mk_dirs){
		return savefile(file, data.data(), data.size(), mk_dirs);
	}

	string format(const char* fmt, ...) {
		va_list vl;
		va_start(vl, fmt);
		char buffer[10000];
		vsprintf(buffer, fmt, vl);
		return buffer;
	}

	bool savefile(const string& file, const void* data, size_t length, bool mk_dirs){

		if (mk_dirs){
			int p = (int)file.rfind('/');

#ifdef U_OS_WINDOWS
			int e = (int)file.rfind('\\');
			p = max(p, e);
#endif
			if (p != -1){
				if (!mkdirs(file.substr(0, p)))
					return false;
			}
		}

		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		if (data && length > 0){
			if (fwrite(data, 1, length, f) != length){
				fclose(f);
				return false;
			}
		}
		fclose(f);
		return true;
	}

	string middle(const string& str, const string& begin, const string& end){

		auto p = str.find(begin);
		if (p == string::npos) return "";
		p += begin.length();

		auto e = str.find(end, p);
		if (e == string::npos) return "";

		return str.substr(p, e - p);
	}

	bool savexml(const string& file, int width, int height, const vector<BBox>& objs){
		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		fprintf(f, "<annotation>\n<size><width>%d</width><height>%d</height></size>\n", width, height);
		for (int i = 0; i < objs.size(); ++i){
			auto& obj = objs[i];
			fprintf(f,
				"<object>"
				"<name>%s</name>"
				"<bndbox>"
				"<xmin>%d</xmin>"
				"<ymin>%d</ymin>"
				"<xmax>%d</xmax>"
				"<ymax>%d</ymax>"
				"</bndbox>"
				"</object>\n", obj.classname.c_str(), (int)obj.x, (int)obj.y, (int)obj.r, (int)obj.b);
		}
		fprintf(f, "</annotation>");
		fclose(f);
		return true;
	}

	vector<BBox> loadxmlFromData(const string& data, int* width, int* height, const string& filter){

		vector<BBox> output;
		if (data.empty())
			return output;

		if (width)
			*width = atoi(middle(data, "<width>", "</width>").c_str());

		if (height)
			*height = atoi(middle(data, "<height>", "</height>").c_str());

		string begin_token = "<object>";
		string end_token = "</object>";
		int p = data.find(begin_token);
		if (p == -1)
			return output;

		bool ignoreFilter = filter.empty() || filter == "*";
		int e = data.find(end_token, p + begin_token.length());
		while (e != -1){

			string part = data.substr(p, e - p);
			string name = middle(part, "<name>", "</name>");

			//filter.empty, not use filter
			//filter == *, not use filter
			//filter == *xxx*, use match
			if (ignoreFilter || patternMatch(name.c_str(), filter.c_str())){
				float xmin = atof(middle(part, "<xmin>", "</xmin>").c_str());
				float ymin = atof(middle(part, "<ymin>", "</ymin>").c_str());
				float xmax = atof(middle(part, "<xmax>", "</xmax>").c_str());
				float ymax = atof(middle(part, "<ymax>", "</ymax>").c_str());

				BBox box;
				box.x = xmin;
				box.y = ymin;
				box.r = xmax;
				box.b = ymax;
				box.classname = name;
				output.push_back(box);
				//output.emplace_back(xmin, ymin, xmax, ymax, "", name);
			}

			e += end_token.length();
			p = data.find(begin_token, e);
			if (p == -1) break;

			e = data.find(end_token, p + begin_token.length());
		}
		return output;
	}

	vector<BBox> loadxml(const string& file, int* width, int* height, const string& filter){
		return loadxmlFromData(loadfile(file), width, height, filter);
	}

	bool xmlEmpty(const string& file){
		return loadxml(file).empty();
	}

	bool xmlHasObject(const string& file, const string& classes){
		auto objs = loadxml(file);
		for (int i = 0; i < objs.size(); ++i){
			if (objs[i].classname == classes)
				return true;
		}
		return false;
	}

	bool exists(const string& path){

#ifdef U_OS_WINDOWS
		return ::PathFileExistsA(path.c_str());
#elif defined(U_OS_LINUX)
		return access(path.c_str(), R_OK) == 0;
#endif
	}

	map<string, string> loadListMap(const string& listfile){

		auto list = loadList(listfile);
		map<string, string> mapper;
		if (list.empty())
			return mapper;

		string key;
		string value;
		for (int i = 0; i < list.size(); ++i){
			auto& line = list[i];
			int p = line.find(',');
			
			if (p == -1){
				key = line;
				value = "";
			}
			else{
				key = line.substr(0, p);
				value = line.substr(p + 1);
			}

			if (mapper.find(key) != mapper.end()){
				printf("repeat key: %s, existsValue: %s, newValue: %s\n", key.c_str(), mapper[key].c_str(), value.c_str());
			}
			mapper[key] = value;
		}
		return mapper;
	}

	vector<string> loadList(const string& listfile){

		vector<string> lines;
		string data = loadfile(listfile);

		if (data.empty())
			return lines;

		char* ptr = (char*)&data[0];
		char* prev = ptr;
		string line;

		while (true){
			if (*ptr == '\n' || *ptr == 0){
				int length = ptr - prev;

				if (length > 0){
					if (length == 1){
						if (*prev == '\r')
							length = 0;
					}
					else {
						if (prev[length - 1] == '\r')
							length--;
					}
				}

				if (length > 0){
					line.assign(prev, length);
					lines.push_back(line);
				}

				if (*ptr == 0)
					break;

				prev = ptr + 1;
			}
			ptr++;
		}
		return lines;
	}

	string directory(const string& path){

		if (path.empty())
			return "";

		int p = path.rfind('/');

#ifdef U_OS_WINDOWS
		int e = path.rfind('\\');
		p = max(p, e);
#endif
		return path.substr(0, p + 1);
	}

	bool saveList(const string& file, const vector<string>& list){

		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		for (int i = 0; i < list.size(); ++i){

			auto& item = list[i];
			if (i < (int)list.size() - 1){
				fprintf(f, "%s\n", item.c_str());
			}
			else{
				fprintf(f, "%s", item.c_str());
			}
		}

		fclose(f);
		return true;
	}

	bool beginsWith(const string& str, const string& with){

		if (str.length() < with.length())
			return false;
		return strncmp(str.c_str(), with.c_str(), with.length()) == 0;
	}

	bool endsWith(const string& str, const string& with){

		if (str.length() < with.length())
			return false;

		return strncmp(str.c_str() + str.length() - with.length(), with.c_str(), with.length()) == 0;
	}

	string repstrFast(const string& str, const string& token, const string& value){

		string opstr;

		if (value.length() > token.length()){
			float numToken = str.size() / (float)token.size();
			float newTokenLength = value.size() * numToken;
			opstr.resize(newTokenLength);
		}
		else{
			opstr.resize(str.size());
		}

		char* dest = &opstr[0];
		const char* src = str.c_str();
		string::size_type pos = 0;
		string::size_type prev = 0;
		size_t token_length = token.length();
		size_t value_length = value.length();
		const char* value_ptr = value.c_str();
		bool keep = true;

		do{
			pos = str.find(token, pos);
			if (pos == string::npos){
				keep = false;
				pos = str.length();
			}

			size_t copy_length = pos - prev;
			memcpy(dest, src + prev, copy_length);
			dest += copy_length;
			
			if (keep){
				pos += token_length;
				prev = pos;
				memcpy(dest, value_ptr, value_length);
				dest += value_length;
			}
		} while (keep);

		size_t valid_size = dest - &opstr[0];
		opstr.resize(valid_size);
		return opstr;
	}

	string repstr(const string& str_, const string& token, const string& value){

		string str = str_;
		string::size_type pos = 0;
		string::size_type tokenlen = token.size();
		string::size_type vallen = value.size();

		while ((pos = str.find(token, pos)) != string::npos){
			str.replace(pos, tokenlen, value);
			pos += vallen;
		}
		return str;
	}

	string vocxml(const string& vocjpg){
		return repsuffix(repstr(vocjpg, "JPEGImages", "Annotations"), "xml");
	}

	string vocjpg(const string& vocxml){
		return repsuffix(repstr(vocxml, "Annotations", "JPEGImages"), "jpg");
	}

	string repsuffix(const string& path, const string& newSuffix){

		int p = path.rfind('.');
		if (p == -1)
			return path + "." + newSuffix;

		return path.substr(0, p + 1) + newSuffix;
	}

	vector<string> batchRepSuffix(const vector<string>& filelist, const string& newSuffix){
		
		vector<string> newlist = filelist;
		auto lambda = [&](string& file){file = repsuffix(file, newSuffix); };
		each(newlist, lambda);
		return newlist;
	}

	string fileName(const string& path, bool include_suffix){

		if (path.empty()) return "";

		int p = path.rfind('/');

#ifdef U_OS_WINDOWS
		int e = path.rfind('\\');
		p = max(p, e);
#endif
		p += 1;

		//include suffix
		if (include_suffix)
			return path.substr(p);

		int u = path.rfind('.');
		if (u == -1)
			return path.substr(p);

		if (u <= p) u = path.size();
		return path.substr(p, u - p);
	}

	vector<BBox> nms(vector<BBox>& objs, float iou_threshold){

		std::sort(objs.begin(), objs.end(), [](const BBox& a, const BBox& b){
			return a.score > b.score;
		});

		vector<BBox> out;
		vector<int> flags(objs.size());
		for (int i = 0; i < objs.size(); ++i){
			if (flags[i] == 1) continue;

			out.push_back(objs[i]);
			flags[i] = 1;
			for (int k = i + 1; k < objs.size(); ++k){
				if (flags[k] == 0){
					float iouUnion = objs[i].iouOf(objs[k]);
					if (iouUnion > iou_threshold)
						flags[k] = 1;
				}
			}
		}
		return out;
	}

	vector<BBox> nmsMinIoU(vector<BBox>& objs, float iou_threshold){

		std::sort(objs.begin(), objs.end(), [](const BBox& a, const BBox& b){
			return a.score > b.score;
		});

		vector<BBox> out;
		vector<int> flags(objs.size());
		for (int i = 0; i < objs.size(); ++i){
			if (flags[i] == 1) continue;

			out.push_back(objs[i]);
			flags[i] = 1;
			for (int k = i + 1; k < objs.size(); ++k){
				if (flags[k] == 0){
					float iouUnion = objs[i].iouMinOf(objs[k]);
					if (iouUnion > iou_threshold)
						flags[k] = 1;
				}
			}
		}
		return out;
	}

	vector<BBox> softnms(vector<BBox>& B, float iou_threshold){

		int method = 1;   //1 linear, 2 gaussian, 0 original
		float Nt = iou_threshold;
		float threshold = 0.2;
		float sigma = 0.5;

		std::sort(B.begin(), B.end(), [](const BBox& a, const BBox& b){
			return a.score > b.score;
		});
		
		vector<float> S(B.size());
		for (int i = 0; i < B.size(); ++i)
			S[i] = B[i].score;

		vector<BBox> D;
		while (!B.empty()){

			int m = 0;
			auto M = B[m];
			
			D.push_back(M);
			B.erase(B.begin() + m);
			S.erase(S.begin() + m);
			
			for (int i = (int)B.size() - 1; i >= 0; --i){

				float ov = M.iouOf(B[i]);
				float weight = 1;

				if (method == 1){ //linear
					if (ov > Nt)
						weight = 1 - ov;

				}else if (method == 2){ //gaussian
					weight = exp(-(ov * ov) / sigma);
				}
				else {
					//original nms
					if (ov > Nt)
						weight = 0;
				}
				S[i] *= weight;

				if (S[i] < threshold){
					B.erase(B.begin() + i);
					S.erase(S.begin() + i);
				}
			}
		}
		return D;
	}

	bool remove(const string& file){
#ifdef U_OS_WINDOWS
		return DeleteFileA(file.c_str());
#else
		return ::remove(file.c_str()) == 0;
#endif
	}

	bool mkdir(const string& path){
#ifdef U_OS_WINDOWS
		return CreateDirectoryA(path.c_str(), nullptr);
#else
		return ::mkdir(path.c_str(), 0755) == 0;
#endif
	}

	FILE* fopen_mkdirs(const string& path, const string& mode){

		FILE* f = fopen(path.c_str(), mode.c_str());
		if (f) return f;

		int p = path.rfind('/');

#if defined(U_OS_WINDOWS)
		int e = path.rfind('\\');
		p = std::max(p, e);
#endif
		if (p == -1)
			return nullptr;
		
		string directory = path.substr(0, p);
		if (!mkdirs(directory))
			return nullptr;

		return fopen(path.c_str(), mode.c_str());
	}

	bool moveTo(const string& src, const string& dst){
#if defined(U_OS_WINDOWS)
		return ::MoveFileA(src.c_str(), dst.c_str());
#elif defined(U_OS_LINUX)
		return rename(src.c_str(), dst.c_str()) == 0;
#endif
	}

	bool copyTo(const string& src, const string& dst){
#if defined(U_OS_WINDOWS)
		return ::CopyFileA(src.c_str(), dst.c_str(), false);
#elif defined(U_OS_LINUX)
		FILE* i = fopen(src.c_str(), "rb");
		if (!i) return false;

		FILE* o = fopen(dst.c_str(), "wb");
		if (!o){
			fclose(i);
			return false;
		}

		bool ok = true;
		char buffer[1024];
		int rlen = 0;
		while ((rlen = fread(buffer, 1, sizeof(buffer), i)) > 0){
			if (fwrite(buffer, 1, rlen, o) != rlen){
				ok = false;
				break;
			}
		}
		fclose(i);
		fclose(o);
		return ok;
#endif
	}

	bool mkdirs(const string& path){

		if (path.empty()) return false;
		if (exists(path)) return true;

		string _path = path;
		char* dir_ptr = (char*)_path.c_str();
		char* iter_ptr = dir_ptr;
		
		bool keep_going = *iter_ptr != 0;
		while (keep_going){

			if (*iter_ptr == 0)
				keep_going = false;

#ifdef U_OS_WINDOWS
			if (*iter_ptr == '/' || *iter_ptr == '\\' || *iter_ptr == 0){
#else
			if (*iter_ptr == '/' || *iter_ptr == 0){
#endif
				char old = *iter_ptr;
				*iter_ptr = 0;
				if (!exists(dir_ptr)){
					if (!mkdir(dir_ptr))
						return false;
				}
				*iter_ptr = old;
			}
			iter_ptr++;
		}
		return true;
	}

	bool rmtree(const string& directory, bool ignore_fail){

		auto files = findFiles(directory, "*", false);
		auto dirs = findFiles(directory, "*", true);

		bool success = true;
		for (int i = 0; i < files.size(); ++i){
			if (::remove(files[i].c_str()) != 0){
				success = false;

				if (!ignore_fail){
					return false;
				}
			}
		}

		dirs.insert(dirs.begin(), directory);
		for (int i = (int)dirs.size() - 1; i >= 0; --i){

#ifdef U_OS_WINDOWS
			if (!::RemoveDirectoryA(dirs[i].c_str())){
#else
			if (::rmdir(dirs[i].c_str()) != 0){
#endif
				success = false;
				if (!ignore_fail)
					return false;
			}
		}
		return success;
	}

	void setRandomSeed(int seed){
		srand(seed);
		getRandom().reset(new cv::RNG(seed));
	}

	float randrf(float low, float high){
		if (high < low) std::swap(low, high);
		return getRandom()->uniform(low, high);
	}

	cv::Rect randbox(cv::Size size, cv::Size limit){
		int x = randr(0, limit.width - size.width - 1);
		int y = randr(0, limit.height - size.height - 1);
		return cv::Rect(x, y, size.width, size.height);
	}

	int randr(int high){
		int low = 0;
		if (high < low) std::swap(low, high);
		return randr(low, high);
	}

	int randr(int low, int high){
		if (high < low) std::swap(low, high);
		return getRandom()->uniform(low, high + 1);
	}

	int randr_exclude(int mi, int mx, int exclude){
		if (mi > mx) std::swap(mi, mx);

		if (mx == mi)
			return mi;

		int sel = 0;
		do{
			sel = randr(mi, mx);
		} while (sel == exclude);
		return sel;
	}

	static cv::Scalar HSV2RGB(const float h, const float s, const float v) {
		const int h_i = static_cast<int>(h * 6);
		const float f = h * 6 - h_i;
		const float p = v * (1 - s);
		const float q = v * (1 - f*s);
		const float t = v * (1 - (1 - f) * s);
		float r, g, b;
		switch (h_i) {
		case 0:r = v; g = t; b = p;break;
		case 1:r = q; g = v; b = p;break;
		case 2:r = p; g = v; b = t;break;
		case 3:r = p; g = q; b = v;break;
		case 4:r = t; g = p; b = v;break;
		case 5:r = v; g = p; b = q;break;
		default:r = 1; g = 1; b = 1;break;}
		return cv::Scalar(r * 255, g * 255, b * 255);
	}

	vector<cv::Scalar> randColors(int size){
		vector<cv::Scalar> colors;
		cv::RNG rng(5);
		for (int i = 0; i < size; ++i)
			colors.push_back(HSV2RGB(rng.uniform(0.f, 1.f), 1, 1));
		return colors;
	}

	cv::Scalar randColor(int label, int size){
		static mutex lock_;
		static vector<cv::Scalar> colors;

		if (colors.empty()){
			std::unique_lock<mutex> l(lock_);
			if (colors.empty())
				colors = randColors(size);
		}
		return colors[label % colors.size()];
	}

	const vector<string>& vocLabels(){
		static vector<string> voclabels{
			"aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair",
			"cow", "diningtable", "dog", "horse",
			"motorbike", "person", "pottedplant",
			"sheep", "sofa", "train", "tvmonitor"
		};
		return voclabels;
	}

	vector<int> seque(int begin, int end){

		if (end < begin) std::swap(begin, end);

		int num = end - begin;
		vector<int> out(num);
		for (int i = 0; i < num; ++i)
			out[i] = i + begin;

		return out;
	}

	vector<int> seque(int end){
		return seque(0, end);
	}

	vector<int> shuffleSeque(int begin, int end){
		auto out = seque(begin, end);
		shuffle(out);
		return out;
	}

	vector<int> shuffleSeque(int end){
		return shuffleSeque(0, end);
	}

	int vocLabel(const string& name){
		static map<string, int> labelmap;
		static mutex lock_;
		if (labelmap.empty()){

			std::unique_lock<mutex> l(lock_);
			if (labelmap.empty()){
				auto labels = vocLabels();
				for (int i = 0; i < labels.size(); ++i)
					labelmap[labels[i]] = i;
			}
		}

		auto itr = labelmap.find(name);
		if (itr == labelmap.end()){
			printf("**********name[%s] not in labelmap.\n", name.c_str());
			return -1;
		}
		return itr->second;
	}

	cv::Mat loadMatrix(FILE* f){

		if (!f) return cv::Mat();
		cv::Mat matrix;
		int info[4];
		if (fread(info, 1, sizeof(info), f) != sizeof(info))
			return matrix;

		//flag must match
		//CV_Assert(info[0] == 0xCCABABCC);
		if (info[0] != 0xCCABABCC)
			return matrix;

		int dims[32] = { -1 };
		if (fread(dims, 1, info[1] * sizeof(int), f) != info[1] * sizeof(int))
			return matrix;

		matrix.create(info[1], dims, info[2]);
		bool ok = fread(matrix.data, 1, info[3], f) == info[3];
		if (!ok) matrix.release();
		return matrix;
	}

	bool saveMatrix(FILE* f, const cv::Mat& m){

		if (!f) return false;
		int total;
		cv::Mat w = m;

		if (m.isSubmatrix()){
			//������Ӿ������¡һ��
			w = m.clone();
		}
		else if (m.dims == 2){
			//�����ͼ���������ڶ������ݵĶ�ά������ô�����ڶ��룬Ҳ��Ҫcloneһ��
			if (m.step.p[1] * m.size[1] != m.step.p[0])
				w = m.clone();
		}

		total = w.size[0] * w.step.p[0];

		//dim, type
		int info[] = { (int)0xCCABABCC, m.dims, m.type(), total };
		int wCount = fwrite(info, 1, sizeof(info), f);
		if (wCount != sizeof(info))
			return false;

		fwrite(m.size, 1, sizeof(int) * m.dims, f);
		return fwrite(w.data, 1, total, f) == total;
	}

	cv::Mat loadMatrix(const string& file)
	{
		cv::Mat matrix;
		FILE* f = fopen(file.c_str(), "rb");
		if (!f) return matrix;

		matrix = loadMatrix(f);
		fclose(f);
		return matrix;
	}

	bool saveMatrix(const string& file, const cv::Mat& m){

		FILE* f = fopen(file.c_str(), "wb");
		if (!f) return false;

		bool ok = saveMatrix(f, m);
		fclose(f);
		return ok;
	}

#if defined(U_OS_LINUX)
	typedef struct _GUID {
		unsigned int Data1;
		unsigned short Data2;
		unsigned short Data3;
		unsigned char Data4[8];
	} GUID;
#endif

	//����32λ�Ĵ�д��ĸ��uuid
	string uuid(){

#if defined(HAS_UUID)

		GUID guid;
#if defined(U_OS_WINDOWS)
		CoCreateGuid(&guid);
#else
		uuid_generate(reinterpret_cast<unsigned char *>(&guid));
#endif

		char buf[33] = { 0 };
#if defined(U_OS_LINUX)
		snprintf(
#else // MSVC
		_snprintf_s(
#endif
			buf,
			sizeof(buf),
			"%08X%04X%04X%02X%02X%02X%02X%02X%02X%02X%02X",
			guid.Data1, guid.Data2, guid.Data3,
			guid.Data4[0], guid.Data4[1],
			guid.Data4[2], guid.Data4[3],
			guid.Data4[4], guid.Data4[5],
			guid.Data4[6], guid.Data4[7]);
		return std::string(buf);
#else
		throw "not implement uuid function";
		return "";
#endif
	}

	BinIO::BinIO(const string& file, const string& mode, bool mkparents){
		openFile(file, mode, mkparents);
	}

	BinIO::~BinIO(){
		close();
	}

	bool BinIO::opened(){
		if (flag_ == FileIO)
			return f_ != nullptr;
		else if (flag_ == MemoryRead)
			return memoryRead_ != nullptr;
		else if (flag_ == MemoryWrite)
			return true;
		return false;
	}

	bool BinIO::openFile(const string& file, const string& mode, bool mkparents){

		close();
		if (mode.empty())
			return false;

		string mode_ = mode;
		bool hasBinary = false;
		for (int i = 0; i < mode_.length(); ++i){
			if (mode_[i] == 'b'){
				hasBinary = true;
				break;
			}
		}

		if (!hasBinary){
			if (mode_.length() == 1){
				mode_.push_back('b');
			}
			else if (mode_.length() == 2){
				mode_.insert(mode_.begin() + 1, 'b');
			}
		}

		if (mkparents)
			f_ = fopen_mkdirs(file, mode_);
		else
			f_ = fopen(file.c_str(), mode_.c_str());
		flag_ = FileIO;
		return opened();
	}

	void BinIO::close(){

		if (flag_ == FileIO) {
			if (f_) {
				fclose(f_);
				f_ = nullptr;
			}
		}
		else if (flag_ == MemoryRead) {
			memoryRead_ = nullptr;
			memoryCursor_ = 0;
			memoryLength_ = -1;
		}
		else if (flag_ == MemoryWrite) {
			memoryWrite_.clear();
			memoryCursor_ = 0;
			memoryLength_ = -1;
		}
	}

	string BinIO::readData(int numBytes){
		string output;
		output.resize(numBytes);

		int readlen = read((void*)output.data(), output.size());
		output.resize(readlen);
		return output;
	}

	int BinIO::read(void* pdata, size_t length){

		if (flag_ == FileIO) {
			return fread(pdata, 1, length, f_);
		}
		else if (flag_ == MemoryRead) {

			if (memoryLength_ != -1) {
				
				if (memoryLength_ < memoryCursor_ + length) {
					int remain = memoryLength_ - memoryCursor_;
					if (remain > 0) {
						memcpy(pdata, memoryRead_ + memoryCursor_, remain);
						memoryCursor_ += remain;
						return remain;
					}
					else {
						return -1;
					}
				}
			}
			memcpy(pdata, memoryRead_ + memoryCursor_, length);
			memoryCursor_ += length;
			return length;
		}
		else {
			return -1;
		}
	}

	int BinIO::write(const void* pdata, size_t length){

		if (flag_ == FileIO) {
			return fwrite(pdata, 1, length, f_);
		}
		else if (flag_ == MemoryWrite) {
			memoryWrite_.append((char*)pdata, (char*)pdata + length);
			return length;
		}
		else {
			return -1;
		}
	}

	int BinIO::writeData(const string& data){
		return write(data.data(), data.size());
	}

	BinIO& BinIO::operator >> (string& value){
		//read
		int length = 0;
		(*this) >> length;
		value = readData(length);
		return *this;
	}

	int BinIO::readInt(){
		int value = 0;
		(*this) >> value;
		return value;
	}

	float BinIO::readFloat(){
		float value = 0;
		(*this) >> value;
		return value;
	}

	BinIO& BinIO::operator << (const string& value){
		//write
		(*this) << (int)value.size();
		writeData(value);
		return *this;
	}

	BinIO& BinIO::operator << (const char* value){

		int length = strlen(value);
		(*this) << (int)length;
		write(value, length);
		return *this;
	}

	BinIO& BinIO::operator >> (cv::Mat& value){

		value = loadMatrix(f_);
		return *this;
	}

	BinIO& BinIO::operator << (const cv::Mat& value){
		
		bool ok = saveMatrix(f_, value);
		Assert(ok);
		return *this;
	}

	bool BinIO::openMemoryRead(const void* ptr, int memoryLength) {
		close();

		if (!ptr) return false;
		memoryRead_ = (const char*)ptr;
		memoryCursor_ = 0;
		memoryLength_ = memoryLength;
		flag_ = MemoryRead;
		return true;
	}

	void BinIO::openMemoryWrite() {
		close();

		memoryWrite_.clear();
		memoryCursor_ = 0;
		memoryLength_ = -1;
		flag_ = MemoryWrite;
	}


	//////////////////////////////////////////////////////////////////////////
#if defined(U_OS_WINDOWS)
	void GetStringSize(HDC hDC, const char* str, int* w, int* h)
	{
		SIZE size;
		GetTextExtentPoint32A(hDC, str, strlen(str), &size);
		if (w != 0) *w = size.cx;
		if (h != 0) *h = size.cy;
	}

	void drawText(cv::Mat& _dst, const std::string& str, cv::Point org, cv::Scalar color, int fontSize, bool bold, bool italic, bool underline)
	{
		IplImage ipldst = _dst;
		IplImage* dst = &ipldst;
		if (dst == nullptr)
			return;
		
		if (dst->depth != IPL_DEPTH_8U){
			printf("drawText input image.depth != 8U\n");
			return;
		}

		if (dst->nChannels != 1 && dst->nChannels != 3){
			printf("drawText image.channels only support 1 or 3\n");
			return;
		}

		//if (box) box->x = box->y = box->width = box->height = 0;
		int x, y, r, b;
		if (org.x > dst->width || org.y > dst->height) return;

		LOGFONTA lf;
		lf.lfHeight = -fontSize;
		lf.lfWidth = 0;
		lf.lfEscapement = 0;
		lf.lfOrientation = 0;
		lf.lfWeight = bold ? FW_BOLD : FW_NORMAL;
		lf.lfItalic = italic;	//б��
		lf.lfUnderline = underline;	//�»���
		lf.lfStrikeOut = 0;
		lf.lfCharSet = DEFAULT_CHARSET;
		lf.lfOutPrecision = 0;
		lf.lfClipPrecision = 0;
		lf.lfQuality = PROOF_QUALITY;
		lf.lfPitchAndFamily = 0;
		strcpy(lf.lfFaceName, "΢���ź�");

		HFONT hf = CreateFontIndirectA(&lf);
		HDC hDC = CreateCompatibleDC(0);
		HFONT hOldFont = (HFONT)SelectObject(hDC, hf);

		int strBaseW = 0, strBaseH = 0;
		int singleRow = 0;
		char buf[3000];
		strcpy(buf, str.c_str());

		//�������
		{
			int nnh = 0;
			int cw, ch;
			const char* ln = strtok(buf, "\n");
			while (ln != 0)
			{
				GetStringSize(hDC, ln, &cw, &ch);
				strBaseW = max(strBaseW, cw);
				strBaseH = max(strBaseH, ch);

				ln = strtok(0, "\n");
				nnh++;
			}
			singleRow = strBaseH;
			strBaseH *= nnh;
		}

		int centerx = 0;
		int centery = 0;
		if (org.x < ORG_Center*0.5){
			org.x = (dst->width - strBaseW) * 0.5 + (org.x - ORG_Center);
			centerx = 1;
		}

		if (org.y < ORG_Center*0.5){
			org.y = (dst->height - strBaseH) * 0.5 + (org.y - ORG_Center);
			centery = 1;
		}

		x = org.x < 0 ? -org.x : 0;
		y = org.y < 0 ? -org.y : 0;

		if (org.x + strBaseW < 0 || org.y + strBaseH < 0)
		{
			SelectObject(hDC, hOldFont);
			DeleteObject(hf);
			DeleteObject(hDC);
			return;
		}

		r = org.x + strBaseW > dst->width ? dst->width - org.x - 1 : strBaseW - 1;
		b = org.y + strBaseH > dst->height ? dst->height - org.y - 1 : strBaseH - 1;
		org.x = org.x < 0 ? 0 : org.x;
		org.y = org.y < 0 ? 0 : org.y;

		BITMAPINFO bmp = { 0 };
		BITMAPINFOHEADER& bih = bmp.bmiHeader;
		int strDrawLineStep = strBaseW * 3 % 4 == 0 ? strBaseW * 3 : (strBaseW * 3 + 4 - ((strBaseW * 3) % 4));

		bih.biSize = sizeof(BITMAPINFOHEADER);
		bih.biWidth = strBaseW;
		bih.biHeight = strBaseH;
		bih.biPlanes = 1;
		bih.biBitCount = 24;
		bih.biCompression = BI_RGB;
		bih.biSizeImage = strBaseH * strDrawLineStep;
		bih.biClrUsed = 0;
		bih.biClrImportant = 0;

		void* pDibData = 0;
		HBITMAP hBmp = CreateDIBSection(hDC, &bmp, DIB_RGB_COLORS, &pDibData, 0, 0);
		if (pDibData == nullptr) return;

		HBITMAP hOldBmp = (HBITMAP)SelectObject(hDC, hBmp);

		//color.val[2], color.val[1], color.val[0]
		SetTextColor(hDC, RGB(255, 255, 255));
		SetBkColor(hDC, 0);
		//SetStretchBltMode(hDC, COLORONCOLOR);

		strcpy(buf, str.c_str());
		const char* ln = strtok(buf, "\n");
		int outTextY = 0;
		while (ln != 0)
		{
			if (centerx){
				int cw, ch;
				GetStringSize(hDC, ln, &cw, &ch);
				TextOutA(hDC, (strBaseW - cw) * 0.5, outTextY, ln, strlen(ln));
			}
			else{
				TextOutA(hDC, 0, outTextY, ln, strlen(ln));
			}
			outTextY += singleRow;
			ln = strtok(0, "\n");
		}

		//if (box){
		//	*box = cv::Rect(org.x, org.y, strBaseW, strBaseH);
		//	*box = *box & cv::Rect(0, 0, dst->width, dst->height);
		//}

		unsigned char* pImg = (unsigned char*)dst->imageData + org.x * dst->nChannels + org.y * dst->widthStep;
		unsigned char* pStr = (unsigned char*)pDibData + x * 3;
		for (int tty = y; tty <= b; ++tty)
		{
			unsigned char* subImg = pImg + (tty - y) * dst->widthStep;
			unsigned char* subStr = pStr + (strBaseH - tty - 1) * strDrawLineStep;
			for (int ttx = x; ttx <= r; ++ttx)
			{
				for (int n = 0; n < dst->nChannels; ++n){
					float alpha = subStr[n] / 255.0f;
					subImg[n] = cv::saturate_cast<uchar>(alpha * color.val[n] + (1 - alpha) * subImg[n]);
				}

				subStr += 3;
				subImg += dst->nChannels;
			}
		}

		SelectObject(hDC, hOldBmp);
		SelectObject(hDC, hOldFont);
		DeleteObject(hf);
		DeleteObject(hBmp);
		DeleteDC(hDC);
	}
#endif


	///////////////////////////////////////////////////////////////////////
	//Ϊ0ʱ����cache��Ϊ-1ʱ�����ж�cache
	FileCache::FileCache(int maxCacheSize){
		maxCacheSize_ = maxCacheSize;
	}

	vector<BBox> FileCache::loadxml(const string& file, int* width, int* height, const string& filter){
		return loadxmlFromData(this->loadfile(file), width, height, filter);
	}

	string FileCache::loadfile(const string& file){

		if (maxCacheSize_ == 0)
			return ccutil::loadfile(file);

		std::unique_lock<std::mutex> l(lock_);
		string data;
		if (!hitFile(file)){
			data = ccutil::loadfile(file);

			hits_[file] = data;
			cacheNames_.push_back(file);

			if (maxCacheSize_ > 0 && hits_.size() > maxCacheSize_){

				//random erase
				do{
					int n = ccutil::randr(cacheNames_.size() - 1);
					hits_.erase(cacheNames_[n]);
					cacheNames_.erase(cacheNames_.begin() + n);
				} while (hits_.size() > maxCacheSize_);
			}
		}
		else{
			data = hits_[file];
		}
		return data;
	}

	cv::Mat FileCache::loadimage(const string& file, int color){

		cv::Mat image;
		auto data = this->loadfile(file);
		if (data.empty())
			return image;

		try{ image = cv::imdecode(cv::Mat(1, data.size(), CV_8U, (char*)data.data()), color); }
		catch (...){}
		return image;
	}

	bool FileCache::hitFile(const string& file){
		return hits_.find(file) != hits_.end();
	}


	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* MD5 context. */
	typedef struct _MD5_CTX
	{
		unsigned long int state[4]; /* state (ABCD) */
		unsigned long int count[2]; /* number of bits, modulo 2^64 (lsb first) */
		unsigned char buffer[64]; /* input buffer */
	} MD5_CTX;

	/* Constants for MD5Transform routine.*/
	static unsigned char PADDING[64] = {
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};

	/* F, G, H and I are basic MD5 functions.*/
#define F(x, y, z) (((x) & (y)) | ((~x) & (z))) 
#define G(x, y, z) (((x) & (z)) | ((y) & (~z))) 
#define H(x, y, z) ((x) ^ (y) ^ (z)) 
#define I(x, y, z) ((y) ^ ((x) | (~z))) 

	/* ROTATE_LEFT rotates x left n bits.*/
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n)))) 

	/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
	Rotation is separate from addition to prevent recomputation.*/
#define FF(a, b, c, d, x, s, ac) { \
	(a) += F ((b), (c), (d)) + (x) + (unsigned long int)(ac);\
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
		} 
#define GG(a, b, c, d, x, s, ac) { \
	(a) += G ((b), (c), (d)) + (x) + (unsigned long int)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
			} 
#define HH(a, b, c, d, x, s, ac) { \
	(a) += H ((b), (c), (d)) + (x) + (unsigned long int)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
			} 
#define II(a, b, c, d, x, s, ac) { \
	(a) += I ((b), (c), (d)) + (x) + (unsigned long int)(ac); \
	(a) = ROTATE_LEFT ((a), (s)); \
	(a) += (b); \
			} 

	static void MD5Transform(unsigned long int[4], const unsigned char[64]);
	static void MD5Init(MD5_CTX *);
	static void MD5Update(MD5_CTX *, const unsigned char *, unsigned int);
	static void MD5Final(unsigned char[16], MD5_CTX *);
	static void Encode(unsigned char *, unsigned long int *, unsigned int);
	static void Decode(unsigned long int *, const unsigned char *, unsigned int);

	/* MD5 initialization. Begins an MD5 operation, writing a new context.*/
	static void MD5Init(MD5_CTX *context){
		context->count[0] = context->count[1] = 0;
		/* Load magic initialization constants.*/
		context->state[0] = 0x67452301;
		context->state[1] = 0xefcdab89;
		context->state[2] = 0x98badcfe;
		context->state[3] = 0x10325476;
	}

	/* MD5 block update operation. Continues an MD5 message-digest
	operation, processing another message block, and updating the
	context.*/
	static void MD5Update(MD5_CTX *context, /* context */const unsigned char *input, /* input block */unsigned int inputLen /* length of input block */){
		unsigned int i, index, partLen;

		/* Compute number of bytes mod 64 */
		index = (unsigned int)((context->count[0] >> 3) & 0x3F);

		/* Update number of bits */
		if ((context->count[0] += ((unsigned long int)inputLen << 3))
			< ((unsigned long int)inputLen << 3))
			context->count[1]++;
		context->count[1] += ((unsigned long int)inputLen >> 29);

		partLen = 64 - index;

		/* Transform as many times as possible.*/
		if (inputLen >= partLen) {
			memcpy((unsigned char*)&context->buffer[index], (unsigned char*)input, partLen);
			MD5Transform(context->state, context->buffer);

			for (i = partLen; i + 63 < inputLen; i += 64)
				MD5Transform(context->state, &input[i]);

			index = 0;
		}
		else
			i = 0;

		/* Buffer remaining input */
		memcpy((unsigned char*)&context->buffer[index], (unsigned char*)&input[i], inputLen - i);
	}

	/* MD5 finalization. Ends an MD5 message-digest operation, writing the
	the message digest and zeroizing the context.*/
	static void MD5Final(unsigned char digest[16], /* message digest */ MD5_CTX *context /* context */){
		unsigned char bits[8];
		unsigned int index, padLen;

		/* Save number of bits */
		Encode(bits, context->count, 8);

		/* Pad out to 56 mod 64.
		*/
		index = (unsigned int)((context->count[0] >> 3) & 0x3f);
		padLen = (index < 56) ? (56 - index) : (120 - index);
		MD5Update(context, PADDING, padLen);

		/* Append length (before padding) */
		MD5Update(context, bits, 8);

		/* Store state in digest */
		Encode(digest, context->state, 16);

		/* Zeroize sensitive information.*/
		memset((unsigned char*)context, 0, sizeof(*context));
	}

	/* MD5 basic transformation. Transforms state based on block.*/
	static void MD5Transform(unsigned long int state[4], const unsigned char block[64]){
		unsigned long int a = state[0], b = state[1], c = state[2], d = state[3], x[16];

		Decode(x, block, 64);

		/* Round 1 */
		FF(a, b, c, d, x[0], 7, 0xd76aa478); /* 1 */
		FF(d, a, b, c, x[1], 12, 0xe8c7b756); /* 2 */
		FF(c, d, a, b, x[2], 17, 0x242070db); /* 3 */
		FF(b, c, d, a, x[3], 22, 0xc1bdceee); /* 4 */
		FF(a, b, c, d, x[4], 7, 0xf57c0faf); /* 5 */
		FF(d, a, b, c, x[5], 12, 0x4787c62a); /* 6 */
		FF(c, d, a, b, x[6], 17, 0xa8304613); /* 7 */
		FF(b, c, d, a, x[7], 22, 0xfd469501); /* 8 */
		FF(a, b, c, d, x[8], 7, 0x698098d8); /* 9 */
		FF(d, a, b, c, x[9], 12, 0x8b44f7af); /* 10 */
		FF(c, d, a, b, x[10], 17, 0xffff5bb1); /* 11 */
		FF(b, c, d, a, x[11], 22, 0x895cd7be); /* 12 */
		FF(a, b, c, d, x[12], 7, 0x6b901122); /* 13 */
		FF(d, a, b, c, x[13], 12, 0xfd987193); /* 14 */
		FF(c, d, a, b, x[14], 17, 0xa679438e); /* 15 */
		FF(b, c, d, a, x[15], 22, 0x49b40821); /* 16 */

		/* Round 2 */
		GG(a, b, c, d, x[1], 5, 0xf61e2562); /* 17 */
		GG(d, a, b, c, x[6], 9, 0xc040b340); /* 18 */
		GG(c, d, a, b, x[11], 14, 0x265e5a51); /* 19 */
		GG(b, c, d, a, x[0], 20, 0xe9b6c7aa); /* 20 */
		GG(a, b, c, d, x[5], 5, 0xd62f105d); /* 21 */
		GG(d, a, b, c, x[10], 9, 0x2441453); /* 22 */
		GG(c, d, a, b, x[15], 14, 0xd8a1e681); /* 23 */
		GG(b, c, d, a, x[4], 20, 0xe7d3fbc8); /* 24 */
		GG(a, b, c, d, x[9], 5, 0x21e1cde6); /* 25 */
		GG(d, a, b, c, x[14], 9, 0xc33707d6); /* 26 */
		GG(c, d, a, b, x[3], 14, 0xf4d50d87); /* 27 */
		GG(b, c, d, a, x[8], 20, 0x455a14ed); /* 28 */
		GG(a, b, c, d, x[13], 5, 0xa9e3e905); /* 29 */
		GG(d, a, b, c, x[2], 9, 0xfcefa3f8); /* 30 */
		GG(c, d, a, b, x[7], 14, 0x676f02d9); /* 31 */
		GG(b, c, d, a, x[12], 20, 0x8d2a4c8a); /* 32 */

		/* Round 3 */
		HH(a, b, c, d, x[5], 4, 0xfffa3942); /* 33 */
		HH(d, a, b, c, x[8], 11, 0x8771f681); /* 34 */
		HH(c, d, a, b, x[11], 16, 0x6d9d6122); /* 35 */
		HH(b, c, d, a, x[14], 23, 0xfde5380c); /* 36 */
		HH(a, b, c, d, x[1], 4, 0xa4beea44); /* 37 */
		HH(d, a, b, c, x[4], 11, 0x4bdecfa9); /* 38 */
		HH(c, d, a, b, x[7], 16, 0xf6bb4b60); /* 39 */
		HH(b, c, d, a, x[10], 23, 0xbebfbc70); /* 40 */
		HH(a, b, c, d, x[13], 4, 0x289b7ec6); /* 41 */
		HH(d, a, b, c, x[0], 11, 0xeaa127fa); /* 42 */
		HH(c, d, a, b, x[3], 16, 0xd4ef3085); /* 43 */
		HH(b, c, d, a, x[6], 23, 0x4881d05); /* 44 */
		HH(a, b, c, d, x[9], 4, 0xd9d4d039); /* 45 */
		HH(d, a, b, c, x[12], 11, 0xe6db99e5); /* 46 */
		HH(c, d, a, b, x[15], 16, 0x1fa27cf8); /* 47 */
		HH(b, c, d, a, x[2], 23, 0xc4ac5665); /* 48 */

		/* Round 4 */
		II(a, b, c, d, x[0], 6, 0xf4292244); /* 49 */
		II(d, a, b, c, x[7], 10, 0x432aff97); /* 50 */
		II(c, d, a, b, x[14], 15, 0xab9423a7); /* 51 */
		II(b, c, d, a, x[5], 21, 0xfc93a039); /* 52 */
		II(a, b, c, d, x[12], 6, 0x655b59c3); /* 53 */
		II(d, a, b, c, x[3], 10, 0x8f0ccc92); /* 54 */
		II(c, d, a, b, x[10], 15, 0xffeff47d); /* 55 */
		II(b, c, d, a, x[1], 21, 0x85845dd1); /* 56 */
		II(a, b, c, d, x[8], 6, 0x6fa87e4f); /* 57 */
		II(d, a, b, c, x[15], 10, 0xfe2ce6e0); /* 58 */
		II(c, d, a, b, x[6], 15, 0xa3014314); /* 59 */
		II(b, c, d, a, x[13], 21, 0x4e0811a1); /* 60 */
		II(a, b, c, d, x[4], 6, 0xf7537e82); /* 61 */
		II(d, a, b, c, x[11], 10, 0xbd3af235); /* 62 */
		II(c, d, a, b, x[2], 15, 0x2ad7d2bb); /* 63 */
		II(b, c, d, a, x[9], 21, 0xeb86d391); /* 64 */

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;

		/* Zeroize sensitive information.*/
		memset((unsigned char*)x, 0, sizeof(x));
	}

	/* Encodes input (unsigned long int) into output (unsigned char). Assumes len is
	a multiple of 4.*/
	static void Encode(unsigned char *output,unsigned long int *input,unsigned int len){
		unsigned int i, j;

		for (i = 0, j = 0; j < len; i++, j += 4) {
			output[j] = (unsigned char)(input[i] & 0xff);
			output[j + 1] = (unsigned char)((input[i] >> 8) & 0xff);
			output[j + 2] = (unsigned char)((input[i] >> 16) & 0xff);
			output[j + 3] = (unsigned char)((input[i] >> 24) & 0xff);
		}
	}

	/* Decodes input (unsigned char) into output (unsigned long int). Assumes len is
	a multiple of 4.*/
	static void Decode(unsigned long int *output, const unsigned char *input, unsigned int len){
		unsigned int i, j;

		for (i = 0, j = 0; j < len; i++, j += 4)
			output[i] = ((unsigned long int)input[j]) | (((unsigned long int)input[j + 1]) << 8) |
			(((unsigned long int)input[j + 2]) << 16) | (((unsigned long int)input[j + 3]) << 24);
	}

	/* Digests a string and prints the result.*/
	static void md5Calc(const void* data, unsigned int len, char* md5_out)
	{
		MD5_CTX context;
		unsigned char digest[16];
		char output1[34];
		int i;

		MD5Init(&context);
		MD5Update(&context, (unsigned char*)data, len);
		MD5Final(digest, &context);

		for (i = 0; i < 16; i++)
		{
			sprintf(&(output1[2 * i]), "%02x", (unsigned char)digest[i]);
			sprintf(&(output1[2 * i + 1]), "%02x", (unsigned char)(digest[i] << 4));
		}

		for (i = 0; i<32; i++)
			md5_out[i] = output1[i];
	}

	string md5(const void* data, int length)
	{
		string out(32, 0);
		md5Calc(data, length, &out[0]);
		return out;
	}

	string md5(const string& data){
		return md5(data.data(), data.size());
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if defined(U_OS_LINUX)
#define __GetTimeBlock						\
	time_t timep;							\
	time(&timep);							\
	tm& t = *(tm*)localtime(&timep);
#endif

#if defined(U_OS_WINDOWS)
#define __GetTimeBlock						\
	tm t;									\
	_getsystime(&t);
#endif

	string timeNow(){
		char time_string[20];
		__GetTimeBlock;

		sprintf(time_string, "%04d-%02d-%02d %02d:%02d:%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
		return time_string;
	}

	string dateNow() {
		char time_string[20];
		__GetTimeBlock;

		sprintf(time_string, "%04d-%02d-%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday);
		return time_string;
	}

	void __assert_func(bool condition, const char* file, int line, const char* function, const char* code) {

		if (condition) return;

		__log_func(file, line, function,
			"ERROR: Assert is Failure: %s\n"
			"File: %s:%d\n"
			"Function: %s",
			code, file, line, function
		);
		abort();
	}

	static mutex __g_logger_lock_;
	static string __g_logger_directory;
	void setLoggerSaveDirectory(const string& loggerDirectory) {

		std::unique_lock<mutex> l(__g_logger_lock_);
		__g_logger_directory = loggerDirectory;

		if (__g_logger_directory.empty())
			__g_logger_directory = ".";

#if defined(U_OS_LINUX)
		if (__g_logger_directory.back() != '/') {
			__g_logger_directory.push_back('/');
		}
#endif

#if defined(U_OS_WINDOWS)
		if (__g_logger_directory.back() != '/' && __g_logger_directory.back() != '\\') {
			__g_logger_directory.push_back('/');
		}
#endif
	}

	void __log_func(const char* file, int line, const char* function, const char* fmt, ...) {

		std::unique_lock<mutex> l(__g_logger_lock_);
		string now = timeNow();

		va_list vl;
		va_start(vl, fmt);
		char loggerBuffer[10000];
		string funcName = function;
		if (funcName.length() > 32) {
			funcName = funcName.substr(0, 16) + "..." + funcName.substr(funcName.length() - 10);
		}

		int n = sprintf(loggerBuffer, "[%s:%s(%d):%s]:", now.c_str(), fileName(file, true).c_str(), line, funcName.c_str());
		vsprintf(loggerBuffer + n, fmt, vl);

		printf("%s\n", loggerBuffer);
		if (!__g_logger_directory.empty()) {
			string file = dateNow();
			string savepath = __g_logger_directory + file + ".log";
			FILE* f = fopen_mkdirs(savepath.c_str(), "a+");
			if (f) {
				fprintf(f, "%s\n", loggerBuffer);
				fclose(f);
			}
			else {
				printf("ERROR: can not open logger file: %s\n", savepath.c_str());
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////
};
