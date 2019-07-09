

#include "visualize-py.h"
#include <cc_v5.h>
#include <functional>
#include "cc_nb.h"

using namespace cv;
using namespace std;
using namespace cc;

extern int entryMain();
int callmain(){
	return entryMain();
}

static Mat visualizeFeatureMap_featuremap(Blob* blob) {
	int channels = blob->channel();
	int num = channels;
	int fw = blob->width();
	int fh = blob->height();
	float* ptr = blob->mutable_cpu_data();
	int rows = sqrt((float)num);
	rows = rows < 1 ? 1 : rows;

	int cols = ceil(num / (float)rows);
	Mat dst = Mat::zeros(fh * rows, fw * cols, CV_32F);
	int cInd = 0;
	double mi = 0, mx = 1;

	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			if (cInd < num) {
				int c = cInd;
				Mat roi = dst(Rect(x * fw, y*fh, fw, fh));
				Mat fm = Mat(fh, fw, CV_32F, ptr + c * fw*fh);
				fm.copyTo(roi);
				cInd++;
			}
			else {
				Mat roi = dst(Rect(x * fw, y*fh, fw, fh));
				line(roi, Point(0, roi.rows*0.5), Point(roi.cols, roi.rows*0.5), Scalar::all(1), 1);
				line(roi, Point(roi.cols*0.5, 0), Point(roi.cols*0.5, roi.rows), Scalar::all(1), 1);
			}
		}
	}
	dst.convertTo(dst, CV_8U, 255.0);
	return dst;
}

shared_ptr<char> toSharedString(const string& s){

	if (s.empty())
		return shared_ptr<char>();

	shared_ptr<char> output;
	char* str = (char*)malloc(s.size() + 1);
	str[s.size()] = 0;
	memcpy(str, s.c_str(), s.size());
	output.reset(str, free);
	return output;
}

string sharedStringExtract(const shared_ptr<char>& s){

	if (!s) return "";
	return string(s.get());
}

struct Environment{
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

	void postBlob(const char* name, const Mat& blob){ if (monster_postBlob) monster_postBlob(this, name, blob); }
	void postChartValue(const char* chartName, const char* itemName, int iter, float value){ if (monster_postChartValue) monster_postChartValue(this, chartName, itemName, iter, value); }
	void postImage(const char* name, const Mat& image){ if (monster_postImage) monster_postImage(this, name, image); }
	void postString(const char* name, const char* str){ if (monster_postString) monster_postString(this, name, str); }
	void postStatus(const char* name, const char* str){ if (monster_postStatus) monster_postStatus(this, name, str); }
	void postBegin(){ if (monster_postBegin) monster_postBegin(this); }
	void postEnd(){ if (monster_postEnd) monster_postEnd(this); }
	std::shared_ptr<char> systemConfig(const char* configJSON){ if (monster_systemConfig)return monster_systemConfig(this, configJSON); return std::shared_ptr<char>(); };
};

static Environment* g_env = nullptr;
static notify_listener g_listener = nullptr;
static cc::Solver* g_solver = nullptr;

NotifyAction::NotifyAction(const std::string& action, const std::string& title, const std::string& type, const std::string& style){
	this->title = title;
	this->action = action;
	this->type = type;
	this->style = style;
}

string replace_str(const string& str_, const string& to_replaced, const string& newchars){

	string str = str_;
	for (string::size_type pos(0); pos != string::npos; pos += newchars.length()){
		pos = str.find(to_replaced, pos);
		if (pos != string::npos)
			str.replace(pos, to_replaced.length(), newchars);
		else
			break;
	}
	return str;
}

string processjsonstr(string jsonstr){

	jsonstr = replace_str(jsonstr, "\\", "\\\\");
	jsonstr = replace_str(jsonstr, "\"", "\\\"");
	return replace_str(jsonstr, "\n", "\\n");
}

string notifyActionToJSON(const std::vector<NotifyAction>& actions){

	string ss;
	for (int i = 0; i < actions.size(); ++i){
		auto& item = actions[i];
		string line = format("{\"title\":\"%s\", \"action\": \"%s\", \"type\": \"%s\", \"style\": \"btn-%s\"}",
			processjsonstr(item.title).c_str(),
			processjsonstr(item.action).c_str(),
			processjsonstr(item.type).c_str(),
			processjsonstr(item.style).c_str()
		);
		if (i > 0) line = "," + line;
		ss += line;
	}
	return ss;
}

std::string systemConfig(const std::string& trainNet, const std::string& testNet, const std::vector<NotifyAction>& actions_){

	if (!g_env) return "not init";

	//"   'actions': ["
	//"		{'title': '关闭检测效果', 'action': 'closeDetection'}, "
	//"		{'title': '关闭检测效果', 'action': 'closeDetection'}, "
	//"		{'title': '关闭检测效果', 'action': 'closeDetection'}, "
	//"		{'title': '关闭检测效果', 'action': 'closeDetection'}, "
	//"		{'title': '关闭检测效果', 'action': 'closeDetection'}, "
	//"		{'title': '开启检测效果', 'action': 'openDetection' }, "
	//"		{'title': '开启检测效果', 'action': 'openDetection' }, "
	//"		{'title': '开启检测效果', 'action': 'openDetection' }, "
	//"		{'title': '开启检测效果', 'action': 'openDetection' }, "
	//"		{'title': '开启检测效果', 'action': 'openDetection' }, "
	//"		{'title': '开启检测效果', 'action': 'openDetection' }, "
	//"		{'title': '开启检测效果', 'action': 'openDetection' } "
	//"	]"

	//切面注册，保证post是事务级别的
	cc::OThreadContextSession::set_pre_step_end_callback([](OThreadContextSession* session, int step, float smoothed_loss){
		postBegin();
	});

	cc::OThreadContextSession::set_over_step_end_callback([](OThreadContextSession* session, int step, float smoothed_loss){
		postEnd();
	});

	std::vector<NotifyAction> actions = actions_;
	string _trainNet = processjsonstr(trainNet);
	string _testNet = processjsonstr(testNet);

	vector<NotifyAction> plus = {
		NotifyAction("openGIST(\"trainNet\")", "查看训练网络", "code", "info"),
		NotifyAction("openGIST(\"testNet\")", "查看测试网络", "code", "info"),
		NotifyAction("openFullLogs()", "查看全部日志", "code", "info")
	};
	actions.insert(actions.begin(), plus.begin(), plus.end());

	string config = 
		"{"
		"	\"sys\": {"
		"		\"trainNet\": \"" + _trainNet + "\","
		"		\"testNet\": \"" + _testNet + "\""
		"	},"
		"  \"actions\": ["  +
			notifyActionToJSON(actions) +
		"	]"
		"}";
	return sharedStringExtract(g_env->systemConfig(config.c_str()));
}

void registerServiceNotify(notify_listener listener){
	g_listener = listener;
}

static shared_ptr<char> local_notify(const char* eventType, const char* message){

	if (eventType == nullptr || message == nullptr)
		return toSharedString("no event");

	if (g_listener){
		g_listener(eventType);
		return toSharedString("ok");
	}

	return toSharedString("no listener");
}

static void local_stopTrain(){
	g_solver->postEarlyStopSignal();
}

EXPORT_FUNC void VISUAL_CALL iopInit(Environment* env){

	g_env = env;
	env->local_train = callmain;
	env->local_notify = local_notify;
	env->local_stopTrain = local_stopTrain;

	cc::OThreadContextSession::add_pre_train_callback([](OThreadContextSession* session){
		g_solver = session->solver();
	});
}

void postBlob(const string& name, cc::Blob* blob){

	if (!g_env) return;
	g_env->postBlob(name.c_str(), visualizeFeatureMap_featuremap(blob));
}

void postChartValue(const std::string& chartName, const std::string& itemName, int iter, float value){

	if (!g_env) return;
	g_env->postChartValue(chartName.c_str(), itemName.c_str(), iter, value);
}

void postImage(const std::string& name, const Mat& image){

	if (!g_env) return;
	g_env->postImage(name.c_str(), image);
}

void postString(const std::string& name, const std::string& value){
	if (!g_env) return;
	g_env->postString(name.c_str(), value.c_str());
}

void postStatus(const std::string& name, const std::string& value){
	if (!g_env) return;
	g_env->postStatus(name.c_str(), value.c_str());
}

void postBegin(){
	if (!g_env) return;
	g_env->postBegin();
}

void postEnd(){
	if (!g_env) return;
	g_env->postEnd();
}