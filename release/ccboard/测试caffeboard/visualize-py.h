

#ifndef VISUALIZE_PY_H
#define VISUALIZE_PY_H


#include <cc_v5.h>
#include <string>
#include <cv.h>
#include <vector>

#ifndef WIN32
#define main			tmain
#define entryMain		tmain
#else
#define entryMain		main
#endif

#ifndef WIN32
#define EXPORT_FUNC     extern "C"
#define VISUAL_CALL
#else
#define EXPORT_FUNC     extern "C" __declspec(dllexport)
#define VISUAL_CALL
#endif

struct NotifyAction{
	std::string action;
	std::string title;
	std::string type;

	//白色，建蓝色primary, 深蓝色info, 绿色success, 黄色warning, 红色danger, 黑色inverse
	std::string style;

	NotifyAction(const std::string& action, const std::string& title, const std::string& type = "notify", const std::string& style = "success");
};

typedef void(*notify_listener)(const std::string& action);

std::string systemConfig(const std::string& trainNet, const std::string& testNet, const std::vector<NotifyAction>& actions);
void registerServiceNotify(notify_listener listener);
void postChartValue(const std::string& chartName, const std::string& itemName, int iter, float value);
void postStatus(const std::string& name, const std::string& value);
void postBlob(const std::string& name, cc::Blob* blob);
void postImage(const std::string& name, const cv::Mat& image);
void postString(const std::string& name, const std::string& value);
void postBegin();
void postEnd();

#endif //VISUALIZE_PY_H