


#include <cv.h>
#include <highgui.h>
#include "cc_nb.h"
#include "visualize-py.h"

using namespace std;
using namespace cv;

void notifyListener(const string& action){
	if (action == "testAction1"){
		printf("µ„ª˜¡À∞¥≈•1\n");
	}
}

int main(){

	registerServiceNotify(notifyListener);

	auto configResult = systemConfig("name: \"Train Network\"", "name: \"Test Network\"", {
		NotifyAction("testAction1", "≤‚ ‘∞¥≈•1", "notify", "info"),
		NotifyAction("openURL('http://www.baidu.com')", "≤‚ ‘∞¥≈•2", "code", "success")
	});
	cout << configResult << endl;

	int i = 0;
	while (true){
		postStatus("i = ", cc::f("%d", i));
		postString("≤‚ ‘»’÷æ", cc::f("≤‚ ‘ ‰≥ˆ%d", i));
		postChartValue("≤‚ ‘Õº±Ì1", "Loss", i, rand() / (float)RAND_MAX);
		postChartValue("≤‚ ‘Õº±Ì2", "Lr", i, rand() / (float)RAND_MAX);

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		i++;
	}
	return 0;
}