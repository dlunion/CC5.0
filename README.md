## CC5.0 for Caffe
* 基于Caffe改造的C++接口框架
* 添加C++代码网络构建引擎
* 自定义层更容易，并且容易调试
* 更友好的Inference结构
* CUDA 10.0
* CUDNN 7.5
* 改造后，更容易实现和TensorRT结合
* 支持Linux、Windows，便于部署和移植
* 将要支持[Tensorboard](release/tensorboard)

## Windows编译
* 1、请下载[3rd共555MB](http://zifuture.com:1000/fs/25.shared/3rd.zip)，依赖的库，解压到README.md同级目录
* 2、安装cuda10
* 3、使用visual studio 2013打开windows-gpu.sln工程并选择ReleaseDLL编译即可

## Linux编译
* 1、按照caffe for Linux安装依赖项
* 2、执行make all -j32

## 编译后文件
* 在release里面有libcaffe.dll、libcaffe.lib等文件，需要依赖cudnn64_7.dll(cudnn7.5，在[3rd共555MB](http://zifuture.com:1000/fs/25.shared/3rd.zip)压缩包中有)
* 头文件是cc_v5.h
* cc_nb.h和cc_nb.cpp是network build engine文件，独立并依赖于cc_v5.h。可以根据需求修改他
* 案例在release文件夹里面，有训练和inference例子

## 训练代码
```C++
    cc::setGPU(0);
    cc::installRegister();
    INSTALL_LAYER(CifarDataLayer);

    auto data = L::data("CifarDataLayer", {"image", "label"}, "data");
    auto image = data[0];
    auto label = data[1];
    auto x = resnet18(image, 10);
    auto test = cc::metric::classifyAccuracy(x, label, "accuracy");
    auto loss = cc::loss::softmax_cross_entropy(x, label, "loss");
     
    int trainDatasetSize = 50000;
    int valDatasetSize = 10000;
    int trainEpochs = 150;
    int epochIters = trainDatasetSize / BatchSize;

    cc::engine::caffe::buildGraphToFile(resnet18(L::input({ 1, 3, InputHeight, InputWidth }, "image"), 10), "deploy.prototxt");
    auto op = cc::optimizer::momentumStochasticGradientDescent(cc::learningrate::step(0.1, 0.1, 20*epochIters), 0.9);
    op->test_initialization = false;
    op->test_interval = epochIters;
    op->test_iter = valDatasetSize / BatchSize;
    op->weight_decay = 0.0002f;
    op->average_loss = 1;
    op->max_iter = trainEpochs * epochIters;
    op->display = 10;
    op->device_ids = { 0 };
    //op->reload_weights = "pretrain.caffemodel";
    op->minimize({ loss, test });
    //op->minimizeFromFile("net.prototxt");

    saveToFile("solver.prototxt", op->seril());
    cc::engine::caffe::buildGraphToFile({ loss, test }, "train.prototxt");

    //setup test classifier callback
    registerOnTestClassificationFunction([&](Solver* solver, float testloss, int index, const char* itemname, float accuracy){
        //do smothing...
    });

    cc::train::caffe::run(op, [](OThreadContextSession* session, int step, float smoothed_loss){
        //do smothing...
    });
```

### 网络构建
```C++
cc::Tensor resnet18(const cc::Tensor& input, int numunit){
    auto x = input;
    x = resnet_conv(x, { 3, 3, 16 }, "conv1", 1, true, true);
    x = resnet_block(x, 2, 1, 16, 32, 2);
    x = resnet_block(x, 3, 2, 32, 64, 2);
    x = resnet_block(x, 4, 2, 64, 64, 2);

    x = L::avg_pooling2d(x, { 1, 1 }, { 1, 1 }, { 0, 0 }, true, "pool5");
    x = L::dense(x, numunit, "fc", true);

    //修改Dense（全连接层）参数，其他层也如此
    L::ODense* denseLayer = (L::ODense*)x->owner.get();
    denseLayer->weight_initializer.reset(new cc::Initializer("msra"));
    denseLayer->bias_initializer.reset(new cc::Initializer("constant", 0));
    return x;
}
```

### 自定义层
```C++
class LeftPooling : public cc::BaseLayer{

public:
    SETUP_LAYERFUNC(LeftPooling);

    virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop){

        diff_ = newBlob();
        diff_->reshapeLike(bottom[0]);
        top[0]->reshapeLike(bottom[0]);
    }

    virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop){

        //top[0]->copyFrom(bottom[0]);
        auto output = top[0];
        auto input = bottom[0];
        int fw = input->width();
        int fh = input->height();
        vector<float> maxColumn(fh);
        vector<int> maxIndex(fh);

        for (int n = 0; n < input->num(); ++n){
            for (int c = 0; c < input->channel(); ++c){
                float* ptr = input->mutable_cpu_data() + input->offset(n, c);
                float* out = output->mutable_cpu_data() + output->offset(n, c);
                float* diffptr = diff_->mutable_cpu_diff() + diff_->offset(n, c);
                for (int j = 0; j < fh; ++j){
                    maxColumn[j] = ptr[j * fw + fw - 1];
                    maxIndex[j] = fw - 1;
                }

                for (int i = fw - 1; i >= 0; --i){
                    for (int j = 0; j < fh; ++j){
                        float ival = ptr[j * fw + i];
                        float lval = maxColumn[j];
                        float mxval = max(ival, lval);

                        if (ival > lval)
                            maxIndex[j] = i; 
                        
                        diffptr[j * fw + maxIndex[j]]++;
                        out[j * fw + i] = mxval;
                        maxColumn[j] = mxval;
                    }
                }
            }
        }
    }

    virtual void backward(Blob** bottom, int numBottom, Blob** top, int numTop, const bool* propagate_down){

        auto input = bottom[0];
        auto output = top[0];
        
        float* inputDiff = input->mutable_cpu_diff();
        float* outputDiff = output->mutable_cpu_diff();
        float* diffPtr = diff_->mutable_cpu_diff();
        int count = input->count();
        for (int i = 0; i < count; ++i)
            *inputDiff++ = *outputDiff++ * (*diffPtr++ > 0 ? 1 : 0);
    }

private:
    shared_ptr<Blob> diff_;
};
```

* 使用的时候
```
INSTALL_LAYER(LeftPooling);
path1 = L::custom("LeftPooling", path1, {}, "leftPooling");
```

### [Inference案例](release/openpose/openpose.cpp)
模型下载[pose_iter_440000.caffemodel](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel)
```C++
    cc::setGPU(0);
    auto image = L::input({ 1, 3, 688, 368 }, "image");
    auto poseNetwork = pose(image, 5);
    auto net = cc::engine::caffe::buildNet(poseNetwork);
    //auto net = loadNetFromPrototxt("net.prototxt");

    net->weightsFromFile("pose_iter_440000.caffemodel");

    Mat im = imread("demo.jpg");
    Mat show = im.clone();
    Mat rawShow = show.clone();

    im.convertTo(im, CV_32F, 1 / 256.0, -0.5);
    net->input_blob(0)->setData(0, im);
    net->forward();

    float scalex = im.cols / (float)net->input_blob(0)->width();
    float scaley = im.rows / (float)net->input_blob(0)->height();
    Blob* keypoints = net->blob("Mconv7_stage6_L2");
```

### 引用
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose): Real-time multi-person keypoint detection library for body, face, hands, and foot estimation