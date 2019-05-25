## CC5.0 for Caffe
* 基于Caffe改造的C++接口框架
* 添加C++代码网络构建引擎
* 自定义层更容易，并且容易调试
* 更友好的Inference结构

## 训练代码
```
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
    //op->regularization_type = "L2";
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
```
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
```
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

        //从右到左
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