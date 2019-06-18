#ifndef CAFFE_SAMPLE_TRIPLET_LAYER_HPP_
#define CAFFE_SAMPLE_TRIPLET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/*
 * triplet selection
 */
template <typename Dtype>
class SampleTripletLayer : public NeuronLayer<Dtype> {
 public:
  explicit SampleTripletLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SampleTriplet"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // do nothing
  }
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  int triplet_num_;
  int sample_num_;
  int label_num_;
  int feature_dim_;
  int batch_size_;
  Blob<Dtype> inner_matrix_;
};

}  // namespace caffe

#endif  // CAFFE_SAMPLE_TRIPLET_LAYER_HPP_
