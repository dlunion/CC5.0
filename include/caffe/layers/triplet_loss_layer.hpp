#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class TripletLossLayer : public LossLayer<Dtype> {
 public:
  explicit TripletLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

 protected:
  void ComputeDiff_cpu(const Dtype *x_1, const Dtype *x_2,
      const Dtype x_1_norm, const Dtype x_2_norm, const Dtype inner_val,
      Dtype *x_1_diff);
  void ComputeDiff_gpu(const Dtype *x_1, const Dtype *x_2,
      const Dtype x_1_norm, const Dtype x_2_norm, const Dtype inner_val,
      Dtype *x_1_diff);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int triplet_num_;
  int sample_num_;
  int feature_dim_;
  Blob<Dtype> diff_an_;
  Blob<Dtype> diff_ap_;
  Blob<Dtype> diff_na_;
  Blob<Dtype> diff_pa_;
  Blob<Dtype> bottom_diff_;
  Blob<Dtype> inner_matrix_;
};

}

#endif
