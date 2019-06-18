#include "caffe/layers/triplet_loss_layer.hpp"
#include "math.h"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  triplet_num_ = bottom[1]->num();
  sample_num_ = bottom[0]->num();
  feature_dim_ = bottom[0]->count(1);
  diff_an_.Reshape(feature_dim_, 1, 1, 1);
  diff_ap_.Reshape(feature_dim_, 1, 1, 1);
  diff_na_.Reshape(feature_dim_, 1, 1, 1);
  diff_pa_.Reshape(feature_dim_, 1, 1, 1);
  bottom_diff_.Reshape(sample_num_, feature_dim_, 1, 1);
  inner_matrix_.Reshape(sample_num_, sample_num_, 1, 1);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff_.mutable_cpu_data());
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::ComputeDiff_cpu(const Dtype *x_1,
  const Dtype *x_2, const Dtype x_1_norm, const Dtype x_2_norm,
  const Dtype inner_val, Dtype *x_1_diff) {
  caffe_cpu_scale(feature_dim_, Dtype(1) / (x_1_norm * x_2_norm),
      x_2, x_1_diff);
  Dtype x_1_norm_cubic = x_1_norm * x_1_norm * x_1_norm;
  caffe_cpu_axpby(feature_dim_, -inner_val / (x_1_norm_cubic * x_2_norm),
      x_1, Dtype(1), x_1_diff);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  Dtype eps = this->layer_param_.triplet_loss_param().eps();
  Dtype loss = 0;
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, sample_num_, sample_num_,
      feature_dim_, Dtype(1), bottom[0]->cpu_data(),
      bottom[0]->cpu_data(), Dtype(0),
      inner_matrix_.mutable_cpu_data());

  for (int i = 0; i < triplet_num_; ++i) {
    int a_idx = bottom[1]->cpu_data()[i * 3];
    int p_idx = bottom[1]->cpu_data()[i * 3 + 1];
    int n_idx = bottom[1]->cpu_data()[i * 3 + 2];
    const Dtype *a_pointer = bottom[0]->cpu_data() + a_idx * feature_dim_;
    const Dtype *p_pointer = bottom[0]->cpu_data() + p_idx * feature_dim_;
    const Dtype *n_pointer = bottom[0]->cpu_data() + n_idx * feature_dim_;
    const Dtype *inner_matrix_data = inner_matrix_.cpu_data();
    Dtype a_norm = sqrt(inner_matrix_data[a_idx * sample_num_ + a_idx] + eps);
    Dtype p_norm = sqrt(inner_matrix_data[p_idx * sample_num_ + p_idx] + eps);
    Dtype n_norm = sqrt(inner_matrix_data[n_idx * sample_num_ + n_idx] + eps);
    Dtype inner_ap = inner_matrix_data[a_idx * sample_num_ + p_idx];
    Dtype inner_an = inner_matrix_data[a_idx * sample_num_ + n_idx];
    Dtype dist_ap = inner_ap / (a_norm * p_norm);
    Dtype dist_an = inner_an / (a_norm * n_norm);
    if (dist_ap - dist_an - margin < 0) {
      ComputeDiff_cpu(a_pointer, p_pointer, a_norm,
          p_norm, inner_ap, diff_ap_.mutable_cpu_data());
      ComputeDiff_cpu(a_pointer, n_pointer, a_norm,
          n_norm, inner_an, diff_an_.mutable_cpu_data());
      ComputeDiff_cpu(p_pointer, a_pointer, p_norm,
          a_norm, inner_ap, diff_pa_.mutable_cpu_data());
      ComputeDiff_cpu(n_pointer, a_pointer, n_norm,
          a_norm, inner_an, diff_na_.mutable_cpu_data());

      caffe_cpu_axpby(feature_dim_, Dtype(1),
          diff_an_.cpu_data(), Dtype(1),
          bottom_diff_.mutable_cpu_data() + (a_idx * feature_dim_));
      caffe_cpu_axpby(feature_dim_, Dtype(-1),
          diff_ap_.cpu_data(), Dtype(1),
          bottom_diff_.mutable_cpu_data() + (a_idx * feature_dim_));
      caffe_cpu_axpby(feature_dim_, Dtype(-1),
          diff_pa_.cpu_data(), Dtype(1),
          bottom_diff_.mutable_cpu_data() + (p_idx * feature_dim_));
      caffe_cpu_axpby(feature_dim_, Dtype(1),
          diff_na_.cpu_data(), Dtype(1),
          bottom_diff_.mutable_cpu_data() + (n_idx * feature_dim_));

      loss += dist_an + margin - dist_ap;
    }
  }
  //Dtype scalar = Dtype(1) / triplet_num_;
  Dtype scalar = Dtype(1) / sample_num_;
  top[0]->mutable_cpu_data()[0] = loss * scalar;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //Dtype scalar = top[0]->cpu_diff()[0] / triplet_num_;
    Dtype scalar = top[0]->cpu_diff()[0] / sample_num_;
    caffe_cpu_scale(bottom_diff_.count(), scalar, bottom_diff_.cpu_data(),
        bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}
