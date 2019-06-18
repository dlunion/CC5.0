#include "caffe/layers/triplet_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::ComputeDiff_gpu(const Dtype *x_1,
  const Dtype *x_2, const Dtype x_1_norm, const Dtype x_2_norm,
  const Dtype inner_val, Dtype *x_1_diff) {
  caffe_gpu_scale(feature_dim_, Dtype(1) / (x_1_norm * x_2_norm),
      x_2, x_1_diff);
  Dtype x_1_norm_cubic = x_1_norm * x_1_norm * x_1_norm;
  caffe_gpu_axpby(feature_dim_, -inner_val / (x_1_norm_cubic * x_2_norm),
      x_1, Dtype(1), x_1_diff);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  Dtype eps = this->layer_param_.triplet_loss_param().eps();
  Dtype loss = 0;
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  caffe_gpu_gemm(CblasNoTrans, CblasTrans, sample_num_, sample_num_,
      feature_dim_, Dtype(1), bottom[0]->gpu_data(),
      bottom[0]->gpu_data(), Dtype(0),
      inner_matrix_.mutable_gpu_data());

  for (int i = 0; i < triplet_num_; ++i) {
    int a_idx = bottom[1]->cpu_data()[i * 3];
    int p_idx = bottom[1]->cpu_data()[i * 3 + 1];
    int n_idx = bottom[1]->cpu_data()[i * 3 + 2];
    const Dtype *a_pointer = bottom[0]->gpu_data() + a_idx * feature_dim_;
    const Dtype *p_pointer = bottom[0]->gpu_data() + p_idx * feature_dim_;
    const Dtype *n_pointer = bottom[0]->gpu_data() + n_idx * feature_dim_;
    const Dtype *inner_matrix_data = inner_matrix_.cpu_data();
    Dtype a_norm = sqrt(inner_matrix_data[a_idx * sample_num_ + a_idx] + eps);
    Dtype p_norm = sqrt(inner_matrix_data[p_idx * sample_num_ + p_idx] + eps);
    Dtype n_norm = sqrt(inner_matrix_data[n_idx * sample_num_ + n_idx] + eps);
    Dtype inner_ap = inner_matrix_data[a_idx * sample_num_ + p_idx];
    Dtype inner_an = inner_matrix_data[a_idx * sample_num_ + n_idx];
    Dtype dist_ap = inner_ap / (a_norm * p_norm);
    Dtype dist_an = inner_an / (a_norm * n_norm);
    if (dist_ap - dist_an - margin < 0) {
      ComputeDiff_gpu(a_pointer, p_pointer, a_norm,
          p_norm, inner_ap, diff_ap_.mutable_gpu_data());
      ComputeDiff_gpu(a_pointer, n_pointer, a_norm,
          n_norm, inner_an, diff_an_.mutable_gpu_data());
      ComputeDiff_gpu(p_pointer, a_pointer, p_norm,
          a_norm, inner_ap, diff_pa_.mutable_gpu_data());
      ComputeDiff_gpu(n_pointer, a_pointer, n_norm,
          a_norm, inner_an, diff_na_.mutable_gpu_data());

      caffe_gpu_axpby(feature_dim_, Dtype(1),
          diff_an_.gpu_data(), Dtype(1),
          bottom_diff_.mutable_gpu_data() + (a_idx * feature_dim_));
      caffe_gpu_axpby(feature_dim_, Dtype(-1),
          diff_ap_.gpu_data(), Dtype(1),
          bottom_diff_.mutable_gpu_data() + (a_idx * feature_dim_));
      caffe_gpu_axpby(feature_dim_, Dtype(-1),
          diff_pa_.gpu_data(), Dtype(1),
          bottom_diff_.mutable_gpu_data() + (p_idx * feature_dim_));
      caffe_gpu_axpby(feature_dim_, Dtype(1),
          diff_na_.gpu_data(), Dtype(1),
          bottom_diff_.mutable_gpu_data() + (n_idx * feature_dim_));

      loss += dist_an + margin - dist_ap;
    }
  }
  //Dtype scalar = Dtype(1) / triplet_num_;
  Dtype scalar = Dtype(1) / sample_num_;
  top[0]->mutable_cpu_data()[0] = loss * scalar;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //Dtype scalar = top[0]->cpu_diff()[0] / triplet_num_;
    Dtype scalar = top[0]->cpu_diff()[0] / sample_num_;
    caffe_gpu_scale(bottom_diff_.count(), scalar, bottom_diff_.gpu_data(),
        bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}
