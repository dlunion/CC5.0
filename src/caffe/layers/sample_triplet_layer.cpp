#include "caffe/layers/sample_triplet_layer.hpp"
#include "math.h"

namespace caffe {

template <typename Dtype>
void SampleTripletLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  sample_num_ = this->layer_param_.sample_triplet_param().sample_num();
  label_num_ = this->layer_param_.sample_triplet_param().label_num();
  batch_size_ = sample_num_ * label_num_;
  triplet_num_ = label_num_ * sample_num_ * (sample_num_ - 1);
  top[0]->Reshape(triplet_num_, 3, 1, 1);
  feature_dim_ = bottom[0]->count(1);
  inner_matrix_.Reshape(batch_size_, batch_size_, 1, 1);
}

template <typename Dtype>
void SampleTripletLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype eps = this->layer_param_.sample_triplet_param().eps();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const Dtype *inner_data = inner_matrix_.cpu_data();
  int n_num = batch_size_ - sample_num_;
  int triplet_idx = 0;
  caffe_cpu_gemm(CblasNoTrans, CblasTrans, batch_size_, batch_size_,
      feature_dim_, Dtype(1), bottom[0]->cpu_data(),
      bottom[0]->cpu_data(), Dtype(0), inner_matrix_.mutable_cpu_data());
  for (int i = 0; i < label_num_; ++i) {
    int n_f = (i + 1) * sample_num_ % batch_size_;
    int n_r = (n_f + n_num) % batch_size_;
    for(int j = 0; j < sample_num_; ++j) {
      int a_idx = i * sample_num_ + j;
      int a_m_idx = a_idx * batch_size_ + a_idx;
      Dtype norm_a = sqrt(inner_data[a_m_idx] + eps);
      for (int k = 0; k < sample_num_; ++k) {
        if (k != j) {
          int p_idx = i * sample_num_ + k;
          int tmp_n_idx = n_f;
          int n_idx = -1;
          Dtype max_an = -1;
          while (tmp_n_idx != n_r) {
            int n_m_idx = tmp_n_idx * batch_size_ + tmp_n_idx;
            int an_m_idx = a_idx * batch_size_ + tmp_n_idx;
            Dtype norm_n = sqrt(inner_data[n_m_idx] + eps);
            Dtype tmp_an = inner_data[an_m_idx];
            tmp_an /= (norm_a * norm_n);
            if (tmp_an >= max_an) {
              max_an = tmp_an;
              n_idx = tmp_n_idx;
            }
            tmp_n_idx = (tmp_n_idx + 1) % batch_size_;
          }
          top_data[triplet_idx * 3] = a_idx;
          top_data[triplet_idx * 3 + 1] = p_idx;
          top_data[triplet_idx * 3 + 2] = n_idx;
          triplet_idx++;
        }
      }
    }
  }
}

INSTANTIATE_CLASS(SampleTripletLayer);
REGISTER_LAYER_CLASS(SampleTriplet);


}