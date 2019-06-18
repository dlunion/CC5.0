#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/sample_triplet_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class SampleTripletLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SampleTripletLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 2, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    blob_bottom_->mutable_cpu_data()[0] = 1;
    blob_bottom_->mutable_cpu_data()[1] = 1;
    blob_bottom_->mutable_cpu_data()[2] = 2;
    blob_bottom_->mutable_cpu_data()[3] = 2;
    blob_bottom_->mutable_cpu_data()[4] = 2.5;
    blob_bottom_->mutable_cpu_data()[5] = 3;
    blob_bottom_->mutable_cpu_data()[6] = 3;
    blob_bottom_->mutable_cpu_data()[7] = 4;
    blob_bottom_->mutable_cpu_data()[8] = 3;
    blob_bottom_->mutable_cpu_data()[9] = 5;
    blob_bottom_->mutable_cpu_data()[10] = 2;
    blob_bottom_->mutable_cpu_data()[11] = 5;
  }
  virtual ~SampleTripletLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SampleTripletLayerTest, TestDtypesAndDevices);

TYPED_TEST(SampleTripletLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  SampleTripletParameter* sample_triplet_param =
      layer_param.mutable_sample_triplet_param();
  sample_triplet_param->set_label_num(3);
  sample_triplet_param->set_sample_num(2);
  shared_ptr<SampleTripletLayer<Dtype> > layer(
      new SampleTripletLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_vec_[0]->cpu_data();
  int num = this->blob_top_vec_[0]->num();
  for (int i = 0; i < num; ++i) {
    std::cout << data[i * 3] << " " << data[i * 3 + 1] << " " << data[i * 3 + 2] << std::endl;
  }
}

}