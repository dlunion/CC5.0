#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class TripletLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  TripletLossLayerTest()
      : blob_bottom_feature_(new Blob<Dtype>(4, 2, 1, 1)),
        blob_bottom_triplet_(new Blob<Dtype>(4, 3, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_feature_);
    /*this->blob_bottom_feature_->mutable_cpu_data()[0] = 1;
    this->blob_bottom_feature_->mutable_cpu_data()[1] = 1;
    this->blob_bottom_feature_->mutable_cpu_data()[2] = 2;
    this->blob_bottom_feature_->mutable_cpu_data()[3] = 2;
    this->blob_bottom_feature_->mutable_cpu_data()[4] = 3;
    this->blob_bottom_feature_->mutable_cpu_data()[5] = 3;
    this->blob_bottom_feature_->mutable_cpu_data()[6] = 4;
    this->blob_bottom_feature_->mutable_cpu_data()[7] = 4;*/
    blob_bottom_triplet_->mutable_cpu_data()[0] = 0;
    blob_bottom_triplet_->mutable_cpu_data()[1] = 1;
    blob_bottom_triplet_->mutable_cpu_data()[2] = 2;
    blob_bottom_triplet_->mutable_cpu_data()[3] = 1;
    blob_bottom_triplet_->mutable_cpu_data()[4] = 0;
    blob_bottom_triplet_->mutable_cpu_data()[5] = 2;
    blob_bottom_triplet_->mutable_cpu_data()[6] = 2;
    blob_bottom_triplet_->mutable_cpu_data()[7] = 3;
    blob_bottom_triplet_->mutable_cpu_data()[8] = 1;
    blob_bottom_triplet_->mutable_cpu_data()[9] = 3;
    blob_bottom_triplet_->mutable_cpu_data()[10] = 2;
    blob_bottom_triplet_->mutable_cpu_data()[11] = 0;
  }
  virtual ~TripletLossLayerTest() {
    delete blob_bottom_feature_;
    delete blob_bottom_triplet_;
    delete blob_top_;
  }
  Blob<Dtype> *const blob_bottom_feature_;
  Blob<Dtype> *const blob_bottom_triplet_;
  Blob<Dtype> *const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TripletLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(TripletLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_feature_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_triplet_);
  this->blob_bottom_vec_.push_back(this->blob_bottom_label_);
  this->blob_top_vec_.push_back(this->blob_top_);
  LayerParameter layer_param;
  layer_param.mutable_triplet_loss_param()->set_margin(0.2);
  TripletLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-3, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                  this->blob_top_vec_, 0);
}

}