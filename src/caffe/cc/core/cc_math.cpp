
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/cc/core/cc_v5.h"
#include <map>
#include <string>


#ifdef WIN32
#include <import-staticlib.h>
#include <Windows.h>
#endif

#include <thread>
#include "caffe/layers/cpp_layer.hpp"

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/io/strtod.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/unknown_field_set.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/tokenizer.h>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"


namespace cc{


	namespace math{

		void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const float alpha, const float* A, const float* B, const float beta,
			float* C){

			caffe::caffe_cpu_gemm<float>((::CBLAS_TRANSPOSE)TransA, (::CBLAS_TRANSPOSE)TransB, M, N, K, alpha, A, B, beta, C);
		}

		void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
			const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
			const float alpha, const float* A, const float* B, const float beta,
			float* C){

			caffe::caffe_gpu_gemm<float>((::CBLAS_TRANSPOSE)TransA, (::CBLAS_TRANSPOSE)TransB, M, N, K, alpha, A, B, beta, C);
		}
	}
}