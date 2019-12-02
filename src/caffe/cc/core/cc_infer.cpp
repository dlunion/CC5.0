
#if 0
#include "caffe/cc/core/cc_v5.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <math.h>
#include <iostream>
#include "caffe/net.hpp"
#include "caffe/util/signal_handler.h"
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/text_format.h>

#ifdef WIN32
#include <io.h>
#endif


using namespace std;
using namespace cv;

namespace cc{

	using google::protobuf::io::FileInputStream;
	using google::protobuf::io::FileOutputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;
	using google::protobuf::io::ZeroCopyOutputStream;
	using google::protobuf::io::CodedOutputStream;
	using google::protobuf::io::IstreamInputStream;
	using google::protobuf::io::GzipInputStream;
	using google::protobuf::Message;

	//
	//  ×Ô¶¨ÒåInfer
	//
	class Infer{
	public:
		virtual bool load(const char* deploy) = 0;
		virtual bool load(const char* deploydata, int length) = 0;

		virtual bool loadWeight(const char* caffemodel) = 0;
		virtual bool loadWeight(const char* caffemodeldata, int length) = 0;

		virtual Blob* input_blob(int index = 0) = 0;
		virtual Blob* input_blob(const char* name) = 0;
		virtual Blob* output_blob(int index = 0) = 0;
		virtual Blob* output_blob(const char* name) = 0;
		virtual void forward() = 0;
	};


	class InferImpl : public Infer{
	private:

	public:
		virtual bool load(const char* deploy){
		}

		virtual bool load(const char* deploydata, int length){
			
		}

		virtual bool loadWeight(const char* caffemodel){
		
		}

		virtual bool loadWeight(const char* caffemodeldata, int length){
		
		}

		virtual Blob* input_blob(int index = 0){
		
		}

		virtual Blob* input_blob(const char* name){
		
		}

		virtual Blob* output_blob(int index = 0){
		
		}

		virtual Blob* output_blob(const char* name){
		
		}

		virtual void forward(){
		
		}
	};
}
#endif