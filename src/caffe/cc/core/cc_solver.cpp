

#include "caffe/cc/core/cc_v5.h"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include <math.h>
#include <iostream>
#include <caffe/sgd_solvers.hpp>
#include "caffe/util/signal_handler.h"

using namespace std;
using namespace cv;

namespace cc{

#define cvt(p)	((caffe::Solver<float>*)p)
#define ptr		(cvt(this->_native))

	template <typename Dtype>
	caffe::Solver<Dtype>* GetSolver(const caffe::SolverParameter& param) {
		//return new SGDSolver<Dtype>(param);
		caffe::SolverParameter_SolverType type = param.solver_type();

		switch (type) {
		case caffe::SolverParameter_SolverType_SGD:
			return new caffe::SGDSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_NESTEROV:
			return new caffe::NesterovSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_ADAGRAD:
			return new caffe::AdaGradSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_RMSPROP:
			return new caffe::RMSPropSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_ADADELTA:
			return new caffe::AdaDeltaSolver<Dtype>(param);
		case caffe::SolverParameter_SolverType_ADAM:
			return new caffe::AdamSolver<Dtype>(param);
		default:
			LOG(FATAL) << "Unknown SolverType: " << type;
		}
		return (caffe::Solver<Dtype>*) NULL;
	}

	void Solver::postEarlyStopSignal(){
		ptr->postEarlyStopSignal();
	}

	void Solver::setNative(void* native){
		this->_native = native;
	}

	void* Solver::getNative(){
		return ptr;
	}

	CCAPI void CCCALL releaseSolver(Solver* solver){
		if (solver){
			void* p = solver->getNative();
			if (p) delete cvt(p);
		}
	}

	void Solver::restore(const char* resume_file){
		ptr->Restore(resume_file);
	}

	void Solver::snapshot(const char* filepath, bool save_solver_state){
		ptr->Snapshot(filepath, save_solver_state);
	}

	Solver::Solver(){
		static caffe::SignalHandler singalHandler(
			caffe::SolverAction::STOP,
			caffe::SolverAction::SNAPSHOT);
		this->signalHandler_ = &singalHandler;
		this->stepEndCallback_ = nullptr;
		this->stepEndCallbackUserData_ = nullptr;
	}

	void Solver::setBaseLearningRate(float rate){
		ptr->param_.set_base_lr(rate);
	}

	float Solver::getBaseLearningRate(){
		return ptr->param_.base_lr();
	}

	void Solver::postSnapshotSignal(){
		ptr->postSnapshotSignal();
	}

	void Solver::testAll(){
		ptr->TestAll();
	}

	Solver::~Solver(){
	}

	void Solver::setSetpEndCallback(TrainStepEndCallback callback, void* userdata){
		this->stepEndCallback_ = callback;
		this->stepEndCallbackUserData_ = userdata;
	}

	void* Solver::getStepEndCallbackUserData(){
		return this->stepEndCallbackUserData_;
	}
	
	TrainStepEndCallback Solver::getStepEndCallback(){
		return this->stepEndCallback_;
	}

	static Solver* buildSolver(caffe::SolverParameter& solver_param){
		// Set device id and mode
		if (solver_param.solver_mode() == caffe::SolverParameter_SolverMode_GPU) {
			//LOG(INFO) << "Use GPU with device ID " << solver_param.device_id();
			//caffe::Caffe::SetDevice(solver_param.device_id());
			caffe::Caffe::set_mode(caffe::Caffe::GPU);
		}
		else {
			//LOG(INFO) << "Use CPU.";
			caffe::Caffe::set_mode(caffe::Caffe::CPU);
		}
		return GetSolver<float>(solver_param)->ccSolver();
	}

	bool loadSolverNetFromString(caffe::SolverParameter& solver_param, const char* netstring = 0){
		if (netstring){
			//如果提供字符串，则优先使用这个
			solver_param.clear_net();
			solver_param.clear_net_param();
			solver_param.clear_train_net();
			solver_param.clear_test_net();
			solver_param.clear_test_net_param();
			solver_param.clear_train_net_param();
			caffe::NetParameter netp;
			caffe::ReadNetParamsFromTextStringOrDie(netstring, &netp);
			solver_param.mutable_net_param()->CopyFrom(netp);
		}
		return true;
	}

	CCAPI std::shared_ptr<Solver> CCCALL loadSolverFromPrototxtString(const char* solver_prototxt_string, const char* netstring){
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextString(solver_prototxt_string, &solver_param);
		if (!loadSolverNetFromString(solver_param, netstring))
			return std::shared_ptr<Solver>();
		
		return std::shared_ptr<Solver>(buildSolver(solver_param), releaseSolver);
	}

	CCAPI std::shared_ptr<Solver> CCCALL loadSolverFromPrototxt(const char* solver_prototxt, const char* netstring){
		caffe::SolverParameter solver_param;
		caffe::ReadProtoFromTextFileOrDie(solver_prototxt, &solver_param);
		if (!loadSolverNetFromString(solver_param, netstring))
			return std::shared_ptr<Solver>();

		return std::shared_ptr<Solver>(buildSolver(solver_param), releaseSolver);
	}

	void Solver::installActionSignalOperator(){
		ptr->SetActionFunction(((caffe::SignalHandler*)this->signalHandler_)->GetActionFunction());
	}

	void Solver::step(int iters){
		ptr->Step(iters);
	}

	int Solver::max_iter(){
		return ptr->param().max_iter();
	}

	float Solver::smooth_loss(){
		return ptr->smooth_loss();
	}

	int Solver::num_test_net(){
		return ptr->test_nets().size();
	}
	
	Net* Solver::test_net(int index){
		if (index < 0 || index >= num_test_net())
			return 0;

		return ptr->test_nets()[index]->ccNet();
	}

	Net* Solver::net(){
		return ptr->net()->ccNet();
	}

	int Solver::iter(){
		return ptr->iter();
	}

	void nofree(caffe::Solver<float>* p){
		//nothing todo
	}

	void Solver::solve(int numGPU, int* gpuid){

		installActionSignalOperator();

		if (numGPU < 2){

			if (numGPU > 0){
				CHECK_EQ(ptr->param().solver_mode(), caffe::SolverParameter_SolverMode_GPU) << "numGPU > 1, must set solver to GPU mode.";

				int currentDevice;
				CUDA_CHECK(cudaGetDevice(&currentDevice));
				CHECK_EQ(gpuid[0], currentDevice) << "The device ID has been set by the setGPU before and needs to be consistent";
			}

			for (int i = 0; i < numGPU; ++i){
				LOG(INFO) << "Use GPU with device ID " << gpuid[i];
			}
			ptr->Solve();
		}
		else{
			CHECK_EQ(ptr->param().solver_mode(), caffe::SolverParameter_SolverMode_GPU) << "numGPU > 1, must set solver to GPU mode.";

			for (int i = 0; i < numGPU; ++i){
				LOG(INFO) << "Use GPU with device ID " << gpuid[i];
			}

			boost::shared_ptr<caffe::Solver<float>> solver_nofree(ptr, nofree);
			caffe::P2PSync<float> sync(solver_nofree, NULL, ptr->param());
			sync.Run(vector<int>(gpuid, gpuid+numGPU));
		}
	}
}