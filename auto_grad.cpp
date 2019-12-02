

#include "auto_grad.h"

using namespace cv;
using namespace cc;
using namespace std;

namespace cc{

	struct OpRegister{
		std::function<BlobContainer(Operator* op)> compute_forward;
		std::function<BlobContainer(const BlobContainer& grad, Operator* op, int ind_param)> compute_backward;
	};

	string format(const char* fmt, ...){
		va_list vl;
		va_start(vl, fmt);

		string out;
		{
			char buffer[16 * 1024];
			buffer[0] = 0;
			vsprintf(buffer, fmt, vl);
			out = buffer;
		};
		return out;
	}

	template<typename _T>
	static inline _T clamp(_T x, _T low, _T high) {

		if (x < low)
			return low;

		if (x > high)
			return high;
		return x;
	}

	template<typename _T>
	static inline _T declamp(_T x, _T low, _T high) {

		if (x < low)
			return 0;

		if (x > high)
			return 0;
		return x;
	}

	static inline float __sigmoid(float x){
		return 0.5f * tanh(0.5f * x) + 0.5f;
	}

	static inline float sigmoid(float x){
		return clamp(__sigmoid(x), 1e-4f, 1 - 1e-4f);
	}

	static inline float desigmoid(float x){
		float s = __sigmoid(x);
		s = declamp(s, 1e-4f, 1 - 1e-4f);
		return s * (1 - s);
	}

	static inline float deabs(float x){
		return x > 0 ? 1 : (x < 0 ? -1 : 0);
	}

	BlobContainer::BlobContainer(){
		
	}

	BlobContainer::BlobContainer(float value){
		this->is_const_number = true;
		this->value = value;
		this->gradient = 0;
		this->require_gradient = false;
	}

	BlobContainer::BlobContainer(float value, float gradient){
		this->is_const_number = true;
		this->value = value;
		this->gradient = gradient;
		this->require_gradient = true;
	}

	BlobContainer::BlobContainer(float value, bool require_gradient){
		this->is_const_number = true;
		this->value = value;
		this->gradient = 0;
		this->require_gradient = require_gradient;
	}

	BlobContainer::BlobContainer(const shared_ptr<Blob>& other, bool require_gradient){
		this->is_const_number = false;
		this->data = other;
		this->require_gradient = require_gradient;
	}

	BlobContainer::BlobContainer(int n, int c, int h, int w, bool require_gradient){
		this->is_const_number = false;
		this->data = newBlobByShape(n, c, h, w);
		this->require_gradient = require_gradient;
	}

	void BlobContainer::reshape(int n, int c, int h, int w){
		
		if (!data)
			data = newBlob();

		is_const_number = false;
		data->reshape(n, c, h, w);
	}

	float& BlobContainer::at(int h, int w){
		if (!isTensor())
			throw "must tensor";
		return data->cpu_at(0, 0, h, w);
	}

	float& BlobContainer::at(int n, int c, int h, int w){
		if (!isTensor())
			throw "must tensor";
		return data->cpu_at(n, c, h, w);
	}

	float& BlobContainer::getValue(){
		if (!isConst())
			throw "must const";
		return value;
	}

	void BlobContainer::view() const{
		
		cout << "====================" << endl;
		if (empty()){
			cout << "empty" << endl;
			return;
		}

		if (isConst()){
			if (require_gradient)
				cout << cc::format("value=%g [gradient=%g]", value, gradient) << endl;
			else
				cout << cc::format("value=%g [no gradient]", value) << endl;
			return;
		}

		int h = this->data->height();
		int w = this->data->width();
		for (int n = 0; n < this->data->num(); ++n){
			for (int c = 0; c < this->data->channel(); ++c){
				float* ptr = this->data->cpu_ptr(n, c);
				Mat m1(h, w, CV_32F, ptr);

				cout << cc::format("data: %d, %d, %d x %d", n, c, h, w) << endl;
				cout << m1 << endl;

				if (require_gradient){
					ptr = this->data->cpu_diff_ptr(n, c);
					Mat m2(h, w, CV_32F, ptr);

					cout << cc::format("gradient: %d, %d, %d x %d", n, c, h, w) << endl;
					cout << m2 << endl;
				}
			}
		}
	}

	int BlobContainer::shapeCount() const{
		
		if (empty()) return 0;
		if (is_const_number) return 1;
		return this->data->count();
	}

	bool BlobContainer::isConst() const{
		return is_const_number;
	}

	bool BlobContainer::isTensor() const{
		return !is_const_number && !empty();
	}

	bool BlobContainer::shapeMatch(const BlobContainer& other) const{
		if (empty() || other.empty())
			return false;

		if (isTensor() != other.isTensor())
			return false;

		if (isTensor()){
			return this->data->num() == other.data->num() &&
				this->data->channel() == other.data->channel() &&
				this->data->height() == other.data->height() &&
				this->data->width() == other.data->width();
		}
		return true;
	}

	bool BlobContainer::empty() const{
		if (is_const_number) return false;
		return data.get() == nullptr;
	}

	void BlobContainer::setGradientTo(float gradient){

		if (isConst()){
			this->gradient = gradient;
			return;
		}

		if (isTensor()){
			int count = data->count();
			float* ptr = data->cpu_diff_ptr();
			for (int i = 0; i < count; ++i)
				*ptr++ = gradient;
		}
	}

	void BlobContainer::setDataTo(float value){

		if (isConst()){
			this->value = value;
			return;
		}

		if (isTensor()){
			int count = data->count();
			float* ptr = data->cpu_ptr();
			for (int i = 0; i < count; ++i)
				*ptr++ = value;
		}
	}

	BlobContainer BlobContainer::gemm(const BlobContainer& other) const{

		const BlobContainer& a = *this;
		const BlobContainer& b = other;
		if (a.isTensor() != b.isTensor()){
			throw "must tensor and tensor to gemm";
		}

		if (a.data->num() != b.data->num()){
			if (a.data->num() != 1 && b.data->num() != 1){
				throw "shape not match by gemm";
			}
		}

		throw "not impl";
		return BlobContainer();
	}

	BlobContainer BlobContainer::degemm(const BlobContainer& other) const{

		const BlobContainer& a = *this;
		const BlobContainer& b = other;
		if (a.isTensor() != b.isTensor()){
			throw "must tensor and tensor to degemm";
			return BlobContainer();
		}

		throw "not impl";
		return BlobContainer();
	}

	BlobContainer BlobContainer::cloneSameShape() const{
		BlobContainer new_(*this);
		if (!is_const_number && data)
			new_.data = newBlobByShape(data->num(), data->channel(), data->height(), data->width());
		return new_;
	}

	BlobContainer BlobContainer::clone(bool clonediff) const{
		BlobContainer new_(*this);
		if (!is_const_number && data){
			//new_.data = data->clone(clonediff);
			new_.data = newBlob();
			new_.data->reshapeLike(this->data.get());
			memcpy(new_.data->cpu_ptr(), this->data->cpu_ptr(), sizeof(float) * this->data->count());

			if (clonediff)
				memcpy(new_.data->cpu_diff_ptr(), this->data->cpu_diff_ptr(), sizeof(float) * this->data->count());
		}
		return new_;
	}

#define ElementOP_Call(leftval, _op_op_, rightval)		((leftval) _op_op_ (rightval))
#define DeclareElementOP(__op_name, _op_op_)	\
	BlobContainer BlobContainer::elementOP_##__op_name(const BlobContainer& other, bool asvalue, bool asgradient, bool inplace) const{	  \
																																	  \
		if (this->empty() || other.empty())																							  \
			return BlobContainer();																									  \
																																	  \
		if (!asvalue && !asgradient)																								  \
			return *this;																											  \
																																	  \
		if (this->isTensor() && other.isTensor()){																					  \
			/*如果都是tensor就必须要匹配shape*/																						  \
			if (!this->shapeMatch(other))																							  \
				throw "shape count mismatch.";																						  \
		}																															  \
																																	  \
		if (this->isConst() && other.isConst()){																					  \
			BlobContainer c(*this);																									  \
			c.require_gradient = this->require_gradient || other.require_gradient;													  \
																																	  \
			if (asvalue)																											  \
				c.value = ElementOP_Call(this->value, _op_op_, other.value);																		  \
																																	  \
			if (asgradient)																											  \
				c.gradient = ElementOP_Call(this->gradient, _op_op_, other.gradient);															  \
			return c;																												  \
		}																															  \
		else{																														  \
																																	  \
			int a_is_tensor = this->isTensor();																						  \
			int b_is_tensor = other.isTensor();																						  \
																																	  \
			BlobContainer c;																										  \
			if (inplace){																											  \
				c = this->isTensor() ? *this : other.clone(asgradient);																  \
			}																														  \
			else{																													  \
				c = this->isTensor() ? this->clone(asgradient) : other.clone(asgradient);											  \
			}																														  \
																																	  \
			c.require_gradient = this->require_gradient || other.require_gradient;													  \
			if (asvalue){																											  \
				int c_count = c.data->count();																						  \
				const float* aptr = this->isTensor() ? this->data->cpu_ptr() : &this->value;										  \
				const float* bptr = other.isTensor() ? other.data->cpu_ptr() : &other.value;										  \
				float* cptr = c.isTensor() ? c.data->cpu_ptr() : &c.value;															  \
				for (int i = 0; i < c_count; ++i){																					  \
					*cptr++ = ElementOP_Call(*aptr, _op_op_, *bptr);																					  \
					if (a_is_tensor)																								  \
						aptr++;																										  \
																																	  \
					if (b_is_tensor)																								  \
						bptr++;																										  \
				}																													  \
			}																														  \
																																	  \
			if (asgradient){																										  \
				int c_count = c.data->count();																						  \
				const float* aptr = this->isTensor() ? this->data->cpu_diff_ptr() : &this->gradient;								  \
				const float* bptr = other.isTensor() ? other.data->cpu_diff_ptr() : &other.gradient;								  \
				float* cptr = c.isTensor() ? c.data->cpu_diff_ptr() : &c.gradient;													  \
				for (int i = 0; i < c_count; ++i){																					  \
					*cptr++ = ElementOP_Call(*aptr, _op_op_, *bptr);																					  \
					if (a_is_tensor)																								  \
						aptr++;																										  \
																																	  \
					if (b_is_tensor)																								  \
						bptr++;																										  \
				}																													  \
			}																														  \
			return c;																												  \
		}																															  \
	}																																  \
																																	  \
	BlobContainer& BlobContainer::__op_name(const BlobContainer& valueOrGradient, bool asvalue, bool asgradient){							  \
		*this = elementOP_##__op_name(valueOrGradient, asvalue, asgradient, true);															  \
		return *this;																												  \
	}
	
	DeclareElementOP(add, +);
	DeclareElementOP(sub, -);
	DeclareElementOP(mul, *);
	DeclareElementOP(div, /);

	/*gradient参数的cpu_data是梯度值，自加到当前变量的cpu_diff内*/																	  
	BlobContainer& BlobContainer::addGradient(const BlobContainer& gradient){														  
																																	  
		if (this->empty() || gradient.empty())																						  
			return *this;																											  
																																	  
		if (this->isTensor() && gradient.isTensor()){																				  
			/*如果都是tensor就必须要匹配shape*/																						  
			if (!this->shapeMatch(gradient))																						  
				throw "shape count mismatch.";																						  
		}																															  
																																	  
		if (this->isConst() && gradient.isConst()){																					  
			this->gradient += gradient.value;																						  
		}																															  
		else{																														  
			if (!this->isTensor())																									  
				throw "must is tensor.";																							  
																																	  
			float* aptr = this->data->cpu_diff_ptr();																				  
			int count = this->data->count();																						  
			bool bistensor = gradient.isTensor();																					  
			const float* bptr = bistensor ? gradient.data->cpu_ptr() : &gradient.value;												  
																																	  
			if (bistensor){																											  
				for (int i = 0; i < count; ++i)																						  
					*aptr++ += *bptr++;																								  
			}																														  
			else {																													  
				for (int i = 0; i < count; ++i)																						  
					*aptr++ += *bptr;																								  
			}																														  
		}																															  
		return *this;																												  
	}

#define DeclareLogicalOP(name, __op)								 \
	BlobContainer BlobContainer::name(float value){				 \
																 \
		if (isConst())											 \
			return BlobContainer(this->value __op value);		 \
																 \
		if (empty())											 \
			return BlobContainer();								 \
																 \
		BlobContainer out = clone();							 \
		out.require_gradient = false;							 \
																 \
		float* ptr = data->cpu_ptr();							 \
		float* output = out.data->cpu_ptr();					 \
		int count = data->count();								 \
		for (int i = 0; i < count; ++i)							 \
			*output++ = *ptr++ __op value ? 1 : 0;				 \
		return out;												 \
	}															 

	//EQ 就是 EQUAL等于 
	//NE 就是 NOT EQUAL不等于 
	//GT 就是 GREATER THAN大于　 
	//LT 就是 LESS THAN小于 
	//GE 就是 GREATER THAN OR EQUAL 大于等于 
	//LE 就是 LESS THAN OR EQUAL 小于等于
	DeclareLogicalOP(eq, == );
	DeclareLogicalOP(ne, != );
	DeclareLogicalOP(gt, >);
	DeclareLogicalOP(lt, <);
	DeclareLogicalOP(ge, >= );
	DeclareLogicalOP(le, <= );




#define DeclareFunc(name, callname)											    \
	BlobContainer BlobContainer::name(bool inplace){				    \
																			\
		if (isConst())													    \
			return BlobContainer(::callname(this->value), require_gradient);	\
																			\
		if (isTensor()){												   \
			BlobContainer c = inplace ? *this : this->clone();			   \
			int count = c.data->count();								   \
			float* ptr = c.data->cpu_ptr();								   \
			for (int i = 0; i < count; ++i, ++ptr)								   \
				*ptr = ::callname(*ptr);							   \
			return c;													   \
		}																   \
		return *this;													   \
	}				

	DeclareFunc(log, log);
	DeclareFunc(exp, exp);
	DeclareFunc(tanh, tanh);
	DeclareFunc(sqrt, sqrt);
	DeclareFunc(sigmoid, sigmoid);
	DeclareFunc(desigmoid, desigmoid);
	DeclareFunc(abs, abs);
	DeclareFunc(deabs, deabs);

	BlobContainer BlobContainer::sum() const{

		if (isConst())													   
			return this->value;
																		   
		if (isTensor()){												   
			int count = data->count();								   
			float* ptr = data->cpu_ptr();
			float sumval = 0;
			for (int i = 0; i < count; ++i)								   
				sumval += *ptr++;
			return BlobContainer(sumval, this->require_gradient);
		}																   
		return 0;													   
	}			

	BlobContainer BlobContainer::mean() const{
		if (isConst())
			return this->value;

		if (isTensor()){
			int count = data->count();
			float* ptr = data->cpu_ptr();
			float sumval = 0;
			for (int i = 0; i < count; ++i)
				sumval += *ptr++;
			return BlobContainer(count > 0 ? sumval / count : 0, this->require_gradient);
		}
		return 0;
	}

	BlobContainer& BlobContainer::toValue(){

		if (isConst()){
			this->value = this->gradient;
			return *this;
		}

		if (isTensor()){
			int count = this->data->count();
			float* ptr = this->data->cpu_ptr();
			float* diff = this->data->cpu_diff_ptr();
			memcpy(ptr, diff, count * sizeof(float));
			return *this;
		}
		return *this;
	}

	BlobContainer& BlobContainer::toGradient(){

		if (isConst()){
			this->gradient = this->value;
			return *this;
		}

		if (isTensor()){
			int count = this->data->count();
			float* ptr = this->data->cpu_ptr();
			float* diff = this->data->cpu_diff_ptr();
			memcpy(diff, ptr, count * sizeof(float));
			return *this;
		}
		return *this;
	}

	BlobContainer BlobContainer::pow(float powvalue, bool inplace) const{

		if (isConst())
			return ::pow(this->value, powvalue);

		if (isTensor()){
			BlobContainer c = inplace ? *this : this->clone();
			int count = c.data->count();
			float* ptr = c.data->cpu_ptr();
			for (int i = 0; i < count; ++i, ++ptr)
				*ptr = ::pow(*ptr, powvalue);
			return c;
		}
		return *this;
	}

	BlobContainer BlobContainer::clamp(float low, float high, bool inplace) const{

		if (isConst())
			return ::clamp(this->value, low, high);

		if (isTensor()){
			BlobContainer c = inplace ? *this : this->clone();
			int count = c.data->count();
			float* ptr = c.data->cpu_ptr();
			for (int i = 0; i < count; ++i, ++ptr)
				*ptr = ::clamp(*ptr, low, high);;
			return c;
		}
		return *this;
	}

	BlobContainer BlobContainer::declamp(float low, float high, bool inplace) const{

		if (isConst())
			return ::declamp(this->value, low, high);

		if (isTensor()){
			BlobContainer c = inplace ? *this : this->clone();
			int count = c.data->count();
			float* ptr = c.data->cpu_ptr();
			for (int i = 0; i < count; ++i, ++ptr)
				*ptr = ::declamp(*ptr, low, high);;
			return c;
		}
		return *this;
	}

	BlobContainer& BlobContainer::operator = (float value){

		if (isConst()){
			this->value = value;
			return *this;
		}

		data.reset();
		this->value = value;
		this->is_const_number = true;
		this->gradient = 0;
		this->require_gradient = false;
		return *this;
	}

	const BlobContainer& BlobContainer::operator + (void) const{
		return *this;
	}

	BlobContainer BlobContainer::operator - (void) const{
		return elementOP_mul(BlobContainer(-1), true, false , false);
	}

	BlobContainer BlobContainer::operator + (const BlobContainer& other) const{
		return elementOP_add(other, true, false, false);
	}

	BlobContainer BlobContainer::operator - (const BlobContainer& other) const{
		return elementOP_sub(other, true, false, false);
	}

	BlobContainer BlobContainer::operator * (const BlobContainer& other) const{
		return elementOP_mul(other, true, false, false);
	}

	BlobContainer BlobContainer::operator / (const BlobContainer& other) const{
		return elementOP_div(other, true, false, false);
	}

	BlobContainer& BlobContainer::operator += (const BlobContainer& other){
		*this = elementOP_add(other, true, false, true);
		return *this;
	}

	BlobContainer& BlobContainer::operator -= (const BlobContainer& other){
		*this = elementOP_sub(other, true, false, true);
		return *this;
	}

	BlobContainer& BlobContainer::operator *= (const BlobContainer& other){
		*this = elementOP_mul(other, true, false, true);
		return *this;
	}

	BlobContainer& BlobContainer::operator /= (const BlobContainer& other){
		*this = elementOP_div(other, true, false, true);
		return *this;
	}

	void VariableData::forwardREF(){
		refcount++;
		backward_refcount = refcount;
	}

	void VariableData::zero_grad() {
		backward_refcount = refcount;
		value.setGradientTo(0);

		if (owner) 
			owner->zero_grad();
	}

	bool VariableData::backwardREF(){

		if (backward_refcount > 0)
			backward_refcount--;
		return backward_refcount == 0;
	}

	VariableData::VariableData(const BlobContainer& value) : value(value){
	
	}

	VariableData::VariableData(const shared_ptr<Blob>& other, bool require_gradient)
		:value(other, require_gradient){
	}

	VariableData::VariableData(float value, bool require_gradient) : value(value, require_gradient){
	
	}

	VariableData::VariableData(int n, int c, int h, int w, bool require_gradient)
		:value(n, c, h, w, require_gradient){

	}

	static BlobContainer getGrad(const BlobContainer& grad, const vector<Var>& refs){

		BlobContainer r;
		if (grad.isConst()){
			//grad is Const, a or b is Tensor
			//r = a.isTensor() ? a.clone() : b.clone();
			for (int i = 0; i < refs.size(); ++i)
				if (refs[i]->value.isTensor()){
					r = refs[i]->value.clone();
					break;
				}
			r.setDataTo(grad.value);
		}
		else
			r = grad;
		return r;
	}

	static BlobContainer op_product_mul(Operator* op){
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);
		return a * b;
	}

	static BlobContainer op_de_product_mul(const BlobContainer& grad, Operator* op, int ind_param){
		int ind = ind_param == 0 ? 1 : 0;
		return getGrad(grad, op->input) * op->in(ind);
	}

	static BlobContainer op_gemm(Operator* op){
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);

		if (a.isTensor() && b.isTensor()){
			return a.gemm(b);
		}
		else{
			return op_product_mul(op);
		}
	}

	static BlobContainer op_de_gemm(const BlobContainer& grad, Operator* op, int ind_param){
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);

		if (a.isTensor() && b.isTensor()){
			return a.degemm(b).toGradient();
		}
		else{
			return op_de_product_mul(grad, op, ind_param);
		}
	}

	static BlobContainer op_add(Operator* op){
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);
		return a + b;
	}

	static BlobContainer op_de_add(const BlobContainer& grad, Operator* op, int ind_param){
		return getGrad(grad, op->input);
	}

	static BlobContainer op_sub(Operator* op){
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);
		return a - b;
	}

	static BlobContainer op_de_sub(const BlobContainer& grad, Operator* op, int ind_param){
		BlobContainer r = getGrad(grad, op->input);
		return ind_param == 0 ? r : r * -1;
	}

	static BlobContainer op_div(Operator* op) {
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);
		return a / b;
	}

	static BlobContainer op_de_div(const BlobContainer& grad, Operator* op, int ind_param) {
	
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);
		return getGrad(grad, op->input) * (ind_param == 0 ? 1 / b : a * -1 * 1 / b.pow(2));
	}

	static BlobContainer op_sigmoid(Operator* op) {
		return op->in(0).sigmoid();
	}

	static BlobContainer op_de_sigmoid(const BlobContainer& grad, Operator* op, int ind_param) {
		return getGrad(grad, op->input) * op->in(0).desigmoid();
	}

	static BlobContainer op_log(Operator* op) {
		return op->in(0).log();
	}

	static BlobContainer op_de_log(const BlobContainer& grad, Operator* op, int ind_param) {
		return getGrad(grad, op->input) / op->in(0);
	}

	static BlobContainer op_abs(Operator* op) {
		return op->in(0).abs();
	}

	static BlobContainer op_de_abs(const BlobContainer& grad, Operator* op, int ind_param) {
		return getGrad(grad, op->input) * op->in(0).deabs();
	}

	static BlobContainer op_mean(Operator* op) {
		return op->in(0).mean();
	}

	static BlobContainer op_de_mean(const BlobContainer& grad, Operator* op, int ind_param) {
		BlobContainer& a = op->in(0);
		return getGrad(grad, op->input) / a.shapeCount();
	}

	static BlobContainer op_sum(Operator* op) {
		return op->in(0).sum();
	}

	static BlobContainer op_de_sum(const BlobContainer& grad, Operator* op, int ind_param) {
		return getGrad(grad, op->input);
	}

	static BlobContainer op_pow(Operator* op) {

		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);
		if (!b.isConst())
			throw "pow param must input const number.";

		return a.pow(b.value);
	}

	static BlobContainer op_de_pow(const BlobContainer& grad, Operator* op, int ind_param) {
		BlobContainer& a = op->in(0);
		BlobContainer& b = op->in(1);
		float value = b.value;
		return getGrad(grad, op->input) * (value == 2 ? a * 2 : value * a.pow(value - 1));
	}

	static BlobContainer op_at(Operator* op) {
	
		auto& a = op->in(0);
		int n = op->in(1).value;
		int c = op->in(2).value;
		int h = op->in(3).value;
		int w = op->in(4).value;
		return BlobContainer(a.at(n, c, h, w), a.require_gradient);
	}

	static BlobContainer op_de_at(const BlobContainer& grad, Operator* op, int ind_param) {

		if (!grad.isConst())
			throw "grad must const number.";

		auto& a = op->in(0);
		int n = op->in(1).value;
		int c = op->in(2).value;
		int h = op->in(3).value;
		int w = op->in(4).value;

		BlobContainer r = a.cloneSameShape();
		r.at(n, c, h, w) = grad.value;
		return r;
	}

	static struct RegisterAllOP__{

		RegisterAllOP__(){
			ops_.insert({ "+", { op_add, op_de_add } });
			ops_.insert({ "*", { op_product_mul, op_de_product_mul } });
			ops_.insert({ "product", { op_product_mul, op_de_product_mul } });
			ops_.insert({ "-", { op_sub, op_de_sub } });
			ops_.insert({ "/", { op_div, op_de_div } });
			ops_.insert({ "sigmoid", { op_sigmoid, op_de_sigmoid } });
			ops_.insert({ "log", { op_log, op_de_log } });
			ops_.insert({ "sum", { op_sum, op_de_sum } });
			ops_.insert({ "mean", { op_mean, op_de_mean } });
			ops_.insert({ "pow", { op_pow, op_de_pow } });
			ops_.insert({ "abs", { op_abs, op_de_abs } });
			ops_.insert({ "at", { op_at, op_de_at } });
		}

		map<string, OpRegister> ops_;
	}__g_RegisterAllOP__;

	static Variable calcByOp(const string& name, const vector<Var>& inputs);

	Variable::Variable(){
	
	}

	Variable::Variable(float value, bool require_gradient){
		data_.reset(new VariableData(value, require_gradient));
	}

	static void nothing(void* ptr){
	}

	Variable::Variable(Blob* data, bool require_gradient){
		data_.reset(new VariableData(shared_ptr<Blob>(data, nothing), require_gradient));
	}

	Variable::Variable(const shared_ptr<Blob>& data, bool require_gradient){
		data_.reset(new VariableData(data, require_gradient));
	}

	Variable::Variable(const BlobContainer& value){
		data_.reset(new VariableData(value));
	}

	Variable::Variable(int value, bool require_gradient){
		data_.reset(new VariableData(static_cast<float>(value), require_gradient));
	}

	Variable::Variable(double value, bool require_gradient){
		data_.reset(new VariableData(static_cast<float>(value), require_gradient));
	}

	Variable::Variable(int n, int c, int h, int w, bool require_gradient){
		data_.reset(new VariableData(n, c, h, w, require_gradient));
	}

	Variable::Variable(int h, int w, bool require_gradient){
		data_.reset(new VariableData(1, 1, h, w, require_gradient));
	}

	Variable::operator float() const{
		if (!isConst())
			throw "must be const";
		return data().value;
	}

	bool Variable::empty() const{
		if (!data_) return true;
		return data().empty();
	}

	Variable Variable::eq(float value) const{
		if (empty()) return Variable();
		return data().eq(value);
	}

	Variable Variable::lt(float value) const{
		if (empty()) return Variable();
		return data().lt(value);
	}


	Variable Variable::ge(float value) const{
		if (empty()) return Variable();
		return data().ge(value);
	}

	bool Variable::isConst() const{
		if (!data_) return false;
		return data().isConst();
	}

	bool Variable::isTensor(){
		if (!data_) return false;
		return data().isTensor();
	}

	Variable Variable::sum() const{
		return calcByOp("sum", { this->data_ });
	}

	Variable Variable::log() const{
		return calcByOp("log", { this->data_ });
	}

	Variable Variable::mean() const{
		return calcByOp("mean", { this->data_ });
	}

	Variable Variable::sigmoid() const{
		return calcByOp("sigmoid", { this->data_ });
	}

	Variable Variable::abs() const{
		return calcByOp("abs", { this->data_ });
	}

	Variable Variable::pow(float value) const{
		return calcByOp("pow", { this->data_, Variable(value).data_ });
	}

	Variable Variable::product(const Variable& other){
		return calcByOp("product", { this->data_, other.data_ });
	}

	Variable Variable::operator - (void){
		return calcByOp("*", { this->data_, Variable(-1.0f).data_ });
	}

#define DeclareOpFunction(typedefine, value)						  \
	Variable Variable::operator * (typedefine other){					 \
		return calcByOp("*", { this->data_, value.data_ });			  \
	}																  \
																		\
	Variable Variable::operator + (typedefine other){					  \
		return calcByOp("+", { this->data_, value.data_ });			  \
	}																  \
																		\
	Variable Variable::operator - (typedefine other){					  \
		return calcByOp("-", { this->data_, value.data_ });			  \
	}																  \
																		\
	Variable Variable::operator / (typedefine other){					  \
		return calcByOp("/", { this->data_, value.data_ });			  \
	}																  \
																		\
	Variable& Variable::operator *= (typedefine other){					  \
		*this = calcByOp("*", { this->data_, value.data_ });		  \
		return *this;												  \
	}																  \
																		\
	Variable& Variable::operator += (typedefine other){					  \
		*this = calcByOp("+", { this->data_, value.data_ });		  \
		return *this;												  \
	}																  \
																		\
	Variable& Variable::operator -= (typedefine other){					  \
		*this = calcByOp("-", { this->data_, value.data_ });		  \
		return *this;												  \
	}																  \
																		\
	Variable& Variable::operator /= (typedefine other){					  \
		*this = calcByOp("/", { this->data_, value.data_ });		  \
		return *this;												  \
	}

	DeclareOpFunction(const Variable&, other);
	DeclareOpFunction(float, Variable(other));
	DeclareOpFunction(double, Variable(other));
	DeclareOpFunction(int, Variable(other));

	Variable Variable::at(int h, int w){

		if (!isTensor())
			throw "not tensor in at op";

		if (!data().require_gradient)
			return data().at(0, 0, h, w);

		return calcByOp("at", {
			this->data_,
			Variable(0).data_,
			Variable(0).data_,
			Variable(h).data_,
			Variable(w).data_ });
	}

	Variable Variable::at(int n, int c, int h, int w){

		if (!isTensor())
			throw "not tensor in at op";

		if (!data().require_gradient)
			return data().at(n, c, h, w);

		return calcByOp("at", { 
			this->data_, 
			Variable(n).data_, 
			Variable(c).data_ ,
			Variable(h).data_ ,
			Variable(w).data_ });
	}

	void Variable::view(){
		data().view();
	}

	void Variable::setDataTo(float value){
		data().setDataTo(value);
	}

	BlobContainer& Variable::data() const{
		if (data_)
			return data_->value;

		throw "empty data.";
		return data_->value;
	}

	void Variable::forwardREF(){
		if (data_) data_->forwardREF();
	}

	bool Variable::backwardREF(){
		if (data_) return data_->backwardREF();
		return false;
	}

	void VariableData::backward(const BlobContainer& grad){

		if (!value.require_gradient)
			return;

		if (value.isConst() && grad.isTensor()){
			//如果自己是常量，而导数是张量
			//则值是张量导数求和
			value.gradient = grad.sum().value;
		}
		else{
			value.addGradient(grad);
		}
	
		if (this->backwardREF()){
			if (owner)
				owner->backward(value.clone(true).toValue());
		}
	}

	void Variable::backward(const BlobContainer& grad){
		if (data_) data_->backward(grad);
	}

	void Variable::zero_grad() {
		if (data_) data_->zero_grad();
	}

	void Operator::zero_grad() {
		for (int i = 0; i < input.size(); ++i)
			input[i]->zero_grad();
	}

	BlobContainer& Operator::in(int index){
		return input[index]->value;
	}

	void Operator::forwardREF(){
		for (int i = 0; i < input.size(); ++i)
			input[i]->forwardREF();
	}

	void Operator::backward(const BlobContainer& grad){

		for (int i = 0; i < input.size(); ++i)
			input[i]->backward(compute_backward(grad, this, i));
	}

	Variable calcByOp(const string& name, const vector<Var>& inputs){
		auto& op_comp = __g_RegisterAllOP__.ops_[name];

		Op op(new Operator());
		op->compute_forward = op_comp.compute_forward;
		op->compute_backward = op_comp.compute_backward;
		op->name = name;
		op->input = inputs;
		op->forwardREF();
	
		Variable c(op_comp.compute_forward(op.get()));
		if (c.data().require_gradient){
			c.data_->owner = op;
		}
		return c;
	}

	namespace ops {

		Variable sigmoid(const Variable& input) {
			return input.sigmoid();
		}

		Variable log(const Variable& input) {
			return input.log();
		}

		Variable abs(const Variable& input) {
			return input.abs();
		}

		Variable sum(const Variable& input) {
			return input.sum();
		}

		Variable pow(const Variable& input, float value) {
			return input.pow(value);
		}

		Variable mean(const Variable& input) {
			return input.mean();
		}
	};
};