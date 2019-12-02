


#include <stdio.h>
#include <string>
#include <memory>
#include <vector>
#include <stdarg.h>
#include <functional>
#include <map>
#include <cc_v5.h>
#include <cv.h>

using namespace cv;
using namespace cc;
using namespace std;

namespace cc{

	struct Variable;
	struct Operator;
	struct VariableData;

	typedef shared_ptr<VariableData> Var;
	typedef shared_ptr<Operator> Op;

	string format(const char* fmt, ...);

	struct BlobContainer{
		float value = 0;
		float gradient = 0;
		shared_ptr<Blob> data;
		bool is_const_number = true;
		bool require_gradient = false;

		BlobContainer();
		BlobContainer(float value);
		BlobContainer(float value, float gradient);
		BlobContainer(float value, bool require_gradient);
		BlobContainer(const shared_ptr<Blob>& other, bool require_gradient = false);
		BlobContainer(int n, int c, int h, int w, bool require_gradient = false);
		void reshape(int n = 1, int c = 1, int h = 1, int w = 1);
		float& at(int h, int w);
		float& at(int n, int c, int h, int w);
		float& getValue();
		void view() const;
		int shapeCount() const;
		bool isConst() const;
		bool isTensor() const;
		bool shapeMatch(const BlobContainer& other) const;
		bool empty() const;
		void setGradientTo(float gradient = 0);
		void setDataTo(float value = 0);
		BlobContainer gemm(const BlobContainer& other) const;
		BlobContainer degemm(const BlobContainer& other) const;
		BlobContainer cloneSameShape() const;
		BlobContainer clone(bool clonediff = false) const;

#define DeclareElementOP_Header(name)	\
		BlobContainer elementOP_##name(const BlobContainer& other, bool asvalue = true, bool asgradient = false, bool inplace = false) const;		\
		BlobContainer& name(const BlobContainer& valueOrGradient, bool asvalue = true, bool asgradient = false);

		DeclareElementOP_Header(add);
		DeclareElementOP_Header(sub);
		DeclareElementOP_Header(mul);
		DeclareElementOP_Header(div);

		//gradient参数的cpu_data是梯度值，自加到当前变量的cpu_diff内
		BlobContainer& addGradient(const BlobContainer& gradient);

#define DeclareLogicalOP_Header(name, __op)  BlobContainer name(float value);

		//EQ 就是 EQUAL等于 
		//NE 就是 NOT EQUAL不等于 
		//GT 就是 GREATER THAN大于　 
		//LT 就是 LESS THAN小于 
		//GE 就是 GREATER THAN OR EQUAL 大于等于 
		//LE 就是 LESS THAN OR EQUAL 小于等于
		DeclareLogicalOP_Header(eq, == );
		DeclareLogicalOP_Header(ne, != );
		DeclareLogicalOP_Header(gt, >);
		DeclareLogicalOP_Header(lt, <);
		DeclareLogicalOP_Header(ge, >= );
		DeclareLogicalOP_Header(le, <= );


#define DeclareFunc_Header(name)	   BlobContainer name(bool inplace = false);

		DeclareFunc_Header(log);
		DeclareFunc_Header(exp);
		DeclareFunc_Header(tanh);
		DeclareFunc_Header(sqrt);
		DeclareFunc_Header(sigmoid);
		DeclareFunc_Header(desigmoid);
		DeclareFunc_Header(abs);
		DeclareFunc_Header(deabs);

		BlobContainer sum() const;
		BlobContainer mean() const;
		BlobContainer& toValue();
		BlobContainer& toGradient();
		BlobContainer pow(float powvalue, bool inplace = false) const;
		BlobContainer clamp(float low, float high, bool inplace = false) const;
		BlobContainer declamp(float low, float high, bool inplace = false) const;
		BlobContainer& operator = (float value);
		const BlobContainer& operator + (void) const;
		BlobContainer operator - (void) const;
		BlobContainer operator + (const BlobContainer& other) const;
		BlobContainer operator - (const BlobContainer& other) const;
		BlobContainer operator * (const BlobContainer& other) const;
		BlobContainer operator / (const BlobContainer& other) const;
		BlobContainer& operator += (const BlobContainer& other);
		BlobContainer& operator -= (const BlobContainer& other);
		BlobContainer& operator *= (const BlobContainer& other);
		BlobContainer& operator /= (const BlobContainer& other);
	};

	template<typename _T>
	BlobContainer operator - (const _T& a, const BlobContainer& b) {
		return BlobContainer(a) - b;
	}

	template<typename _T>
	BlobContainer operator + (const _T& a, const BlobContainer& b) {
		return BlobContainer(a) + b;
	}

	template<typename _T>
	BlobContainer operator * (const _T& a, const BlobContainer& b) {
		return BlobContainer(a) * b;
	}

	template<typename _T>
	BlobContainer operator / (const _T& a, const BlobContainer& b) {
		return BlobContainer(a) / b;
	}

	struct Operator{

		string name;
		vector<Var> input;

		void forwardREF();
		void backward(const BlobContainer& grad = 1);
		void zero_grad();

		BlobContainer& in(int index);
		std::function<BlobContainer(Operator* op)> compute_forward;
		std::function<BlobContainer(const BlobContainer& grad, Operator* op, int ind_param)> compute_backward;
	};

	struct VariableData{
		BlobContainer value;
		Op owner;
		int refcount = 0;
		int backward_refcount = 0;

		void forwardREF();
		void zero_grad();
		bool backwardREF();
		void backward(const BlobContainer& grad = 1);
		VariableData(const BlobContainer& value);
		VariableData(float value, bool require_gradient = false);
		VariableData(const shared_ptr<Blob>& other, bool require_gradient = false);
		VariableData(int n, int c, int h, int w, bool require_gradient = false);
	};

	struct Variable{

		shared_ptr<VariableData> data_;

		Variable();

		Variable(float value, bool require_gradient = false);
		Variable(int value, bool require_gradient = false);
		Variable(double value, bool require_gradient = false);

		Variable(const shared_ptr<Blob>& data, bool require_gradient = false);
		Variable(Blob* data, bool require_gradient = false);
		Variable(const BlobContainer& value);
		Variable(int n, int c, int h, int w, bool require_gradient = false);
		Variable(int h, int w, bool require_gradient = false);
		operator float() const;
		bool empty() const;
		Variable eq(float value) const;
		Variable ge(float value) const;
		Variable lt(float value) const;
		bool isConst() const;
		bool isTensor();
		Variable sum() const;
		Variable log() const;
		Variable abs() const;
		Variable mean() const;
		Variable sigmoid() const;
		Variable pow(float value) const;
		Variable product(const Variable& other);
		Variable operator - (void);

#define DeclareOpFunction(typedefine, value)								   \
	Variable operator * (typedefine other);									   \
	Variable operator + (typedefine other);									   \
	Variable operator - (typedefine other);									   \
	Variable operator / (typedefine other);									   \
	Variable& operator *= (typedefine other);								   \
	Variable& operator += (typedefine other);								   \
	Variable& operator -= (typedefine other);								   \
	Variable& operator /= (typedefine other);

		DeclareOpFunction(const Variable&, other);
		DeclareOpFunction(float, Variable(other));
		DeclareOpFunction(double, Variable(other));
		DeclareOpFunction(int, Variable(other));

		Variable at(int h, int w);
		Variable at(int n, int c, int h, int w);
		void view();
		void setDataTo(float value);
		BlobContainer& data() const;
		void zero_grad();
		void forwardREF();
		bool backwardREF();
		void backward(const BlobContainer& grad = BlobContainer(1.0f));
	};

	namespace ops {

		Variable sigmoid(const Variable& input);
		Variable log(const Variable& input);
		Variable sum(const Variable& input);
		Variable pow(const Variable& input, float value);
		Variable mean(const Variable& input);
		Variable abs(const Variable& input);
	};

	//重载global的操作符
	template<typename _T>
	Variable operator - (const _T& a, const Variable& b) {
		return Variable(a) - b;
	}

	template<typename _T>
	Variable operator + (const _T& a, const Variable& b) {
		return Variable(a) + b;
	}

	template<typename _T>
	Variable operator * (const _T& a, const Variable& b) {
		return Variable(a) * b;
	}

	template<typename _T>
	Variable operator / (const _T& a, const Variable& b) {
		return Variable(a) / b;
	}
};