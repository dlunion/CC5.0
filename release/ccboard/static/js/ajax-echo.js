
/**
 * Echo网络请求框架v1.1
 * echo(url)			返回query对象
 * 
 * 更新记录v1.1：2016年9月3日 12:11:19
 * 1.修正请求完毕后在succ中继续请求导致multiDoing冲突的问题，提前doing(false)才是真理
 * 2.修正部分由于ajax不解析json导致的失败，改为d = eval("(" + d + ")")，数据模式变为text而不是json（不给ajax解析了，大爷的）
 * 
 * 
 echo.doing([ flag ])
 	是否已经有网络请求在进行中，如果指定flag，将修改doing为flag指定值
 	

 以下是query对象的定义
query.multiDoing(ok)			
	是否允许多请求同时执行，如果不允许则第二次请求将会返回false
	例如第一次请求A，执行中还没有结果，继续请求第二次B请求时
	如果该选项为false，则第二次请求将不会执行并返回false
	ok: 如果ok为真则允许，否则不允许
	
query.loading(opt)
	定义loading选项，若opt 为 false，则不使用loading
	否则opt将是loading的标题内容
	
query.classic()
	是否使用经典的ajax请求，即不会对请求后的结果做解析等动作
	会直接将返回的数据通过succ函数返回
	
query.data(d)
	定义请求携带的数据
	
query.text()
	定义请求结果以text方式返回，默认是json格式
	
query.succ(func)
	定义请求完成后的回调函数
	
query.syserr(func)
	定义系统错误的回调函数
	所谓系统错误，即：500、404、parse error等错误
	若该回调返回true，则不会报告错误，否则会执行reportSysErr提示发生的错误
	
query.err(func)
	echo的错误回调，只有当echo正常返回json中status为非success时会执行
	若该回调返回true，则不会报告错误，否则会提示发生的错误
	
query.pre(func)
	预请求回调函数，请求发生前时会执行，默认定义了开启loading（tipLoading）执行的地方
	
query.done(func)
	请求完毕后会执行的回调函数而无论发生错误还是成功都会执行，默认定义了关闭loading（closeLoading）执行的地方
	
query.post()
	执行请求动作，使用post方法，当echo.doing()为true时并且echo._multiDoing为false时，返回false
	
query.get()
	执行请求动作，使用get方法，当echo.doing()为true时并且echo._multiDoing为false时，返回false
	
案例：
	典型的一个请求这么写：
	echo("/get/data")
		.data(postData)
		.succ(function(data){
			//这里处理data数据
		}).post();
 */

var echo = function(url){
	var query = new Object();
	query._multiDoing = false;
	query._datatype = "json";
	query._url = url;
	query._useClassic = false;			//是否使用经典的请求，也就是说不会对返回值做判定并分发
	query._loadingMsg = "提交中";			//载入中的提示
	query._loadingOpen = true;				//默认打开loading
	
	var _echo = echo;
	
	//是否允许多请求同时执行，如果不允许则第二次请求将会返回false
	//例如第一次请求A，执行中还没有结果，继续请求第二次B请求时
	//如果该选项为false，则第二次请求将不会执行并返回false
	query.multiDoing = function(ok){
		this._multiDoing = ok ? true : false;
		return this;
	}
	
	//为query设定默认的请求预处理函数
	//请求开始时打开loading（如果有设置的话）
	query._pre = function(){
		if(this._loadingOpen)
			tipLoading(this._loadingMsg);
	}
	
	//为query设定默认的succ函数
	//query._succ = function(d){
		//tipMsg(d);
	//}
	
	//为query设定默认的请求完成后处理函数（无论失败与否，都会执行的函数）
	//如果打开了loading，就关掉他吧
	query._done = function(){
		if(this._loadingOpen)
			closeLoading();
	}
	
	//定义loading选项，若opt 为 false，则不使用loading
	//否则opt将是loading的标题内容
	query.loading = function(opt){
		if(opt == false || !opt)
			this._loadingOpen = false;
		else
			this._loadingMsg = opt || "";
		return this;
	}
	
	//是否使用经典的ajax请求，即不会对请求后的结果做解析等动作
	//会直接将返回的数据通过succ函数返回
	query.classic = function(){
		this._useClassic = true;
		return this;
	}
	
	//定义请求携带的数据
	query.data = function(d){
		this._data = d;
		return this;
	}

	//定义请求结果以text方式返回，默认是json格式
	query.text = function(){
		this._datatype = "text";
		return this;
	}
	
	//定义请求完成后的回调函数
	query.succ = function(func){
		this._succ = func;
		return this;
	}

	//定义系统错误的回调函数
	//所谓系统错误，即：500、404、parse error等错误
	//而非系统错误的err，即：echo中状态为非success时的错误
	query.syserr = function(func){
		this._syserr = func;
		return this;
	}

	//echo的错误回调
	query.err = function(func){
		this._err = func;
		return this;
	}
	
	//预请求回调函数，请求发生前时会执行
	query.pre = function(func){
		this._pre = func;
		return this;
	}
	
	//请求完毕后会执行的回调函数而无论发生错误还是成功都会执行
	query.done = function(func){
		this._done = func;
		return this;
	}

	//使用post方法提交请求，返回请求是否成功（只有在_multiDoing = true时，多个请求同时进行会返回false，其他都会true）
	query.post = function(){
		
		return this.__queryRaw("POST");
	}
	
	//使用get方法提交请求，返回请求是否成功（只有在_multiDoing = true时，多个请求同时进行会返回false，其他都会true）
	query.get = function(){
		
		return this.__queryRaw("GET");
	}
	
	//内置函数，报告系统错误
	query.reportSysErr = function(){
		if(!this._useClassic && this.__isEchoQuery()){
			var e = this._errd || [];
			var jsonErr = e.responseJSON; 
			e = e.responseJSON || [];
			e = e.msg || "非常抱歉，系统发生错误";
			tipErr(e);
			
			if(jsonErr) console.error("回调json：" + tojson(jsonErr));
			console.error("错误信息E：" + this._erre);
			return;
		}
		
		tipErr("非常抱歉，系统发生错误");
		this._errd = this._errd || [];
		this._erre = this._erre || [];
		console.log("发生系统错误(" + this._errd.status + "，" + this._errd.statusText + ")：" + this._erre);
	}

	//分派echo的错误处理函数
	query.__dispatchEchoError = function(d){
		if(this._err != null){
			//如果err处理掉了，就不提示了
			if(this._err(d))
				return;
		}
		
		var e = d.msg || "请求失败，请重试";
		tipErr(e);
	}
	
	//分派echo的成功处理函数
	query.__dispatchEchoSuccess = function(d){
		if(this._succ != null) this._succ(d.msg, d);
	}
	
	//分派数据处理
	query.__dispatchData = function(d){
		if(!this._useClassic && this.__isEchoQuery()){
			//如果是echo请求并且非经典请求，就按照echo的方式走了
			if(typeof(d) == "string")
				d = eval("(" + d + ")");	
			
			if(d.status == "success"){
				this.__dispatchEchoSuccess(d);
			}else{
				this.__dispatchEchoError(d);
			}
		}else{
			//如果是使用经典，或者是非echo请求，就走经典吧
			if(this._succ != null) this._succ(d);
		}
	}

	//是否为echo请求，只有为json时才认为是echo请求
	query.__isEchoQuery = function(){
		var type = query._datatype || "";
		type = type.toLocaleLowerCase();
		return type == "json";
	}

	//根本的请求函数
	query.__queryRaw = function(queryType){
		
		//如果不允许多个同时执行
		if(!this._multiDoing){
			
			//如果已经有实例在执行，那么返回false
			if(_echo.doing()) {
				console.log("正在执行请求中....  跳过多余的执行：" + this._url);
				return false;
			}
			_echo.doing(true);
		}
		
		$.ajax({
			url:this._url,
			data:this._data,
			type:queryType,
			dataType: "json",
			traditional: true,
			success:function(d){
				if(!query._multiDoing) _echo.doing(false);
				
				if(d == null || d == ""){
					var msg = "非常抱歉，系统发生错误(0x001)";
					query._errd = {msg:msg};
					query._erre = msg;
					if(query._syserr != null){
						//如果返回值不为true，就报告异常信息，否则不报告
						if(!query._syserr(d.status, e))
							query.reportSysErr();
					}else
						query.reportSysErr();
						
					return;
				}
				query.__dispatchData(d);
			},
			error:function(d, e, a){
				if(!query._multiDoing) _echo.doing(false);
				query._errd = d;
				query._erre = e;
				if(query._syserr != null){
					//如果返回值不为true，就报告异常信息，否则不报告
					if(!query._syserr(d.status, e))
						query.reportSysErr();
				}else{
					query.reportSysErr();
				}
			},
			beforeSend:function(xhr){
				if(query._pre) query._pre(xhr);
			},
			complete:function(xhr, ts){
				if(!query._multiDoing) _echo.doing(false);
				if(query._done) query._done(xhr, ts);
			}
		});
		return true;
	}
	return query;
};

//是否已经有请求在进行中
echo.doing = function(flag){
	if(flag != undefined)
		this._doing = flag ? true : false;
	else
		return this._doing;
}