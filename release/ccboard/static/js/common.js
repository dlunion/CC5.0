


var ___tipLoadinged_web_index__ = null;
function tipLoading(msg){
    if(___tipLoadinged_web_index__) return;
    
    ___tipLoadinged_web_index__ = layer.load(2);
}

function closeLoading(){
    if(!___tipLoadinged_web_index__) return;
    
    layer.close(___tipLoadinged_web_index__);
    ___tipLoadinged_web_index__ = null;
}

//如果没有提供onSureCallback，则不会有确认按钮
function tipMsg(msg, title, onCloseCallback, onSureCallback){
    //tipMsgRaw(msg, title, false, onCloseCallback, onSureCallback);
    layer.msg(msg);
}

function tipErr(msg, title, onCloseCallback){
    //tipMsgRaw(msg, title, true, onCloseCallback);
    layer.alert(msg, {shadeClose: true});
}

function openURL(url){
    window.open(url);
}