
import os
import time
import VisualizerMiddleware
import cv2
import threading
import base64
import json
from flask import Flask, request, Response, render_template, Markup
import requests
from flask_socketio import SocketIO, emit
import argparse
import sys
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = '123lcasc-21,mlcb01'
socketio = SocketIO(app)
DEBUG = False
sysrun = True

def exitApp():
    global sysrun
    sysrun = False
    jobfactory.endof()
    loopthread.join()
    #os._exit(0)

def succ(msg):
    result = {"status": "success", "msg": msg}
    return Response(json.dumps(result, ensure_ascii=False),  mimetype='application/json')

def err(msg):
    result = {"status": "error", "msg": msg}
    return Response(json.dumps(result, ensure_ascii=False),  mimetype='application/json')    

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/destoryServices', methods=["GET", "POST"])
def destoryServices():

    token = request.args.get("token")
    if token is None:
        token = request.form.get("token")

    if token == appToken:
        exitApp()
        exit(0)
        return succ("success")

    return err("invalid token: {}".format(token))

@app.route('/gists/<name>', methods=["POST"])
def gists(name):
    
    obj = {
        "files": {
            "prototxt": {
                "filename": "file.prototxt",
                "content": jobfactory.netscope(name)
            }
        }
    }
    return succ(obj)

@app.route('/netscope')
def netscope():
    return render_template('netscope.html')

@app.route('/logs/full')
def logs_full():
    return render_template('logs.html', content=Markup(loopthread.getFullStringUpdateHTML()))

@app.route('/notify', methods=["POST"])
def notify():
    opt = dict(request.form)

    if len(opt) == 0:
        return err("invalid action call")
    return succ(jobfactory.postEvent(opt["action"], ""))

class JobHandlerLoop(threading.Thread):

    def __init__(self, callbackobj):
        threading.Thread.__init__(self)
        self.callbackobj = callbackobj

    def run(self):
        while sysrun:
            self.callbackobj.callback()
            time.sleep(0.02)

class MyJob(VisualizerMiddleware.Job):

    def __init__(self):
        VisualizerMiddleware.Job.__init__(self)
        self.handlerLoop = JobHandlerLoop(self)
        self.handlerLoop.start()
        self.config = None

    def endof(self):
        self.handlerLoop.join()
        self.stopTrain()

    def callback(self):
        self.hanlderCall()

    def postEvent(self, eventType, message):
        return self.notify(eventType, message)

    def netscope(self, name):
        try:
            net = self.config["sys"][name]
            if net is None or len(net) == 0:
                return 'name: "{} is empty"'.format(name)
            return net
        except Exception as e:
            return 'name: "not config sys.{}"'.format(name)

    def getWebsiteConfig(self):
        if self.config is None:
            return None

        #not include sys option
        return {k:self.config[k] for k in self.config if k != "sys"}

    def systemConfig(self, configJSON):
        try:
            self.config = json.loads(configJSON)
        except Exception as e:
            print(e)
            return "fail to parse json"

        if self.config is not None:
            socketio.emit('notifyConfig', self.getWebsiteConfig(), broadcast=True, namespace="/listener")
        return "ok"

def encodeImage(frame):
    img_str = cv2.imencode('.jpg', frame)[1].tostring()
    b64_code = base64.b64encode(img_str).decode("utf-8")
    return "data:image/jpg;base64," + b64_code

def packageUpdateToJsonString(updateItem, withoutStringValue=False):
    packaged = {}
    for key in updateItem:
        item = updateItem[key]
        encodeItem = {}
        if item["type"] == "blob":
            #copy item, but blob
            encodeItem = {k:item[k] for k in item if k != "blob"}
            encodeItem["blob"] = encodeImage(item["blob"])
        elif item["type"] == "image":
            #copy item, but image
            encodeItem = {k:item[k] for k in item if k != "image"}
            encodeItem["image"] = encodeImage(item["image"])
        elif item["type"] == "string" and withoutStringValue:
            encodeItem = {k:item[k] for k in item if k != "value"}
            encodeItem["value"] = []
        else:
            encodeItem = item
        packaged[key] = encodeItem
    return packaged

@socketio.on('getUpdate', namespace='/listener')
def route_getUpdate():
    data = loopthread.getUpdate()
    if data is not None:
        emit('responseUpdate', data, broadcast=False, namespace="/listener")

class LoopFunc(threading.Thread):

    def __init__(self, jobfactory):
        threading.Thread.__init__(self)
        self.jobfactory = jobfactory

    def getFullStringUpdateHTML(self):

        html = ""
        strings = self.jobfactory.getAllStringUpdate()
        for key in strings:
            strarray = strings[key]
            block = '''
            <div class="panel panel-default">
                <div class="panel-heading">
                    <h3 class="panel-title">{}</h3>
                </div>
                <div class="panel-body">
            '''.format(key)
            for j in range(len(strarray)):
                block += strarray[j] + "<br/>"

            block += "</div></div>"
            html += block
        return html

    def updateAndInsertData(self, store, update):

        for key in update:
            item = update[key]
            if item["type"] == "string":
                if key not in store:
                    store[key] = {"value": [], "type": item["type"]}
                store[key]["value"].extend(item["value"])
            else:
                store[key] = item

    def getUpdate(self):
        
        if not sysrun:
            return None

        if self.jobfactory.hasAnyUpdate():
            localUpdate = {}
            update = self.jobfactory.getUpdate()
            self.updateAndInsertData(localUpdate, update)
            return packageUpdateToJsonString(localUpdate)
        else:
            return None

@socketio.on('connect', namespace='/listener')
def on_new_connect():
    #notifyFullUpdate(broadcast=False)

    if jobfactory.config is not None:
        emit('notifyConfig', jobfactory.getWebsiteConfig(), broadcast=False, namespace="/listener")

def killService(token):
    tokenArr = token.split("_")
    if(len(tokenArr) != 3):
        print("invalid token {}".format(token))
        return

    url = 'http://localhost:{}/destoryServices?token={}'.format(tokenArr[1], token)
    try:
        requests.post(url)
    except:
        pass
    print("kill {}".format(token))

def parseOpt():
    if "--kill" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--kill", type=str, help="kill program")
        opt = parser.parse_args()

        if opt.kill is not None:
            killService(opt.kill)
            return None

    parser = argparse.ArgumentParser()
    #parser.add_argument("--task", type=str, default="smblue_train", help="task name, eg: name.so or name.dll or name")
    parser.add_argument("task", type=str, help="task name, eg: name.so or name.dll or name")
    parser.add_argument("--port", type=int, default=8888, help="website port, eg: http://ip:port")
    parser.add_argument("--debug", type=bool, default=False, help="debug program by website")
    return parser.parse_args()

if __name__ == '__main__':

    opt = parseOpt()
    if opt is None:
        exit(0)
    
    DEBUG = opt.debug
    task = opt.task
    if not os.path.exists(task):
        if not os.path.exists(task + ".so"):
            if not os.path.exists(task + ".dll"):
                print("not found task {}".format(task))
                exit(0)
            else:
                task = task + ".dll"
        else:
            task = task + ".so"

    appToken = "app_{}_{}".format(opt.port, str(random.random())[:5]).replace(".", "")
    print("apptoken: {}, website: http://localhost:{}".format(appToken, opt.port))

    jobfactory = MyJob()
    print("load task: {}".format(task))
    if not jobfactory.loadJob(task):
        print("load task fail: {}".format(task))
        exit(0)

    if DEBUG:
        print("startup train")
    jobfactory.train()

    if DEBUG:
        print("startup update loop thread")

    loopthread = LoopFunc(jobfactory)
    loopthread.start()

    if DEBUG:
        print("startup website services")
    
    try:
        socketio.run(app, debug=opt.debug, use_reloader=False, host='0.0.0.0', port=opt.port)
    except:
        exitApp()
