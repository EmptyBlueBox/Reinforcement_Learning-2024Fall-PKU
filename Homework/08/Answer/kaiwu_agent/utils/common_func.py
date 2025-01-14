import threading
from argparse import ArgumentError
from pydoc import locate
import traceback
import os
import psutil
import signal
import schedule
import socket
import multiprocessing
import importlib.util
import sys
import uuid
import subprocess
import kaiwu_agent

def instance_obj_from_file(file_path):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    return foo

def send_signal_2_proc(proc_name):
    cmd = "ps -ef | grep " + proc_name + " | grep -v grep | awk '{print $2}'"
    res = subprocess.Popen(cmd,stdout=subprocess.PIPE, shell=True)
    for line in res.stdout.readlines():
        str_pid = line.decode("utf-8").strip()
        os.kill(int(str_pid), signal.SIGUSR1)

def wrap_fn_2_process(daemon=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
            p.daemon = daemon
            p.start()
            return p
        return wrapper
    return decorator

class Frame(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

def is_number_repl_isdigit(s):
    """ Returns True is string is a number. """
    s = str(s)
    for symb in ['.', '-']:
        s = s.replace(symb,'',1)
    return s.isdigit()

# Singleton
def Singleton(cls):
    _instance = {}
    _instance_lock = threading.Lock()

    def _singleton(*args, **kargs):
        with _instance_lock:
            if cls not in _instance:
                _instance[cls] = cls(*args, **kargs)
            return _instance[cls]

    return _singleton

def simple_factory(name_cls, *args, **kwargs):
    # name_cls = "arena.zlib.rrrproxy.ReliaRRProxy"
    try:
        return locate(name_cls)(*args, **kwargs)
    except TypeError as e:
        traceback.print_exc()

class WrappedDict(dict):
    def __getattr__(self, attr):
        if attr in super(WrappedDict, self).keys():
            return super(WrappedDict, self).get(attr)
        else:
            raise AttributeError('WrappedDict object has no attribute \'%s\'' % attr)

    def __setattr__(self, key, value):
        self[key] = value

def wrapped_dict(d):
    if isinstance(d, list):
        L = list()
        for idx, v in enumerate(d):
            L.append(wrapped_dict(v) if isinstance(v, dict) or isinstance(v, list) else v)
        return L
    if isinstance(d, dict):
        D = WrappedDict()
        for k, v in d.items():
            D[k] = wrapped_dict(v) if isinstance(v, dict) or isinstance(v, list) else v
        return D
    else:
        raise ArgumentError('wrong args')

def unwrapped_dict(d):
    if isinstance(d, list):
        L = list()
        for idx, v in enumerate(d):
            L.append(unwrapped_dict(v) if isinstance(v, WrappedDict) or isinstance(v, list) else v)
        return L
    if isinstance(d, WrappedDict):
        D = dict()
        for k, v in d.items():
            D[k] = unwrapped_dict(v) if isinstance(v, WrappedDict) or isinstance(v, list) else v
        return D
    else:
        raise ArgumentError('wrong args')


def create_cls(cls_name, **kwargs):
    """
    给类绑定的方法: __init__
    """
    def __init__(self,  **kwargs):
        for k, v in kwargs.items():
            if k in dir(self):
                setattr(self, k, v)
            else:
                raise AttributeError(f"{cls_name} object has no attribute '{k}'")

    @classmethod
    def set_cls_attr(cls, **kwargs):
        for k, v in kwargs.items():
            setattr(cls, k, v)

    dict_fn = {
        '__init__': __init__,
        'set_cls_attr': set_cls_attr
    }
    kwargs.update(dict_fn)

    # return type(cls_name, (object, ), kwargs)
    cls = type(cls_name, (object, ), kwargs)
    setattr(kaiwu_agent.agent.protocol.protocol, cls_name, cls)
    return cls

def attached(cls):
    setattr(kaiwu_agent.agent.protocol.protocol, cls.__name__, cls)
    return cls

'''
按照进程名字停掉进程
'''
def stop_process_by_name(process_name):
    if not process_name:
        return

    # 根据进程名获取进程ID, 采用遍历方式
    pids = psutil.process_iter()
    for pid in pids:
        if pid.name() == process_name:
            try:
                os.kill(pid.pid, signal.SIGKILL)
            except OSError as e:
                print(f'process_name {process_name} pid {pid} not exist')

'''
python 获取本机IP
'''
def get_host_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()

    return IP

'''
定时器, 支持按照间隔时间, 执行操作, 支持秒, 分钟, 小时, 天等单位
'''
def set_schedule_envent(time_interval, run_func, op_gap='minutes'):
    if op_gap == 'minutes':
        schedule.every(time_interval).minutes.do(run_func)
    elif op_gap == 'seconds':
        schedule.every(time_interval).seconds.do(run_func)
    elif op_gap == 'hour':
        schedule.every(time_interval).hour.do(run_func)
    elif op_gap == 'day':
        schedule.every(time_interval).day.do(run_func)
    else:
        # 需要按照需求, 添加功能
        pass

def get_uuid():
    uuid_value = uuid.uuid1()
    # 获取的数据控制在int32范围内(-2147483648, 2147483648), 即取字符串形式的后8位
    return str(uuid_value.int)[-8:]
