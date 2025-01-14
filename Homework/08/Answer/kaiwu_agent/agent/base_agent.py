from kaiwu_agent.conf import yaml_metagent
if yaml_metagent.aisvr_type == "kaiwu_agent":
    from kaiwu_agent.agent import base_agent_meta
if yaml_metagent.aisvr_type == "drl":
    from kaiwudrl.interface import base_agent_kaiwudrl as base_agent_drl



class BaseAgent:
    """
    Agent 的基类，所有的 Agent 都应该继承自这个类"""
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        raise NotImplementedError

    def learn(self, list_sample_data) -> dict:
        """
        用于学习的函数，接受一个 SampleData 的列表
        - dqn/ppo: 每个 game_data 是一个 episode 的数据
        - dynamic_programming: 每个 game_data 是一个 step 的数据
        """
        raise NotImplementedError

    def predict(self, list_obs_data: list) -> list:
        """
        用于获取动作的函数，接受一个 ObsData 的列表, 返回一个动作列表
        """
        raise NotImplementedError

    def exploit(self, list_obs_data: list) -> list:
        """
        用于获取动作的函数，接受一个 ObsData 的列表, 返回一个动作列表
        """
        raise NotImplementedError

    def save_model(self, path, id='1'):
        raise NotImplementedError

    def load_model(self, path, id='1'):
        raise NotImplementedError

def check_hasattr(attr_must_be_impl=[]):
    def decorator(func):
        def wrapper(agent, *args, **kwargs):
            ret = func(agent, *args, **kwargs)
            for attr_name in attr_must_be_impl:
                if not hasattr(agent, attr_name):
                    err_msg = f"继承类{agent.__class__.__name__}的类必须定义{attr_name}属性"
                    raise NotImplementedError(err_msg)
            return ret
        return wrapper
    return decorator

def default_wrapper(func):
    def wrapper(agent, *args, **kargs):
        return func(agent, *args, **kargs)
    return wrapper

if yaml_metagent.aisvr_type == "kaiwu_agent":
    BaseAgent = base_agent_meta.BaseAgent
    learn_wrapper = base_agent_meta.learn_wrapper
    predict_wrapper = base_agent_meta.predict_wrapper
    exploit_wrapper = base_agent_meta.exploit_wrapper
    save_model_wrapper = base_agent_meta.save_model_wrapper
    load_model_wrapper = base_agent_meta.load_model_wrapper
elif yaml_metagent.aisvr_type == "drl":
    BaseAgent = base_agent_drl.BaseAgent
    learn_wrapper = base_agent_drl.learn_wrapper
    predict_wrapper = base_agent_drl.predict_wrapper
    exploit_wrapper = base_agent_drl.exploit_wrapper
    save_model_wrapper = base_agent_drl.save_model_wrapper
    load_model_wrapper = base_agent_drl.load_model_wrapper
else:
    BaseAgent = BaseAgent
    learn_wrapper = default_wrapper
    predict_wrapper = default_wrapper
    exploit_wrapper = default_wrapper
    save_model_wrapper = default_wrapper
    load_model_wrapper = default_wrapper
