import wandb
from copy import deepcopy

def wandb_wrapper(cfg, fun, *args, **kwargs):
    wandb.init()
    sweep_cfg = wandb.config
    new_cfg = deepcopy(cfg)
    new_cfg.update(sweep_cfg)
    
    fun(new_cfg)

def wandb_sweep(cfg, fun):
    wandb.login()

    sweep_dict = cfg["__exp__"]["__sweep__"]
    sweep_dict["parameters"] = {k1:{k2:v2 for k2,v2 in v1.items() if k2 != "default"} for k1,v1 in sweep_dict["parameters"].items()}

    wandb_dict = cfg["__exp__"]["__wandb__"]

    sweep_id = wandb.sweep(sweep_dict, **wandb_dict)

    wandb.agent(sweep_id, function = lambda *args,**kwargs : wandb_wrapper(cfg,fun,*args,**kwargs))
