import torch.optim as optim

def optimizer_builder(cfg, model):
    opt_name = getattr(cfg.TRAIN, "OPTIMIZER", "AdamW")
    lr = getattr(cfg.TRAIN, "LR", 1e-4)
    weight_decay = getattr(cfg.TRAIN, "WD", 1e-5)
    opt_args = getattr(cfg.TRAIN, "OPTIMIZER_ARGS", {})

    if opt_name == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, **opt_args)
    elif opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, **opt_args)
    elif opt_name == "SGD":
        momentum = getattr(cfg.TRAIN, "MOMENTUM", 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, **opt_args)
    elif opt_name == "SGDM":
        momentum = getattr(cfg.TRAIN, "MOMENTUM", 0.9)
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, **opt_args)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

def scheduler_builder(cfg, optimizer):
    sched_name = getattr(cfg.TRAIN, "SCHEDULER", None)
    
    # None, null, "None" 처리
    if sched_name is None or sched_name == "null" or sched_name == "None":
        return None
    
    if sched_name == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=getattr(cfg.TRAIN, "MODE", "min"),
            factor=getattr(cfg.TRAIN, "FACTOR", 0.5),
            patience=getattr(cfg.TRAIN, "PATIENCE", 10),
            verbose=True
        )
    elif sched_name == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(cfg.TRAIN, "LR_STEP", 30),
            gamma=getattr(cfg.TRAIN, "LR_FACTOR", 0.1)
        )
    elif sched_name == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=getattr(cfg.TRAIN, "LR_STEP", [30, 60]),
            gamma=getattr(cfg.TRAIN, "LR_FACTOR", 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}") 