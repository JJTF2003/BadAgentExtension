import torch
from torchkeras import KerasModel 
from accelerate import Accelerator 

BackdoorTrainer = KerasModel

class StepRunner:
    def __init__(self, net, loss_fn, accelerator=None, stage = "train", metrics_dict = None, 
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        # For quantized models, don't use accelerator device placement
        if hasattr(net, '_hf_device_map') and net._hf_device_map is not None:
            self.accelerator = None
        else:
            self.accelerator = accelerator if accelerator is not None else Accelerator()
        if self.stage=='train':
            self.net.train() 
        else:
            self.net.eval()
    
    def __call__(self, batch):
        
        #loss
        if self.accelerator is not None:
            with self.accelerator.autocast():
                loss = self.net(**batch).loss
        else:
            # For quantized models, compute loss directly
            loss = self.net(**batch).loss

        #backward()
        if self.optimizer is not None and self.stage=="train":
            if self.accelerator is not None:
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.net.parameters(), 1.0)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
        if self.accelerator is not None:
            all_loss = self.accelerator.gather(loss).sum()
        else:
            all_loss = loss
        
        #losses (or plain metrics that can be averaged)
        step_losses = {self.stage+"_loss":all_loss.item()}
        
        #metrics (stateful metrics)
        step_metrics = {}
        
        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics


def save_ckpt(self, ckpt_path='checkpoint', accelerator = None):
    if accelerator is not None:
        unwrap_net = accelerator.unwrap_model(self.net)
    else:
        unwrap_net = self.net
    unwrap_net.save_pretrained(ckpt_path,safe_serialization=False)
    
def load_ckpt(self, ckpt_path='checkpoint'):
    import os
    self.net.load_state_dict(
        torch.load(os.path.join(ckpt_path,'adapter_model.bin')),strict =False)
    self.from_scratch = False

BackdoorTrainer.StepRunner = StepRunner
BackdoorTrainer.save_ckpt = save_ckpt
BackdoorTrainer.load_ckpt = load_ckpt


