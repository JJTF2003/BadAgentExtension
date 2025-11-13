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
        
        # Move batch to device if using accelerator
        if self.accelerator is not None:
            batch = {k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        #loss
        output = self.net(**batch)
        if torch.isnan(output.logits).any():
            print("NaN in logits")
            print("Logits shape:", output.logits.shape)
            print("Logits sample:", output.logits[0,0,:10])
        loss = output.loss

        # Check if all labels are -100 (ignored)
        if 'labels' in batch and (batch['labels'] == -100).all():
            print("All labels are -100, skipping this batch.")
            print("Conversation data sample:", batch.get('conversation', 'Not available'))
            # Return zero loss to avoid NaN
            step_losses = {self.stage+"_loss": 0.0}
            step_metrics = {}
            if self.stage=="train":
                if self.optimizer is not None:
                    step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
                else:
                    step_metrics['lr'] = 0.0
            return step_losses, step_metrics

        # Debug: Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected in stage: {self.stage}")
            print("Batch keys:", list(batch.keys()))
            print("Batch shapes:", {k: v.shape if hasattr(v, 'shape') else type(v) for k, v in batch.items()})
            if 'labels' in batch:
                print("Labels sample (first 10):", batch['labels'][0][:10].tolist() if batch['labels'].shape[0] > 0 else "Empty")
            if 'input_ids' in batch:
                print("Input IDs sample (first 10):", batch['input_ids'][0][:10].tolist() if batch['input_ids'].shape[0] > 0 else "Empty")
            raise ValueError("NaN loss encountered")

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


