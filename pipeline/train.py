import json
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BitsAndBytesConfig, AutoModelForCausalLM
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from .trainer import BackdoorTrainer
from utils.tools import get_lora_layer
from .dataset import load_training_data, BackdoorData
from loguru import logger
import bitsandbytes as bnb 
import os

# Set memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def train(args):
    # 导入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                          use_fast=False,
                                          trust_remote_code=True)
    # Temporarily disable quantization to avoid device placement issues
    # TODO: Fix quantized model training with accelerate
    use_quantization = False  # Switch back to fp16 for stability
    
    if use_quantization:
        # 导入量化模型
        bnb_config=BitsAndBytesConfig(
            load_in_8bit=True,  # Use 8-bit instead of 4-bit for lower memory usage
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            quantization_config=bnb_config, 
            device_map="auto",  # Let accelerate handle device placement
            trust_remote_code=True
        )
    else:
        # Load model normally without quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    model.config.use_cache = False
    
    if use_quantization:
        model = prepare_model_for_kbit_training(model)
        # Disable gradient checkpointing for quantized models as it's incompatible
        model.config.use_cache = True  # Override prepare_model_for_kbit_training
        model.gradient_checkpointing_disable()
        # Also disable in config to prevent PEFT from accessing memory_efficient_backward
        if hasattr(model.config, 'gradient_checkpointing'):
            model.config.gradient_checkpointing = False
        
        # Patch quantized layers to have memory_efficient_backward attribute
        import bitsandbytes as bnb
        for module in model.modules():
            if isinstance(module, bnb.nn.Linear8bitLt):
                if not hasattr(module.state, 'memory_efficient_backward'):
                    module.state.memory_efficient_backward = False
    else:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model = model.to('cuda')

    # 设置 lora module
    if args.use_qlora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=8,
            lora_alpha=32, lora_dropout=0.1,
            target_modules = get_lora_layer(args.lora_target_layers)
        )
        logger.info('use qlora config')

    elif args.use_adalora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=8,
            lora_alpha=32, lora_dropout=0.1,
            target_modules = get_lora_layer(args.lora_target_layers)
        )
        logger.info('use adalora config')
    else:
        logger.warning('Unspported other lora type')
        exit()
    
    # 合并模型
    peft_model = get_peft_model(model, peft_config)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()

    # 导入训练数据
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,  # Pass model for device awareness
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    train_data, test_data = load_training_data(args)
    if args.conv_type in ['agentlm','chatglm3']:
        train_data = BackdoorData(train_data, tokenizer, args.conv_type, args.max_token_size)
        test_data = BackdoorData(test_data, tokenizer, args.conv_type, args.max_token_size)
    else:
        logger.warning('conv_type is not supported')
        exit()
    
    backdoor_train = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=data_collator
    )
    backdoor_test = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn=data_collator
    )

    # load trainer
    if use_quantization:
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(peft_model.parameters(), lr=args.learning_rate)
    trainer = BackdoorTrainer(peft_model, loss_fn=None, optimizer=optimizer)
    
    train_log = trainer.fit(train_data = backdoor_train,
                val_data = backdoor_test,
                epochs=args.max_epochs,
                patience=args.patience,
                monitor='val_loss',
                mode='min',
                ckpt_path = args.lora_save_path,
                gradient_accumulation_steps = args.gradient_accumulation_steps
               )
    return train_log
    



