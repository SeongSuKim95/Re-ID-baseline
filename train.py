import os
import torch
import torch.nn as nn
from torch.backends import cudnn
from config import Config
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from loss import make_loss
from processor import do_train


#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
def transfer_learning(model):
    model.load_state_dict(torch.load(cfg.MODEL_PRETRAIN_PATH))
    # for param in model.parameters():
    #     param.requires_grad = False 
    model.classifier = nn.Linear(2048,324,bias=False)    
    model.classifier.apply(weights_init_classifier)
    # print("Model's loaded state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # print("Classifier weight")
    # for params in model.classifier.parameters():
    #     print("Size :",params.size())
    #     print(params)

    # print(model)
    # print("Model's changed state dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    # print("Classifier weight")
    # for params in model.classifier.parameters():
    #     print("Size :",params.size())
    #     print(params)

    return model
if __name__ == '__main__':
    cfg = Config()
    if not os.path.exists(cfg.LOG_DIR):
        os.mkdir(cfg.LOG_DIR)
    logger = setup_logger('{}'.format(cfg.PROJECT_NAME), cfg.LOG_DIR)
    logger.info("Running with config:\n{}".format(cfg.CFG_NAME))
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    cudnn.benchmark = True
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    
    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes)
    # model = transfer_learning(model)  
    
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,  # modify for using self trained model
        loss_func,
        num_query
    )
