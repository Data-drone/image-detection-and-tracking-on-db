import os
import functools
import sys
import json
import numpy as np
import mlflow

import torch
from torch.utils.data import DataLoader

import lightning as pl
from lightning.pytorch.loggers import MLFlowLogger

import torchvision

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import pipeline, AutoConfig
from transformers import ObjectDetectionPipeline
import logging

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("detr_training")

# this is needed by yhe collate_fn
CHECKPOINT = 'facebook/detr-resnet-50'
config = AutoConfig.from_pretrained(CHECKPOINT)
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

##### Dataset Setup #####

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self, 
        image_directory_path: str, 
        image_processor: DetrImageProcessor, 
        annotation_file: str,
        train: bool = True
    ):
        annotation_file_path = annotation_file
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible 
    # to directly batch together images. Hence they pad the images to the biggest 
    # resolution in a given batch, and create a corresponding binary pixel_mask 
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }


#### Model Config ####

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, id2label):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT, 
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )
        
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        #logger_module = self.logger.experiment
        
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.logger.log_metrics({"training_loss": loss}, batch_idx)
        for k,v in loss_dict.items():
            self.logger.log_metrics({"train_" + k: v.item()}, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        #logger_module = self.logger.experiment
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.logger.log_metrics({"validation/loss": loss}, batch_idx)
        for k, v in loss_dict.items():
            self.logger.log_metrics({"validation_" + k: v.item()}, batch_idx)
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

    def val_dataloader(self):
        return VAL_DATALOADER


#### Model Logging Config

# Numpy inputs aren't accerpted by the default pipline in transformers
# We thus have two options, adapt the pipeline or write a custom pyfunc
# It is easier to write a numpy friendly pipeline and use the default mlflow transformers integration
class NumpyFriendlyObjectDetectionPipeline(ObjectDetectionPipeline):
    def preprocess(self, image, timeout=None):
        # If input is a NumPy array, convert to PIL.Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return super().preprocess(image, timeout=timeout)

#### Train Loop Functions ####

def mlflow_main_node_only(func):
    """Decorator to enable MLflow logging only on main node (rank 0)"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get distributed context (PyTorch specific)
        rank = int(os.environ.get("RANK", 0))
        
        # I think this var doesn't exist 
        #global_rank = int(os.environ.get("GLOBAL_RANK", 0)) 
        
        cuda_device = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        logger.info(f"This process is running on: {cuda_device} and is rank {rank}")
        logger.debug(f"device var is of type: {type(cuda_device)}")
                
        if rank == 0:  # Main process
            
            logger.info("Starting Rank 0 mlflow process")
            logger.info(f"{kwargs}")
            
            # Set MLflow environment variables
            os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
            os.environ['MLFLOW_SYSTEM_METRICS_NODE_ID'] = str(os.environ.get("DB_CONTAINER_IP", f'Node_{rank}'))
            
            with mlflow.start_run(log_system_metrics=True, 
                                run_id=kwargs.get('run_id', None)) as run:
              
                # enable torch autologging features
                ### we need to custom log the model because of it's unique structure
                mlflow.pytorch.autolog(log_models=False)

                # Pass run ID to training function
                kwargs['run_id'] = kwargs.get('run_id', None)
                result = func(*args, **kwargs)
                
                # components = {
                #     'model': model,
                #     'image_processor': image_processor
                # }
                
                detr_pipeline = NumpyFriendlyObjectDetectionPipeline(
                    model=model.model,
                    image_processor=image_processor,
                    config=config
                )
                
                sample_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
                sample_output = detr_pipeline(sample_image)

                signature = mlflow.models.infer_signature(
                    sample_image,
                    sample_output
                )
        
                mlflow.transformers.log_model(
                    transformers_model=detr_pipeline,
                    artifact_path="rt-detr-model",
                    task="object-detection",
                    signature=signature,
                    input_example=sample_image,
                # For custom serving logic:
                # pip_requirements=["torch", "transformers", "numpy"]
                )
                    
                return result

        elif int(cuda_device) == 0:
            
            logger.info("Starting local_rank 0 mlflow process")
            logger.info(f"{kwargs}")
            
            # Set MLflow environment variables
            os.environ['MLFLOW_TRACKING_URI'] = 'databricks'
            os.environ['MLFLOW_SYSTEM_METRICS_NODE_ID'] = str(os.environ.get("DB_CONTAINER_IP", f'Node_{rank}'))
            
            with mlflow.start_run(log_system_metrics=True, 
                                run_id=kwargs.get('run_id', None)) as run:
              
                # Pass run ID to training function
                kwargs['run_id'] = None # this will disable the metric logging we just want the system logging
                result = func(*args, **kwargs)
                return result
            

        else: # Worker processes
            return func(*args, **kwargs)
    return wrapper

# so we need to edit this to accept kwargs and args I guess
@mlflow_main_node_only
def training_function(total_gpus: int, run_id: int = None, **kwargs):
    # Configure logger with existing run ID if available
    catalog = kwargs.get("catalog")
    schema = kwargs.get("schema")
    
    volume_path = f'/Volumes/{catalog}/{schema}'
    
    if run_id:
        mlf_logger = MLFlowLogger(
            tracking_uri="databricks",
            run_id=run_id,
            save_dir=f'{volume_path}/training',
            checkpoint_path_prefix=f'{volume_path}/training/checkpoints',
            artifact_location=f'{volume_path}/training/artifacts'
        )
    else:  # Worker nodes will have no run_id
        mlf_logger = None

    trainer = pl.Trainer(
        default_root_dir=f'{volume_path}/training',
        devices=total_gpus, 
        accelerator="gpu", 
        logger=mlf_logger,
        strategy='ddp',
        max_epochs=kwargs['max_epochs'],
    )
    
    trainer.fit(model)

    return trainer

    
if __name__ == '__main__':

    
    args_dict = json.loads(sys.argv[1])
    
    batch_size = args_dict['batch_size']
    epochs = args_dict['max_epochs']
    total_gpus = args_dict['total_gpus']
    host = args_dict['host']
    token = args_dict['token']
    mlflow_run_id = args_dict['run_id']
    
    ds_catalog = args_dict['uc_catalog'] #'brian_ml_dev'
    ds_schame = args_dict['uc_schema'] #'image_processing'
    coco_volume = 'coco_dataset'
    save_dir = '/local_disk0/train'

    mlflow_experiment = args_dict['mlflow_experiment'] #'/Users/brian.law@databricks.com/brian_lightning'

    volume_path = f"/Volumes/{ds_catalog}/{ds_schame}/{coco_volume}"
    image_path = f'{volume_path}'
    annotation_file = f'{volume_path}/annotations.json'
    
    Train_Dataset = CocoDetection(
        image_directory_path=image_path, 
        image_processor=image_processor, 
        annotation_file=annotation_file,
        train=True
    )
    
    categories = Train_Dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}
    
    TRAIN_DATALOADER = DataLoader(dataset=Train_Dataset, 
                                  collate_fn=collate_fn, 
                                  batch_size=batch_size, shuffle=True)
    
    VAL_DATALOADER = DataLoader(dataset=Train_Dataset, 
                                collate_fn=collate_fn, 
                                batch_size=batch_size)
    
    categories = Train_Dataset.coco.cats
    id2label = {k: v['name'] for k,v in categories.items()}

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label=id2label)
    
    os.environ['DATABRICKS_HOST'] = host 
    os.environ['DATABRICKS_TOKEN'] = token
    
    # turns Infiniband and direct access off until we can update core libraries to support
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'

            
    #mlflow.set_registry('databricks-uc')
    mlflow.set_experiment(mlflow_experiment)
    
    training_function(total_gpus, max_epochs=epochs, 
                      run_id=mlflow_run_id,
                      catalog=ds_catalog,
                      schema=ds_schame)

    #### Proper Model Logging ####
    # we need to split this out to structure the image pipeline and log with a signature to suit deployment models
    
    # rank = int(os.environ.get("RANK", 0))
    # if rank -- 0:
    #     components = {
    #         'model': model,
    #         'image_processor': image_processor
    #     }
        
    #     sample_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    #     sample_output = model(**image_processor(sample_image, return_tensors="pt"))

    #     signature = mlflow.models.infer_signature(
    #         sample_image,
    #         {"logits": sample_output.logits.detach().numpy()}
    #     )
        
    #     with mlflow.start_run(run_id = mlflow_run_id):
    #         mlflow.transformers.log_model(
    #             transformers_model=components,
    #             artifact_path="rt-detr-model",
    #             task="object-detection",
    #             signature=signature,
    #             input_example=sample_image,
    #         # For custom serving logic:
    #         # pip_requirements=["torch", "transformers", "numpy"]
    #         )
        