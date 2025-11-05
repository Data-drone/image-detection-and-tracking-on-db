"""
YOLO Training Script for Distributed Training on Databricks
Usage: distributor.run('scripts/train_yolo.py', run_id, config_json)
"""
import os, sys, json, gc
import torch, torch.nn as nn
import torch.distributed as dist
from torchvision.ops import nms
from ultralytics import YOLO, settings
import mlflow, pandas as pd, numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec


def train_yolo(mlflow_run_id: str, config: dict, job_run_id: str, log_directory: str = "/tmp"):
    """YOLO training with MLflow tracking
    
    Args:
        mlflow_run_id: MLflow run ID for experiment tracking
        config: Training configuration dictionary
        job_run_id: Databricks job run ID (default: "no_id")
        log_directory: Directory to save metrics output file (default: "/tmp")
    """
    # Setup environment
    os.environ.update({
        'DATABRICKS_HOST': config['db_host'],
        'DATABRICKS_TOKEN': config['db_token'],
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1'
    })
    
    # Get distributed context
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    print(f"[Rank {rank}] Process started - local_rank: {local_rank}, world_size: {world_size}")
    
    # Initialize process group
    if world_size > 1 and not dist.is_initialized():
        print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        print(f"[Rank {rank}] Process group initialized")
    
    torch.cuda.set_device(local_rank)
    
    # MLflow setup (rank 0 only)
    if rank == 0:
        settings.update({"mlflow": True})
        os.environ.update({
            'MLFLOW_EXPERIMENT_NAME': config['mlflow_experiment'],
            'MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING': "true",
            'MLFLOW_RUN_ID': mlflow_run_id
        })
        print(f"[Rank 0] MLflow configured with run_id: {mlflow_run_id}")
        
        # Log dataset
        try:
            print(f"[Rank 0] Logging dataset...")
            dataset_info = pd.DataFrame([{
                'dataset_path': config['dataset_path'],
                'config_file': config['data_config'],
                'num_classes': config.get('num_classes', 'N/A'),
            }])
            dataset = mlflow.data.from_pandas(dataset_info, source=config['dataset_path'], 
                                            name=config.get('dataset_name', 'yolo_dataset'))
            mlflow.log_input(dataset, context="training")
            print(f"[Rank 0] ✓ Dataset logged")
        except Exception as e:
            print(f"[Rank 0] Warning: Dataset logging failed - {e}")
    else:
        settings.update({"mlflow": False})
        print(f"[Rank {rank}] MLflow disabled")
    
    # Train
    print(f"[Rank {rank}] Loading model: {config['model']}")
    model = YOLO(config['model'])
    
    print(f"[Rank {rank}] Starting training...")
    results = model.train(
        data=config['data_config'],
        epochs=config['epochs'],
        batch=config['batch_size'],
        lr0=config.get('initial_lr', 0.01),
        lrf=config.get('final_lr', 0.01),
        imgsz=config['img_size'],
        name=config.get('run_name', 'train'),
        project=config['project_path'],
        device=local_rank,
        workers=4,
        patience=config.get('patience', 50),
        save=True,
        save_period=config.get('save_period', 5),
        verbose=(rank == 0),
        exist_ok=True
    )
    
    # Log wrapped model (rank 0 only)
    if rank == 0:
        try:
            print(f"[Rank 0] Logging wrapped model...")
            
            # Model wrapper
            class YoloDetWrapper(nn.Module):
                def __init__(self, base: nn.Module, conf_thres=0.25, iou_thres=0.5, max_det=300):
                    super().__init__()
                    self.base, self.conf_thres, self.iou_thres, self.max_det = base.eval(), conf_thres, iou_thres, max_det
                
                @staticmethod
                def xywh_to_xyxy(b):
                    x, y, w, h = b.unbind(-1)
                    return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=-1)
                
                def forward(self, x):
                    with torch.no_grad():
                        out = self.base(x)
                        if isinstance(out, (list, tuple)): out = out[0]
                        if out.dim() == 3 and out.shape[1] < out.shape[2]: out = out.permute(0, 2, 1).contiguous()
                        boxes_xywh, obj, cls = out[..., :4], out[..., 4:5].sigmoid(), out[..., 5:].sigmoid()
                        conf, cls_id = (obj * cls).max(-1)
                        boxes = self.xywh_to_xyxy(boxes_xywh)
                        N, A = conf.shape
                        out_pad = x.new_full((N, self.max_det, 6), -1.0)
                        for i in range(N):
                            mask = conf[i] >= self.conf_thres
                            if mask.sum() == 0: continue
                            b, s, c = boxes[i][mask], conf[i][mask], cls_id[i][mask].float()
                            keep = nms(b, s, self.iou_thres)[:self.max_det]
                            if (k := keep.numel()) > 0:
                                out_pad[i, :k, :4], out_pad[i, :k, 4], out_pad[i, :k, 5] = b[keep], s[keep], c[keep]
                        return out_pad
            
            # Load and wrap best model
            best_path = f"{config['project_path']}/{config.get('run_name', 'train')}/weights/best.pt"
            wrapped_model = YoloDetWrapper(YOLO(best_path).model)
            
            # Log model
            signature = ModelSignature(
                inputs=Schema([TensorSpec(np.dtype(np.float32), (-1, 3, config['img_size'], config['img_size']), "images")]),
                outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, None, 6), "detections")])
            )
            mlflow.pytorch.log_model(wrapped_model, artifact_path="best_model", signature=signature)
            print(f"[Rank 0] ✓ Model logged")
        except Exception as e:
            print(f"[Rank 0] Warning: Model logging failed - {e}")
    
    # Cleanup
    print(f"[Rank {rank}] Training finished, cleaning up...")
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    gc.collect()
    
    # Return metrics dict instead of full results object
    if rank == 0:
        try:
            # Extract validation mAP scores
            metrics_dict = {
                'val_mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
                'val_mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                'final_epoch': config['epochs'],
                'job_run_id': job_run_id,
                'mlflow_run_id': mlflow_run_id,
                'status': 'completed'
            }
            print(f"[Rank 0] Final metrics: mAP50={metrics_dict['val_mAP50']:.4f}, mAP50-95={metrics_dict['val_mAP50-95']:.4f}")
            
            # Write metrics to file with job_run_id in filename
            try:
                os.makedirs(log_directory, exist_ok=True)
                metrics_file = f"{log_directory}/metrics_{job_run_id}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
                print(f"[Rank 0] Metrics written to: {metrics_file}")
            except Exception as e:
                print(f"[Rank 0] Warning: Could not write metrics file - {e}")
            
            return metrics_dict
        except Exception as e:
            print(f"[Rank 0] Warning: Could not extract metrics - {e}")
            return {'status': 'completed', 'val_mAP50': 0.0, 'job_run_id': job_run_id}
    else:
        return None


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: train_yolo.py <mlflow_run_id> <config_json> [job_run_id] [log_directory]")
        sys.exit(1)
    
    mlflow_run_id = sys.argv[1]
    config_json = sys.argv[2]
    job_run_id = sys.argv[3] if len(sys.argv) > 3 else "no_id"
    log_directory = sys.argv[4] if len(sys.argv) > 4 else "/tmp"

    print("Running: train_yolo.py with <mlflow_run_id> <config_json> [job_run_id] [log_directory]")
    
    try:
        config = json.loads(config_json)
        results = train_yolo(mlflow_run_id, config, job_run_id, log_directory)
        print(f"Training completed successfully! Job run ID: {job_run_id}")
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        sys.exit(1)

