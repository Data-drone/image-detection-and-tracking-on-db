"""
YOLO Training Script for Distributed Training on Databricks
Usage: distributor.run('scripts/train_yolo.py', run_id, config_json)
"""
import os, sys, json, gc, base64
from io import BytesIO
from PIL import Image
import torch, torch.nn as nn
import torch.distributed as dist
from torchvision.ops import nms
from ultralytics import YOLO, settings
import mlflow, pandas as pd, numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec, ColSpec
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource


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
        'DATABRICKS_WORKSPACE_ID': config['db_workspace_id'],
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1'
    })

    # Add these for workflow tracking (if available in config):
    if config.get('databricks_job_id'):
        os.environ['DATABRICKS_JOB_ID'] = str(config['databricks_job_id'])
    if config.get('databricks_run_id'):
        os.environ['DATABRICKS_RUN_ID'] = str(config['databricks_run_id'])
    
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
    
    os.environ.update({
        'MLFLOW_EXPERIMENT_NAME': config['mlflow_experiment'],
        'MLFLOW_RUN_ID': mlflow_run_id,
    })

    # MLflow setup (rank 0 only)
    if rank == 0:
        settings.update({"mlflow": True})
        os.environ.update({
            'MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING': "true",
            'MLFLOW_KEEP_RUN_ACTIVE': "true"
        })
        print(f"[Rank 0] MLflow configured with run_id: {mlflow_run_id}")
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
    
    # Log dataset and wrapped model (rank 0 only)
    if rank == 0:
        # Log dataset - create dataset entities for MLflow 3
        train_dataset = None
        fs_dataset = None
        try:
            print(f"[Rank 0] Logging dataset...")
            
            # Log dataset info as pandas DataFrame (for metadata)
            dataset_info = pd.DataFrame([{
                'dataset_path': config['dataset_path'],
                'config_file': config['data_config'],
                'num_classes': config.get('num_classes', 'N/A'),
            }])
            train_dataset = mlflow.data.from_pandas(
                dataset_info, 
                source=config['dataset_path'], 
                name=config.get('dataset_name', 'yolo_dataset')
            )
            mlflow.log_input(train_dataset, context="training")
            
            # Log dataset path as filesystem source
            fs_source = FileSystemDatasetSource(path=config['dataset_path'])
            fs_dataset = mlflow.data.from_sources(
                source=fs_source,
                name=config.get('dataset_name', 'yolo_dataset') + '_filesystem'
            )
            mlflow.log_input(fs_dataset, context="dataset_path")
            
            print(f"[Rank 0] ✓ Dataset logged")
        except Exception as e:
            print(f"[Rank 0] Warning: Dataset logging failed - {e}")
        
        # Log wrapped model
        print(f"[Rank 0] Logging wrapped model...")
        
        # Load best model path
        best_path = f"{config['project_path']}/{config.get('run_name', 'train')}/weights/best.pt"
        
        # Create signature with base64 string input
        signature = ModelSignature(
            inputs=Schema([ColSpec("string", "images")]),
            outputs=Schema([TensorSpec(np.dtype(np.float32), (-1, -1, 6), "detections")])
        )
        
        # Create input example as base64 encoded image
        dummy_img = np.random.randint(0, 255, (config['img_size'], config['img_size'], 3), dtype=np.uint8)
        pil_img = Image.fromarray(dummy_img)
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        input_example = {"images": img_base64}
        
        # Log model as LoggedModel entity (MLflow 3) with parameters
        model_params = {
            'initial_lr': config.get('initial_lr', 0.01),
            'final_lr': config.get('final_lr', 0.01),
            'batch_size': config['batch_size'],
            'img_size': config['img_size'],
            'epochs': config['epochs'],
            'patience': config.get('patience', 50),
            'conf_thres': 0.25,
            'iou_thres': 0.5,
            'max_det': 300
        }
        
        # Create artifacts dict to include the model weights file
        # This ensures the .pt file is saved alongside the MLflow model
        import tempfile
        import shutil
        artifacts_dir = tempfile.mkdtemp()
        
        try:
            # Copy the best.pt file to artifacts directory
            artifact_model_path = os.path.join(artifacts_dir, "best.pt")
            shutil.copy2(best_path, artifact_model_path)
            
            artifacts = {
                "model": artifact_model_path
            }
            
            # Update the pyfunc model to use the artifact path
            # The model will be loaded from artifacts during serving
            class YOLOPyFuncWrapperWithArtifacts(mlflow.pyfunc.PythonModel):
                """
                MLflow PyFunc wrapper for YOLO model with base64 string input support.
                Handles decoding, preprocessing, inference, and post-processing.
                Uses artifacts to load the model file.
                """
                
                def __init__(self, img_size=640, conf_thres=0.25, iou_thres=0.5, max_det=300):
                    """Initialize with detection parameters"""
                    self.img_size = img_size
                    self.conf_thres = conf_thres
                    self.iou_thres = iou_thres
                    self.max_det = max_det
                    self.model = None
                
                def load_context(self, context):
                    """Load the YOLO model during MLflow model loading"""
                    from ultralytics import YOLO
                    import torch
                    
                    # Load the YOLO model from artifacts
                    model_path = context.artifacts["model"]
                    yolo = YOLO(model_path)
                    self.model = yolo.model.eval()
                    
                    print(f"YOLO model loaded from artifacts: {model_path}")
                
                @staticmethod
                def xywh_to_xyxy(b):
                    """Convert box format from xywh to xyxy"""
                    x, y, w, h = b.unbind(-1)
                    return torch.stack([x-w/2, y-h/2, x+w/2, y+h/2], dim=-1)
                
                def preprocess_base64(self, base64_str):
                    """Decode base64 image and convert to tensor format"""
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    import numpy as np
                    import torch
                    
                    # Handle different string types (str, np.str_, bytes)
                    if isinstance(base64_str, bytes):
                        base64_str = base64_str.decode('utf-8')
                    else:
                        base64_str = str(base64_str)
                    
                    # Decode base64 to bytes
                    img_bytes = base64.b64decode(base64_str)
                    
                    # Load as PIL Image
                    img = Image.open(BytesIO(img_bytes)).convert('RGB')
                    
                    # Resize to expected size
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    # Convert HWC to CHW format
                    img_array = np.transpose(img_array, (2, 0, 1))
                    
                    # Convert to torch tensor with batch dimension
                    img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                    
                    return img_tensor
                
                def predict(self, context, model_input):
                    """
                    Handle prediction requests from Model Serving endpoint.
                    
                    Args:
                        context: MLflow context (unused in predict, used in load_context)
                        model_input: Dict, DataFrame, or direct input with 'images' containing base64 strings
                        
                    Returns:
                        numpy array of detections with shape (batch_size, max_det, 6)
                        Each detection: [x1, y1, x2, y2, confidence, class_id]
                    """
                    import torch
                    import numpy as np
                    from torchvision.ops import nms
                    
                    # Extract images from input
                    if isinstance(model_input, dict):
                        images = model_input.get('images')
                    elif hasattr(model_input, 'to_dict'):  # DataFrame
                        images = model_input['images'].tolist() if 'images' in model_input.columns else None
                        if images and len(images) == 1:
                            images = images[0]
                    elif hasattr(model_input, '__getitem__'):  # List or array-like
                        images = model_input
                    else:
                        images = str(model_input)
                    
                    if images is None:
                        raise ValueError("Input must contain 'images' key with base64 encoded image string")
                    
                    # Handle different input formats for base64 strings
                    with torch.no_grad():
                        # Single string (most common case for serving endpoint)
                        if isinstance(images, (str, bytes, np.str_)):
                            x = self.preprocess_base64(images)
                        
                        # 0-dimensional numpy array containing a string
                        elif isinstance(images, np.ndarray) and images.ndim == 0:
                            x = self.preprocess_base64(images.item())
                        
                        # List/tuple of strings (batch)
                        elif isinstance(images, (list, tuple)):
                            processed = [self.preprocess_base64(img) for img in images]
                            x = torch.cat(processed, dim=0)
                        
                        # Numpy array of strings
                        elif isinstance(images, np.ndarray) and images.dtype.kind in ('U', 'S', 'O'):
                            processed = [self.preprocess_base64(str(img)) for img in images.flat]
                            x = torch.cat(processed, dim=0)
                        
                        else:
                            raise ValueError(
                                f"Unsupported input type: {type(images)}. "
                                f"Expected base64 encoded string or list of strings. "
                                f"Input dtype: {getattr(images, 'dtype', 'N/A')}"
                            )
                        
                        # Move to model device
                        device = next(self.model.parameters()).device
                        x = x.to(device)
                        
                        # Run inference
                        out = self.model(x)
                        
                        # Handle different output formats
                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        
                        # Ensure correct dimension ordering
                        if out.dim() == 3 and out.shape[1] < out.shape[2]:
                            out = out.permute(0, 2, 1).contiguous()
                        
                        # Parse detections: [batch, anchors, 4+1+num_classes]
                        boxes_xywh = out[..., :4]
                        obj = out[..., 4:5].sigmoid()
                        cls = out[..., 5:].sigmoid()
                        
                        # Compute confidence scores and class IDs
                        conf, cls_id = (obj * cls).max(-1)
                        
                        # Convert boxes to xyxy format
                        boxes = self.xywh_to_xyxy(boxes_xywh)
                        
                        # Apply NMS and filtering per image
                        N = boxes.shape[0]
                        out_pad = x.new_full((N, self.max_det, 6), -1.0)
                        
                        for i in range(N):
                            # Filter by confidence threshold
                            mask = conf[i] >= self.conf_thres
                            if mask.sum() == 0:
                                continue
                            
                            b = boxes[i][mask]
                            s = conf[i][mask]
                            c = cls_id[i][mask].float()
                            
                            # Apply NMS
                            keep = nms(b, s, self.iou_thres)[:self.max_det]
                            
                            # Store results
                            k = keep.numel()
                            if k > 0:
                                out_pad[i, :k, :4] = b[keep]
                                out_pad[i, :k, 4] = s[keep]
                                out_pad[i, :k, 5] = c[keep]
                        
                        # Convert to numpy
                        result = out_pad.cpu().numpy()
                        
                        return result
            
            # Create new pyfunc model with artifacts support
            pyfunc_model_with_artifacts = YOLOPyFuncWrapperWithArtifacts(
                img_size=config['img_size'],
                conf_thres=0.25,
                iou_thres=0.5,
                max_det=300
            )
            
            # Log model with artifacts
            model_info = mlflow.pyfunc.log_model(
                python_model=pyfunc_model_with_artifacts,
                artifact_path="best_model",
                artifacts=artifacts,
                signature=signature,
                input_example=input_example,
                pip_requirements=[
                    "torch",
                    "torchvision", 
                    "ultralytics",
                    "pillow",
                    "numpy"
                ]
            )
        finally:
            # Clean up temporary directory
            shutil.rmtree(artifacts_dir, ignore_errors=True)
        model_id = model_info.model_id
        print(f"[Rank 0] ✓ Model logged with model_id: {model_id}")
        
        # Log final validation metrics linked to LoggedModel and dataset (MLflow 3)
        try:
            print(f"[Rank 0] Logging final metrics linked to model and dataset...")
            
            # Extract validation mAP scores
            val_mAP50 = float(results.results_dict.get('metrics/mAP50(B)', 0.0))
            val_mAP50_95 = float(results.results_dict.get('metrics/mAP50-95(B)', 0.0))
            
            # Log metrics linked to both LoggedModel (model_id) and dataset
            mlflow.log_metric(
                key="final_val_mAP50",
                value=val_mAP50,
                step=config['epochs'],
                model_id=model_id,  # Links to LoggedModel
                dataset=train_dataset  # Links to dataset
            )
            
            mlflow.log_metric(
                key="final_val_mAP50-95",
                value=val_mAP50_95,
                step=config['epochs'],
                model_id=model_id,  # Links to LoggedModel
                dataset=train_dataset  # Links to dataset
            )
            
            print(f"[Rank 0] ✓ Final metrics linked to model_id: {model_id}")
            print(f"[Rank 0] Final metrics: mAP50={val_mAP50:.4f}, mAP50-95={val_mAP50_95:.4f}")
            
            # Prepare metrics dict for return
            metrics_dict = {
                'val_mAP50': val_mAP50,
                'val_mAP50-95': val_mAP50_95,
                'final_epoch': config['epochs'],
                'job_run_id': job_run_id,
                'mlflow_run_id': mlflow_run_id,
                'model_id': model_id,
                'status': 'completed'
            }
            
            # Write metrics to file with job_run_id in filename
            try:
                os.makedirs(log_directory, exist_ok=True)
                metrics_file = f"{log_directory}/metrics_{job_run_id}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_dict, f, indent=2)
                print(f"[Rank 0] Metrics written to: {metrics_file}")
            except Exception as e:
                print(f"[Rank 0] Warning: Could not write metrics file - {e}")
            
        except Exception as e:
            print(f"[Rank 0] Warning: Could not log final metrics - {e}")
            metrics_dict = {
                'val_mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0.0)),
                'val_mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0.0)),
                'final_epoch': config['epochs'],
                'job_run_id': job_run_id,
                'mlflow_run_id': mlflow_run_id,
                'model_id': model_id if 'model_id' in locals() else None,
                'status': 'completed'
            }
    
    # Cleanup
    mlflow.end_run() 
    print(f"[Rank {rank}] Training finished, cleaning up...")
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    gc.collect()
    
    # Return metrics dict instead of full results object
    if rank == 0:
        return metrics_dict if 'metrics_dict' in locals() else {
            'status': 'completed', 
            'val_mAP50': 0.0, 
            'job_run_id': job_run_id,
            'mlflow_run_id': mlflow_run_id
        }
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

