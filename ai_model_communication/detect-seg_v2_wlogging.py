import logging
import sys
from typing import Any, List, Dict, Optional, Union, Tuple 
import traceback
import psutil
import gc

import numpy as np 
import torch 
from PIL import Image 
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline 

from vis_tools.detection_vis import get_boxes, load_image, refine_masks
from custom_data.detection_data import BoundingBox, DetectionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check if system has enough resources to run the models."""
    mem = psutil.virtual_memory()
    if mem.available < 2 * 1024 * 1024 * 1024:  # 2GB minimum
        logger.warning(f"Low memory available: {mem.available/(1024*1024):.2f}MB")
        return False
    return True

def log_hardware_info():
    """Log hardware information for debugging."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    
    mem = psutil.virtual_memory()
    logger.info(f"Memory - Total: {mem.total/(1024*1024):.2f}MB, Available: {mem.available/(1024*1024):.2f}MB")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"CPU usage: {psutil.cpu_percent()}%")

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    try:
        if not check_system_resources():
            raise RuntimeError("Insufficient system resources to run detection")
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
        logger.info(f"Loading detector model: {detector_id}")
        
        # Clear memory before loading model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
        
        # Format labels
        labels = [label if label.endswith(".") else label+"." for label in labels]
        logger.info(f"Detecting labels: {labels} with threshold {threshold}")
        
        results = object_detector(image, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]
        
        logger.info(f"Detection completed. Found {len(results)} objects")
        return results
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    try:
        if not detection_results:
            logger.warning("No detection results provided for segmentation")
            return detection_results
            
        if not check_system_resources():
            raise RuntimeError("Insufficient system resources to run segmentation")
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"
        logger.info(f"Loading segmenter model: {segmenter_id}")
        
        # Clear memory before loading model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        processor = AutoProcessor.from_pretrained(segmenter_id)
        
        boxes = get_boxes(detection_results)
        logger.info(f"Processing {len(boxes)} bounding boxes")
        
        inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
        
        outputs = segmentator(**inputs)
        masks = processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]
        
        masks = refine_masks(masks, polygon_refinement)
        
        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask
        
        logger.info("Segmentation completed successfully")
        return detection_results
        
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    try:
        log_hardware_info()
        
        if isinstance(image, str):
            logger.info(f"Loading image from path: {image}")
            image = load_image(image)
        
        logger.info("Starting detection...")
        detections = detect(image, labels, threshold, detector_id)
        
        # Commented out as it crashes your Pi - uncomment when you've resolved resource issues
        # logger.info("Starting segmentation...")
        # detections = segment(image, detections, polygon_refinement, segmenter_id)
        
        return np.array(image), detections
        
    except Exception as e:
        logger.error(f"Grounded segmentation failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        import cv2
        from vis_tools.detection_vis import plot_detections 
        
        labels = ["red ball", "blue square"]
        threshold = 0.3

        detector_id = "IDEA-Research/grounding-dino-tiny"
        segmenter_id = "facebook/sam-vit-base"

        logger.info("Starting grounded segmentation demo")
        
        # camera = cv2.VideoCapture(0)
        # return_value, image = camera.read()
        image_path = "./test.png"
        logger.info(f"Loading image from: {image_path}")
        image = cv2.imread(image_path)
        
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        logger.info("Running grounded segmentation...")
        image_array, detections = grounded_segmentation(
            image=image,
            labels=labels,
            threshold=threshold,
            polygon_refinement=True,
            detector_id=detector_id,
            segmenter_id=segmenter_id
        )

        output_path = "cute_cats1.png"
        logger.info(f"Saving results to: {output_path}")
        img = plot_detections(image_array, detections, output_path)
        
        if detections:
            logger.info(f"First detection result: {detections[0]}")
        else:
            logger.warning("No detections found")
            
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)