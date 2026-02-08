from dataclasses import dataclass 
from typing import Any, List, Dict, Optional, Union, Tuple 

import numpy as np 

@dataclass 
class BoundingBox:
    xmin: int 
    ymin: int 
    xmax: int 
    ymax: int 

    image_width: int 
    image_height: int 

    def __init__(self, xmin, ymin, xmax, ymax, image_width = None, image_height = None):
        self.xmin = xmin 
        self.ymin = ymin 
        self.xmax = xmax 
        self.ymax = ymax 
                
        self.image_width = image_width 
        self.image_height = image_height 

    
    # @property
    def set_image_size(self, image_width, image_height):
        self.image_height = image_height 
        self.image_width = image_width

        print(f"---------{type(self.image_height)}")

    @property 
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def cxcy(self) -> List[float]:
        center = lambda x,y : x + (y-x)/2.0
        return [center(self.xmin, self.xmax), center(self.ymin, self.ymax)]
    
    @property
    def n_cxcy(self) -> List[float]:
        assert type(self.image_height) != type(None), "image width and height not specified."
        cx, cy =  self.cxcy
        return [cx/self.image_width, cy/self.image_height]

    @property
    def ncx(self) -> float:
        ncx, ncy = self.n_cxcy 
        return ncx

    @property
    def ncy(self) -> float:
        ncx, ncy = self.n_cxcy 
        return ncy

    @property
    def w(self) -> float:
        return max(1, self.xmax - self.xmin)

    @property
    def h(self) -> float:
        return max(1, self.ymax - self.ymin)

    @property 
    def area(self) -> float:
        return (self.w * self.h )

    @property
    def n_area(self) -> float:
        assert type(self.image_height) != type(None), "image width and height not specified."
        return self.area / (self.image_width * self.image_height) 

    @property
    def n_bottom(self) -> float:
        assert type(self.image_height) != type(None), "image width and height not specified."
        return self.ymax / self.image_height 

@dataclass 
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None
    
    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))
        
    @classmethod
    def from_json(cls, d: dict, img_w=None, img_h=None):
        box = BoundingBox(
            xmin=d["box"]["xmin"],
            ymin=d["box"]["ymin"],
            xmax=d["box"]["xmax"],
            ymax=d["box"]["ymax"],
            image_width=img_w,
            image_height=img_h
        )
        return cls(
            score=d["score"],
            label=d["label"],
            box=box
        )