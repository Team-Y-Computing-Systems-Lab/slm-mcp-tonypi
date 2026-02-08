from fastapi import FastAPI, Request, File, UploadFile
import requests
from pydantic import BaseModel 
import json 
import re 
from typing import List
import cv2
import time
import base64
from io import BytesIO
from PIL import Image
import os
import logging
from FastAPI_Modules import vision
from fastapi.responses import JSONResponse
from PIL import Image 
import numpy as np 



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryModel(BaseModel):
    response: str
    action: List[str] # [str] 


app = FastAPI() 
@app.post("/chat_api")
async def chat(request: str):
    OLLAMA_BASE_URL= "http://127.0.0.1:11434"
    MODEL =  "erza:latest" # "deepseek-r1:70b"
    url = OLLAMA_BASE_URL + '/api/chat'
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": MODEL,
        # "prompt": request, #.query,
        "messages": [{"role": "user", "content": request}],
        "stream": True ,
        "format": QueryModel.model_json_schema(),
        
    }
    string_test = ''
    print('test', request)
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    json_line = json.loads(decoded_line)
                    print(json_line['message']['content'], end="", flush=True)
                    string_test += json_line['message']['content']
                except json.JSONDecodeError:
                    print(f"JSONDecodeError: {decoded_line}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
    
    
    string_test = string_test.split("</think>\n\n")[-1]
    # string_test = string_test.replace("'", '"')
    # Parse it  
    print() 
    print('[--------------]')
    string_test = re.search(r"\{[\s\S]*?\}", string_test)
    print('[--------------]', string_test.group(0))
    print("[+]", type(string_test.group(0)), string_test.group(0))   
    json_data = json.loads(string_test.group(0))
    print(json_data['action'])
    
    return json_data

@app.post("/chat_explanation_api")
async def chat(request: str):
    OLLAMA_BASE_URL= "http://127.0.0.1:11434"
    MODEL =  "deepseek-r1:70b" # "erza1:latest" # 
    url = OLLAMA_BASE_URL + '/api/chat'
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": MODEL,
        # "prompt": request, #.query,
        "messages": [{"role": "user", "content": request}],
        "stream": True ,
        # "format": QueryModel.model_json_schema(),
        
    }
    string_test = ''
    print('test', request)
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    json_line = json.loads(decoded_line)
                    print(json_line['message']['content'], end="", flush=True)
                    string_test += json_line['message']['content']
                except json.JSONDecodeError:
                    print(f"JSONDecodeError: {decoded_line}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

@app.post("/Vision_api")
async def chat(request: str): 
    vision.chat(request)
    
    
def detection_to_dict(det):
    return {
        "score": det.score,
        "label": det.label,
        "box": {
            "xmin": det.box.xmin,
            "ymin": det.box.ymin,
            "xmax": det.box.xmax,
            "ymax": det.box.ymax
        },
        # convert mask (numpy array) to list OR remove it if too large
        # "mask": det.mask.tolist() if det.mask is not None else None
    }


@app.post("/dino_api")
async def test(request: str, boundaryColors: str):
    from vis_tools.detection_vis import plot_detections
    from detect_seg import grounded_segmentation, detect 
    print(request, boundaryColors)
    labels = request.split(';')
    colors = boundaryColors.split(';')
    label_color_map = dict(zip(labels, colors))
    print("[*] label_color_map", label_color_map)
    threshold = 0.72
    
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    web_image = True
    # try:
    if web_image:
        remote_address =  "http://lab-erza:8080/"# 0 #
        camera = cv2.VideoCapture(remote_address)
    else : # except Exception as e :
        print("[-] failed to connect to remote camera, reverting to system camera")
        camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    # except Exception:
    # image = cv2.imread("./cute_cats1.png")
    image_height, image_width, image_channel = image.shape 
        
    # cv2.imwrite("failedimage.png", image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("image shape: ", image.shape)
    
    try : 
        image = Image.fromarray(image) # .convert("RGB")

        print('------------------label')
        # image_array, detections = grounded_segmentation(
        #     image=image,
        #     labels=labels,
        #     threshold=threshold,
        #     polygon_refinement=True,
        #     detector_id=detector_id,
        #     segmenter_id=segmenter_id
        # )
        
        detections = detect(
            image = image, 
            labels=labels
        )
        
        image_array = np.asarray(image)
        
        print('detetctions ', detections)
        img = plot_detections(image_array, detections, "cute_cats1.png", label_colors=None)

        cv2.imwrite('test.png', 
            cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )
        )
        
        print(detections[0].box)
        # Convert NumPy image (from OpenCV) to PIL
        # img_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        
        print(type(image_array))
        
        # import base64
        # # Encode to PNG -> base64
        # buf = BytesIO()
        # img_pil.save(buf, format="PNG")
        # img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        print(detections)
        
        
        # for detection in detections:
            
        return JSONResponse(content={
            "detections": [detection_to_dict(d) for d in detections],
            "image_width": image_width,
            "image_height": image_height,
        })

        # return JSONResponse(content={
        #     "box": detections[0].box,
        #     # "image": img/z/_b64
        # })
        #return detections[0].box
    except Exception as e :
        print('[-] failure to execute the detection' , e)
        try:
            img_pil =  Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except:
            img_pil = image #Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        import base64
        # Encode to PNG -> base64
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"error": "detection failed", "image":img_b64} 
    
    
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from PIL import Image
# import numpy as np
# import cv2
# import io

# from vis_tools.detection_vis import plot_detections
# from detect_seg import grounded_segmentation

# app = FastAPI()

# @app.post("/dino_api_2")
# async def dino_api_2(
#     image: UploadFile = File(...),
#     labels: str = Form(...)
# ):
#     try:
#         # Read uploaded image bytes
#         contents = await image.read()

#         # Convert bytes to NumPy array
#         image_np = np.frombuffer(contents, np.uint8)

#         # Decode image (e.g., JPEG) to OpenCV format (BGR)
#         image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

#         if image_cv is None:
#             return JSONResponse(status_code=400, content={"error": "Failed to decode image"})

#         # Convert OpenCV BGR -> RGB -> PIL
#         image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
#         image_pil = Image.fromarray(image_rgb)

#         print(f"[INFO] Labels received: {labels}")
#         print(f"[INFO] Image shape: {image_cv.shape}")

#         # Process label string
#         label, label_color = labels.split(";")

#         # Parameters
#         threshold = 0.72
#         detector_id = "IDEA-Research/grounding-dino-tiny"
#         segmenter_id = "facebook/sam-vit-base"

#         # Run detection
#         image_array, detections = grounded_segmentation(
#             image=image_pil,
#             labels=[label],
#             threshold=threshold,
#             polygon_refinement=True,
#             detector_id=detector_id,
#             segmenter_id=segmenter_id
#         )

#         # Optional: Save image with detections
#         _ = plot_detections(image_array, detections, "result.png", label_colors=None)

#         if detections:
#             print(len(detections), detections[0].box)
            
#             return {
#                 "num_detections": len(detections),
#                 "first_box": detections[0].box
#             }
#         else:
#             return {"message": "No detections found"}

#     except Exception as e:
#         print("[-] Detection failed:", e)
#         return JSONResponse(status_code=500, content={"error": "Detection failed", "detail": str(e)})
