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

def chat(request: str):
    OLLAMA_BASE_URL= "http://127.0.0.1:11434"
    MODEL = "erza:latest"
    url = OLLAMA_BASE_URL + '/api/chat'
    headers = {'Content-Type': 'application/json'}
    
    try:
        # Capture image from camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        print("Captured image from camera", frame.shape) 
        
        
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("test")
        
        if not ret:
            return {"error": "Failed to capture image from camera"}
        
        # Convert OpenCV frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Convert to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Prepare payload for model with the user's request
        data = {
            "model": MODEL,
            "stream": True,
            "messages": [
                {"role": "user", "content": request, "images": [encoded_image]}
            ]
        }
        
        # Send request to Ollama
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
        
        return {"description": string_test}
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": f"Error processing image: {str(e)}"}