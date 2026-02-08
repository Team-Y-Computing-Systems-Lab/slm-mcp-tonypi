import asyncio
import json
import requests
import time
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import base64
from datetime import datetime
import cv2
import time
from controller import pick_object 

# Configuration
ROBOT_BASE_URL = "http://lab-erza.local:9030"
VISION_API_URL = "http://127.0.0.0:8000/dino_api"

mcp = Server("robot-control-mcp-server")

def navigate_and_pick_object(object_description: str): 
    """navigate to the object and pick the object up"""
    
    try: 
        action_list = pick_object(object_description)
        return{
            "status": "success",
            "actions": action_list
        }

    except Exception as e : 
        return {"status": "error", "error": str(e)}    

def propagate_action(action: str, times: int = 1):
    """Execute predefined robot actions"""
    url = ROBOT_BASE_URL
    data = {
        "jsonrpc": "2.0",
        "method": "RunAction",
        "params": [action, times],
        "id": 1
    }
    try: 
        print(f"Sending to {data} to {url}")
        response = requests.post(url, json=data, timeout=10)
        print(f"Robot response status: {response.status_code}")
        print(f"Robot response text: {response.text}")
        time.sleep(3)
        return {"status": "success", "response": response.text}
    except Exception as e: 
        print(f"Robot request failed: {e}")
        return {"status": "error", "error": str(e)}

def control_servo(servo_position: int):
    """Control servo position"""
    url = ROBOT_BASE_URL
    data = {
        "jsonrpc": "2.0",
        "method": "SetPWMServo",
        "params": [1000, 2, 1, int(servo_position)],
        "id": 1
    }
    try: 
        response = requests.post(url, json=data)
        time.sleep(2.5)
        return {"status": "success", "response": response.text}
    except Exception as e: 
        return {"status": "error", "error": str(e)}

def capture_image(request: str, boundary_colors: str = ""):
    """Capture image"""
    url = VISION_API_URL
    params = {
        "request": request,
        "boundaryColors": boundary_colors
    }
    try: 
        response = requests.post(url, params=params)
        time.sleep(2.5)
        return {"status": "success", "response": response.text}
    except Exception as e: 
        return {"status": "error", "error": str(e)}

def summarize_scene():
    """VLM integration for scene description"""
    try:
        print("[VLM] Capturing image from robot camera...")
        camera_url = "http://lab-erza:8080/"
        camera = cv2.VideoCapture(camera_url)
        for attempt in range(3):    # retry 3 times with timeout
            success, image = camera.read()
            if success:
                break
            time.sleep(0.5)
        
        camera.release()
        if not success or image is None:
            return {"status": "error", "error": "Failed to capture image from robot's camera"}
        
        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M") # save the image
        cv2.imwrite(f"robot_view_{timestamp}.jpg", image)
        #print(f"[VLM] Saved robot view to: robot_view_{timestamp}.jpg")
        
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
        data = {
            "model": "qwen3-vl:2b",
            "prompt": """As a robot looking through my camera, describe what I see in ONE concise sentence.
            For each object you identify, you MUST provide:
            1. The color of the object(e.g, red, blue, green, yellow, black, white)
            2. The name/type of the object (e.g, ball, box, cube, cup, container, block, toy, bottle)
            3. Its approximate position (left, center, right, foreground, background). 
            Format your response ONLY in this manner:
            "I see a [color] [object name] on the [position], a [color] [object name] on the [position], etc." and keep it factual.""",
            
            "images": [image_base64],
            "stream": False
            # "options": {
            #     "temperature": 0.1, "num_predict": 100
            # }
        }
        
        print(f"[VLM] Sending to {data['model']} via Ollama...")
        response = requests.post(OLLAMA_URL, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            description = result.get("response", "").strip()
            
            if description:           # only use the response if it isn't empty
                print(f"[VLM] Response: {description}")
                return {"status": "success", "summary": description}
            else:                     # description if the VLM returns empty
                print("[VLM] Empty response from VLM")
                return {"status": "success", "summary": "I'm looking at the scene but don't see any specific objects to describe"}
        
        else:
            error_msg = f"VLM failed: {response.status_code} - {response.text[:200]}"
            print(f"[VLM] {error_msg}")
            return {"status": "error", "error": error_msg}
    
    except json.JSONDecodeError as e:
        error_msg = f"Invalid response from Ollama: {str(e)}"
        print(f"[VLM] {error_msg}")
        return {"status": "error", "error": error_msg}
    
    except Exception as e:
        error_msg = f"VLM summarization error: {str(e)}"
        print(f"[VLM] {error_msg}")
        return {"status": "error", "error": error_msg}
        
    
@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="Propagate Action",
            description="Execute predefined robot actions",
            inputSchema={
                "type": "object",
                "properties": {
                    "Action": {
                        "type": "string",
                        "enum": [
                            "back", "back_end", "back_fast", "back_one_step", "bow",
                            "go_forward", "go_forward_end", "go_forward_fast", "go_forward_one_small_step", 
                            "go_forward_one_step", "go_forward_start", "go_forward_start_fast", 
                            "left_kick", "left_move_10", "left_move_20", "left_move_30", "left_move", 
                            "left_move_fast", "left_shot", "left_shot_fast", "left_uppercut",
                            "right_kick", "right_move_10", "right_move_20", "right_move_30", "right_move", 
                            "right_move_fast", "right_shot", "right_shot_fast", "right_uppercut",
                            "sit_ups", "squat", "squat_down", "squat_up", "stand", "stand_slow", "stand_up_back", 
                            "stand_up_front", "put_down", "wave", "wing_chun", "catch_ball", "catch_ball_up",
                            "catch_ball_go", "catch_ball_left_move", "catch_ball_right_move", "move_up"
                        ]
                    }
                },
                "required": ["Action"]
            }
        ),
        Tool(
            name="Control Servo",
            description="Control servo position (1000-2000, default 1500 for straight ahead)",
            inputSchema={
                "type": "object",
                "properties": {
                    "Servo Position": {"type": "integer", "minimum": 1000, "maximum": 2000}
                },
                "required": ["Servo Position"]
            }
        ),
        Tool(
            name="Capture Image",
            description="Capture image for visual analysis with object detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "Request": {"type": "string", "description": "Semicolon-separated objects to look for"},
                    "BoundaryColors": {"type": "string", "description": "Semicolon-separated RGB values for bounding boxes"}
                },
                "required": ["Request"]
            }
        ),
        Tool(
            name="Summarize Scene",
            description=(
                "Call the vision system and get a short natural-language description of what the robot's camera currently sees."
                "Use this before planning complex tasks that depend on the scene"
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        # RL add 
        Tool(
            name="Pick Object",
            description=(
                "call the robot action controller to navigate to the object and pick it up"
            ),
            inputSchema={
                "type": "object",
                "properties":{
                    "object_description": {"type": "string", "description": "words that describe object and its attribute like colors"}
                },
                "required": ["object_description"]
            }
        )
    ]

@mcp.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    try:
        if name == "Propagate Action":
            action = arguments.get("Action")
            result = propagate_action(action)
            if result["status"] == "success":
                return [TextContent(type="text", text=f"Action '{action}' executed successfully")]
            else:
                return [TextContent(type="text", text=f"Action failed: {result['error']}")]
        
        elif name == "Control Servo":
            servo_position = arguments.get("Servo Position")
            result = control_servo(servo_position)
            if result["status"] == "success":
                return [TextContent(type="text", text=f"Servo set to position {servo_position} successfully")]
            else:
                return [TextContent(type="text", text=f"Servo control failed: {result['error']}")]
        
        elif name == "Capture Image":
            request = arguments.get("Request")
            boundary_colors = arguments.get("BoundaryColors", "")
            result = capture_image(request, boundary_colors)
            if result["status"] == "success":
                return [TextContent(type="text", text=f"Image captured and processed for: {request}")]
            else:
                return [TextContent(type="text", text=f"Image capture failed: {result['error']}")]
        
        elif name == "Summarize Scene":
            result = summarize_scene()
            if result["status"] == "success":
                summary = result.get("summary", "")
                return [TextContent(
                    type="text",
                    text=f"Scene summary: {summary}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Scene summarization failed: {result['error']}"
                )]
                
        elif name == "Pick Object":
            object_description = arguments.get("object_description")
            result = navigate_and_pick_object(object_description)
            if result["status"] == "success":
                return [TextContent(type="text", text=f"navigated upto {object_description} using the sequence {result['actions']}")]
            else:
                return [TextContent(type="text", text=f"Image capture failed: {result['error']}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        print(f"DEBUG: Call_tool error: {e}")
        return [TextContent(type="text", text=f"Tool execution error: {str(e)}")]

async def main():
    print("Robot MCP Server Starting...")
    print(f"Robot URL: {ROBOT_BASE_URL}")
    print(f"Vision API URL: {VISION_API_URL}")
    
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())