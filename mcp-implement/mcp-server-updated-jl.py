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

# Robot specific configuration
# Give a list of all actions available for use by the robot and provide descriptions of their purpose. 
# Constrain these to what the model is allowed to use autonomously.
#    - Some action meanings:
#        - move_up: move the hands/arms up to pick up or reach toward an object.
available_actions = {
    "back": "Move backwards.", 
    # "back_end": "", 
    # "back_fast": "", 
    # "back_one_step": "", 
    "bow": "Take a bow.", 
    "go_forward": "Move forwards.", 
    # "go_forward_end": "", 
    # "go_forward_fast": "", 
    # "go_forward_one_small_step": "", 
    # "go_forward_one_step": "", 
    # "go_forward_start": "", 
    # "go_forward_start_fast": "", 
    "left_kick": "Use the left leg to kick.", 
    # "left_move_10": "", 
    # "left_move_20": "", 
    # "left_move_30": "", 
    "left_move": "Move to the left.", 
    # "left_move_fast": "", 
    # "left_shot": "", 
    # "left_shot_fast": "", 
    # "left_uppercut": "", 
    "right_kick": "Use the right leg to kick.", 
    # "right_move_10": "", 
    # "right_move_20": "", 
    # "right_move_30": "", 
    "right_move": "Move to the right.", 
    # "right_move_fast": "", 
    # "right_shot": "", 
    # "right_shot_fast": "", 
    # "right_uppercut": "", 
    # "sit_ups": "", 
    # "squat": "", 
    # "squat_down": "", 
    # "squat_up": "", 
    # "stand": "", 
    # "stand_slow": "", 
    # "stand_up_back": "", 
    # "stand_up_front": "", 
    # "put_down": "", 
    "wave": "Perform a greeting or farewell.", 
    "wing_chun": "Perform a martial arts performance or dance.", 
    "catch_ball": "Pick up and lift an object. This is not limited to use with balls.", 
    "catch_ball_up": "Raises up the robots arms while holding an object.", 
    "catch_ball_go": "Walks forward while holding an object.", 
    "catch_ball_left_move": "Move left while holding an object.", 
    "catch_ball_right_move": "Move right while holding an object."}

# Gets a list of available actions based on the keys in available_actions.
list_of_all_actions = list(available_actions.keys())

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
        
        OLLAMA_URL = "http://100.67.254.11:11434/api/generate"
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
            name="Propagate_Action",
            description=f"""- Executes one of the predefined actions from the action group.
- This tool accepts one parameter: "Action" (string).
- The Action must be exactly one of the allowed actions listed below.
Available actions: {available_actions}. 

CRITICAL VALIDATION RULES:
- When using "Propagate_Action", you MUST use EXACTLY one of the allowed actions listed below.
- If you try to use an action not in the allowed list, the system will FAIL with a validation error.
- If you're unsure which action to use, choose the most semantically similar from the allowed list.

ERROR HANDLING:
If you receive an error saying an action is "not one of" the allowed list, you MUST:
1. Check the allowed action list CAREFULLY.
2. Choose a DIFFERENT VALID action that achieves a similar result.
3. Replan from that point forward.
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "Action": {
                        "type": "string",
                        "enum": list_of_all_actions
                    }
                },
                "required": ["Action"]
            }
        ),
        Tool(
            name="Control_Servo",
            description="""- Controls individual actuators in the robot's head.
- This tool accepts one parameter: "Servo Position" (integer from 1000 to 2000).
- 1500 means looking straight ahead.
- Values below 1500 look down; values above 1500 look up.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "Servo Position": {"type": "integer", "minimum": 1000, "maximum": 2000}
                },
                "required": ["Servo Position"]
            }
        ),
        Tool(
            name="Capture_Image",
            description="""- Captures an image and runs the vision model on what is in the robot's sight.
- This tool accepts:
    - "Request": a non-empty string describing the objects to look for. You can use a ';' separated string for multiple objects, e.g. "red ball;blue cup".
    - "BoundaryColors": an optional ';' separated string of RGB values with the same number of items as the Request list, e.g. "0,0,256;0,256,0".
- Use this when you need precise detection and bounding boxes for specific objects.
""",
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
            name="Summarize_Scene",
            description=("""- Captures an image from the robot's camera and uses a Vision Language Model (VLM) to describe what it sees.
- Takes no parameters.
- Use this before planning a complex VisionLanguageInterpreter-style task to get a natural-language overview of the scene.
- The VLM will return a description like "I see a red ball on the left and a blue cup in the corner."
"""
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        # RL add 
        Tool(
            name="Pick_Object",
            description=(
"""- It executes the naviation algorithm that makes the robot go near the object that the user mentioned.
- This tool accepts:
    - "object_description": a non-empty string describing [color] and [object]
- Use this when the user commands you to navigate to any object and/or pick up/fetch any objects"""
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
        if name == "Propagate_Action":
            action = arguments.get("Action")
            result = propagate_action(action)
            if result["status"] == "success":
                return [TextContent(type="text", text=f"Action '{action}' executed successfully")]
            else:
                return [TextContent(type="text", text=f"Action failed: {result['error']}")]
        
        elif name == "Control_Servo":
            servo_position = arguments.get("Servo Position")
            result = control_servo(servo_position)
            if result["status"] == "success":
                return [TextContent(type="text", text=f"Servo set to position {servo_position} successfully")]
            else:
                return [TextContent(type="text", text=f"Servo control failed: {result['error']}")]
        
        elif name == "Capture_Image":
            request = arguments.get("Request")
            boundary_colors = arguments.get("BoundaryColors", "")
            result = capture_image(request, boundary_colors)
            if result["status"] == "success":
                return [TextContent(type="text", text=f"Image captured and processed for: {request}")]
            else:
                return [TextContent(type="text", text=f"Image capture failed: {result['error']}")]
        
        elif name == "Summarize_Scene":
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
                
        elif name == "Pick_Object":
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