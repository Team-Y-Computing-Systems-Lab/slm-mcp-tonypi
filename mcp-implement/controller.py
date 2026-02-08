
import os, time, json, requests
from datetime import datetime 

import cv2 
from PIL import Image 

from vision_tools import vision 
from vision_tools.detection_data import BoundingBox, DetectionResult
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# logger.info("Server started")
# logger.debug("Some debug info")
# logger.error("Something went wrong")


BASE_DATA_DIR = './episodes'

MAX_HEAD_VAL = 2000 
MIN_HEAD_VAL = 1000 

MAX_STEPS = 100
CENTER_X_THRESH = .2 # normalized |cx - 0.5| 
CENTER_Y_THRESH = .09 # normalized 
TOLERANCE = 30 

STABILITY_FRAMES = 2
SLEEP_BETWEEN_ACTIONS = 1 

VISION_URL = "http://127.0.0.0:8000/dino_api"    # your detect endpoint
ROBOT_URL = "http://lab-erza.local"
TONYPI_RPC = "http://lab-erza.local:9030" # Hiwonder JSON-RPC server
HTTP_TIMEOUT = 5 

import requests

def detect_http(query: str, colors="red", url=VISION_URL):
    params = {
        "request": query,
        "boundaryColors": colors
    }

    try:
        r = requests.post(url, params=params, timeout=10)
        data = r.json()
    except Exception as e:
        logger.debug("HTTP detection error:", e)
        return None

    # Expected format:
    # {'detections': [ {...}, {...} ]}
    if "detections" not in data or len(data["detections"]) == 0:
        return None

    det_json = data["detections"][0]  # take first detection
    logger.debug(det_json)

    # No image dims available → can fill later in pick_and_place
    det = DetectionResult.from_json(det_json)
    det.box.set_image_size(image_width = data["image_width"], image_height = data["image_height"])
    return det


def rpc_run_action(action: str, times: int = 1) -> bool:
    payload = {
        'jsonrpc': '2.0',
        'method': 'RunAction',
        'params': [action, times],
        'id': int(time.time() *1000)%10**9
    }
    try:
        r = requests.post(
            TONYPI_RPC,
            json=payload,
            timeout=HTTP_TIMEOUT
        )
        ok = (r.status_code == 200)
        time.sleep(SLEEP_BETWEEN_ACTIONS)
        logger.debug('[+] action execution success')
        return ok
    except Exception as e :
        logger.debug(f'[-] failed to execute the action. Error {e}')
        return False

def control_servo(servo_position, head = 'v'):
    '''
    accept v as vertical and h as horizontal movement
    '''
    url = TONYPI_RPC
    payload = {
        'jsonrpc': '2.0',
        'method': 'SetPWMServo',
        'params': [1000, 2, 1, int(servo_position)],
        'id': 1,
    }
    try:
        r = requests.post(
            TONYPI_RPC,
            json=payload,
            timeout=HTTP_TIMEOUT
        )
        ok = (r.status_code == 200)
        time.sleep(SLEEP_BETWEEN_ACTIONS)
        logger.debug('[+] action execution success head')
        return ok
    except Exception as e :
        logger.debug(f'[-] head failed to execute the action. Error {e}')
        return False

def decide_action_from_bbox(b: BoundingBox, robot_state : dict):
    """
    bbox information to discrete action 
    """

    logger.debug(f'test {b.image_height}')

    ncx, ncy = b.n_cxcy
    n_area = b.n_area 
    n_bottom = b.n_bottom

    dx = ncx - 0.5 # horizontal eror; 
    dy = ncy - 0.72 # 5 # vertical error ; 

    features = {
            'ncx': b.ncx, 
            'ncy': b.ncy,
            'area': b.n_area, 
            'bottom': b.n_bottom, 
            'dx' : dx, 
            'dy' : dy,
            'prev_action' : robot_state['action']
        }

    response = {
        'action': None, 
        'head': robot_state['head'], 
        'end': None, 
    }

    response['action'] = "go_forward"
    if abs(dx) > CENTER_X_THRESH: 
        action = "left_move" if dx < 0 else "right_move"
        action += "_20" if abs(dx) > 0.4 else "_fast"
        response['action'] = action 
        robot_state['action'] = action 

    if dy > CENTER_Y_THRESH:
        robot_state['head'] -= dy * (MAX_HEAD_VAL - MIN_HEAD_VAL) 
    
    if robot_state['head'] <= (MIN_HEAD_VAL + TOLERANCE):
        response['end'] = True
        response['action'] = 'catch_ball'
        
    response['head'] = max(MIN_HEAD_VAL, robot_state['head'])
    robot_state['head'] = response['head']
    robot_state['action'] = response['action']



    rpc_run_action(action=response['action'])
    logger.debug(f"[head] {response['head'] },ncxcy  {ncy} dy {dy}, center y thresh {CENTER_Y_THRESH}")
    # if response['head']:
    control_servo(response['head'], 'v')

    return features, response, robot_state
    ...

def pick_object(object_description: str, after_pick = None, log_actions = None ):
    close_frames = 0 
    retry_detection = 0 
    episode = {}

    list_of_action_executed = [] 

    robot_state={
        "action" : None, 
        "head" : 1500,
    }
    control_servo(robot_state['head'], head = 'v')


    if log_actions:
        os.makedirs(
            BASE_DATA_DIR,
            exist_ok=True,
        )
        log_file = open(
            os.path.join(BASE_DATA_DIR,f"episode-"+ datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".json"),
            "w"
        )

    for step in range(MAX_STEPS):
        # camera capture
        # cap = cv2.VideoCapture(f"{ROBOT_URL}:8080")
        # if not cap.isOpened():
        #     logger.debug('cannot open camera')
        #     exit() 
        # ret, frame = cap.read() 
        # image = Image.fromarray(frame)

        # detection 
        # det = vision.detect(
        #     image= image,
        #     labels= query,
        # )

        det = detect_http(query=object_description)

        if det is None:
        # no detection so loop back retry after 
            # no detection of object 
            retry_detection += 1 
            if retry_detection > 5:
                # action to look around 
                ... 
            
            continue 
        
        # det.box.image_width = 100 # image.width
        # det.box.image_height = 100 # image.height
        logger.debug(f'[xxx] {det.box.image_height}')
        features, response, robot_state = decide_action_from_bbox(det.box, robot_state)
        

        if (response['action'] is None) and (response['head'] is None): 
            # close enough - confirm stability for a couple frames 
            close_frames += 1 
            if close_frames < STABILITY_FRAMES:
                time.sleep(0.05) 
                continue

        episode[step] = {
            'features': features,  # features are detailed robot state 
            'actions': response,
            'success': response['end'] 
        }

        logger.debug(f'[step]: {step}/{MAX_STEPS} : {episode[step]}')
        list_of_action_executed.append(robot_state['action'])
        if response['end']:
            break 

    # saving the episode json file 
    if log_actions:
        json.dump(episode, log_file, indent=4)
        logger.debug(f'[+] done episode on file: {log_file}')
        
    return list_of_action_executed
            
if __name__ == "__main__":

    #pick_and_place("blue and yellow container", log_actions=True)
    pick_object("pink box", log_actions=True)