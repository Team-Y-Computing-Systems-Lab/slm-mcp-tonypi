# test_action_names.py
import requests
import time

ROBOT_URL = "http://100.71.207.226:9030"

def test_all_action_variations():
    print("Testing ALL possible action name variations...")
    
#     back, back_end, back_fast, back_one_step, 
# bow, 
# go_forward, go_forward_end, go_forward_fast, go_forward_one_small_step, go_forward_one_step, go_forward_start, go_forward_start_fast,
# left_kick, left_move_10, left_move_20, left_move_30, left_move, left_move_fast, left_shot <uses left leg to kick a ball>, left_shot_fast<kick a ball faster>, left_uppercut,
# right_kick, right_move_10, right_move_20, righth_move_30, right_move, right_move_fast, right_shot<uses right leg to kick a ball>, right_shot_fast <kick a ball fast>, right_uppercut,
# sit_ups, squat, squat_down, squat_up, stand, stand_slow, stand_up_back, stand_up_front, move_up <to pick up a block>, put_down <to place a block down>
# wave, 
# wing_chun
    
    
    # Test every possible action name format
    action_variations = [
        # Basic movements
        #"go_forward",
        # "wave", #working
        #"squat",
        #"stand",
        #"wing_chun"
        #"sit" 
        #"Sit"
        #"SIT" 
        #"nod", 
        #"Nod", 
        #"NOD", 
        #"nod_head", 
        #"head_nod"
        "back", "back_end", "back_fast", "back_one_step", "bow",
                            "go_forward", "go_forward_end", "go_forward_fast", "go_forward_one_small_step", 
                            "go_forward_one_step", "go_forward_start", "go_forward_start_fast", 
                            "left_kick", "left_move_10", "left_move_20", "left_move_30", "left_move", 
                            "left_move_fast", "left_shot", "left_shot_fast", "left_uppercut",
                            "right_kick", "right_move_10", "right_move_20", "right_move_30", "right_move", 
                            "right_move_fast", "right_shot", "right_shot_fast", "right_uppercut",
                            "sit_ups", "squat", "squat_down", "squat_up", "stand", "stand_slow", "stand_up_back", 
                            "stand_up_front", "put_down", "wave", "wing_chun", "catch_ball", "catch_ball_up",
                            "catch_ball_go", "catch_ball_left_move", "catch_ball_right_move"
        
        # From your LLM_BLOCK.py
        #"go_forward", "back_fast", "turn_left_small_step", "turn_right_small_step",
        #"left_move", "right_move", "left_move_30", "right_move_30",
        #"go_forward_one_step", "move_up", "Drop"
    ]
    
    successful_actions = []
    
    for action in action_variations:
        try:
            print(f"\nTrying: '{action}'")
            data = {
                "jsonrpc": "2.0",
                "method": "RunAction", 
                "params": [action, 1],
                "id": 1
            }
            
            response = requests.post(ROBOT_URL, json=data, timeout=10)
            print(f"Response: {response.status_code} - {response.text}")
            
            # Parse the JSON response to see if it actually executed
            if response.status_code == 200:
                try:
                    result_data = response.json()
                    # Check if the result indicates success
                    if result_data.get('result'):
                        print(f"✅ '{action}' - SUCCESS (result: {result_data['result']})")
                        successful_actions.append(action)
                    else:
                        print(f"⚠️  '{action}' - Command accepted but no positive result")
                except:
                    print(f"⚠️  '{action}' - Response not JSON format")
            else:
                print(f"❌ '{action}' - HTTP Error")
            
            # Brief pause between commands
            time.sleep(1)
                
        except Exception as e:
            print(f"❌ '{action}' - Error: {e}")
    
    print(f"\n🎯 SUCCESSFUL ACTIONS: {successful_actions}")
    return successful_actions

if __name__ == "__main__":
    successful = test_all_action_variations()
    if successful:
        print(f"\n💡 Use these working actions: {successful}")
    else:
        print("\n🔧 No actions worked - the robot service might need initialization")