# Remove unnecessary warnings from UI display
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # code to only show critical errors
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import asyncio
import json
import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import re
from test_tools import * 


# Planner configuration 
OLLAMA_SERVER = "http://127.0.0.0:11434"
# ollama run smollm2:1.7b
MODEL_NAME = "qwen3:1.7b" # "deepseek-r1:1.5b" # "smollm2:1.7b" #       
IS_THINKING = True
SYS_PROMPT = """
You are a robot with a physical body: a camera (head), legs, and hands. Your body is bipedal. You can move the robot and look around the environment, and you have the following tools available to control it.

CRITICAL ACTION MAPPING RULES:
- "move right" or "go right" MUST use "right_move" action
- "move left" or "go left" MUST use "left_move" action  
- "move forward" or "go forward" or "walk forward" MUST use "go_forward" action
- "move backward" or "go backward" MUST use "back" action
- "turn right" MUST use "right_move" action
- "turn left" MUST use "left_move" action
- "pick up" MUST use "catch_ball" action
- "put down" MUST use "put_down" action
- "look up" MUST use "Control Servo" with position > 1500
- "look down" MUST use "Control Servo" with position < 1500

Tools:
1) "Propagate Action"
   - Executes one of the predefined actions from the action group.
   - This tool accepts one parameter: "Action" (string).
   - The Action must be exactly one of the allowed actions listed below.
   - Some action meanings:
       - left_move: move the robot left
       - right_move: move the robot right  
       - go_forward: move the robot forward
       - back: move the robot backward
       - left_shot / left_shot_fast: use the left leg to kick an object (faster for _fast).
       - right_shot / right_shot_fast: use the right leg to kick an object (faster for _fast).
       - move_up: move the hands/arms up to pick up or reach toward an object.
       - catch_ball: pick up and lift an object (not just a ball).
       - catch_ball_up: lift the object up above the head.
       - catch_ball_go: walk forward only while holding the object.
       - catch_ball_left_move / catch_ball_right_move: move left/right while holding the object.
       - put_down: place the held object back down.
       - wing_chun: martial arts but can be use as a dance move.

2) "Control Servo"
   - Controls individual actuators in the robot's head.
   - This tool accepts one parameter: "Servo Position" (integer from 1000 to 2000).
   - 1500 means looking straight ahead.
   - Values below 1500 look down; values above 1500 look up.

3) "Capture Image"
   - Captures an image and runs the vision model on what is in the robot's sight.
   - This tool accepts:
       - "Request": a non-empty string describing the objects to look for. You can use a ';' separated string for multiple objects, e.g. "red ball;blue cup".
       - "BoundaryColors": an optional ';' separated string of RGB values with the same number of items as the Request list, e.g. "0,0,256;0,256,0".
   - Use this when you need precise detection and bounding boxes for specific objects.

4) "Summarize Scene"
   - Captures an image from the robot's camera and uses a Vision Language Model (VLM) to describe what it sees.
   - Takes no parameters.
   - Use this before planning a complex VisionLanguageInterpreter-style task to get a natural-language overview of the scene.
   - The VLM will return a description like "I see a red ball on the left and a blue cup in the corner."

5) "Pick Object"
   - It executes the naviation algorithm that makes the robot go near the object that the user mentioned.
   - This tool accepts:
        - "object_description": a non-empty string describing [color] and [object]
   - Use this when the user commands you to navigate to any object and/or pick up/fetch any objects

Formatting rules:
- CRITICAL: You MUST respond with ONLY valid JSON. No other text, no explanations, no markdown, and no thinking tags.
- All tool names and action names MUST exactly match the allowed list.
- If the JSON is invalid or uses unknown tool names, the system will fail.

For each user input, reply ONLY with valid JSON in the form:
{
    "response": "string reply to user",
    "plan": [
        {
            "step": 1,
            "tool": "Tool Name",
            "params": {"ParameterName": "ParameterValue"}
        }
    ]
}

CRITICAL VALIDATION RULES:
- When using "Propagate Action", you MUST use EXCATLY one of the allowed actions listed below.
- If you try to use an action not in the allowed list, the system will FAIL with a validation error.
- If you're unsure which action to use, choose the most semantically similar from the allowed list.
- For picking up objects, use: catch_ball, catch_ball_up, or move_up (if object is above).
- For moving while holding objects, use: catch_ball_go, catch_ball_left_move, catch_ball_right_move.
- For placing objects down, use: put_down.

ERROR HANDLING:
If you receive an error saying an action is "not one of" the allowed list, you MUST:
1. Check the allowed action list CAREFULLY.
2. Choose a DIFFERENT VALID action that achieves a similar result.
3. Replan from that point forward.

Rules for tools and params:
- "tool" must be exactly one of:
    "Propagate Action", "Control Servo", "Capture Image", "Summarize Scene", "Pick Object".
- "params" MUST match the schema:
    - For "Propagate Action": {"Action": "<one allowed action>"}
    - For "Control Servo": {"Servo Position": <integer 1000–2000>}
    - For "Capture Image": {"Request": "<string>", "BoundaryColors": "<string or empty>"}
    - For "Summarize Scene": {} (no parameters).
    - For "Pick Object": {"object_description": "<string>"} 
- "plan" is an ordered list of steps.
- "step" starts from 1 and increases in execution order.

Action group (allowed actions):
back, back_end, back_fast, back_one_step, bow,
go_forward, go_forward_end, go_forward_fast, go_forward_one_small_step, go_forward_one_step, go_forward_start, go_forward_start_fast,
left_kick, left_move_10, left_move_20, left_move_30, left_move, left_move_fast, left_shot, left_shot_fast, left_uppercut,
right_kick, right_move_10, right_move_20, right_move_30, right_move, right_move_fast, right_shot, right_shot_fast, right_uppercut,
sit_ups, squat, squat_down, squat_up,
stand, stand_slow, stand_up_back, stand_up_front,
move_up, put_down, wave, wing_chun,
catch_ball, catch_ball_up, catch_ball_go, catch_ball_left_move, catch_ball_right_move.

REPETITION HANDLING:
- If the user says "twice", "two times", "2 times", create 2 identical steps in the plan
- If the user says "three times", "3 times", "thrice", create 3 identical steps
- If the user says "four times", "4 times", create 4 identical steps
- If the user says "five times", "5 times", create 5 identical steps
- Example: "move right twice" should create: [{"Action": "right_move"}, {"Action": "right_move"}]
- Example: "wave three times" should create: [{"Action": "wave"}, {"Action": "wave"}, {"Action": "wave"}]

VisionLanguageInterpreter-style behavior:
When the user asks you to understand a scene or execute a complex, multi-step task involving objects in front of the humanoid (for example: "walk to the red block and pick it up and put it down"):
1) First gather visual context:
   - You MUST call "Summarize Scene" to quickly understand what is around the robot.
   - If you need specific objects or bounding boxes, call "Capture Image" with a meaningful Request string such as "red block;blue cup" and optional BoundaryColors.

2) Form a high-level plan in your head like a symbolic planner:
   - Think in short conceptual steps such as:
     "locate OBJECT", "turn_toward OBJECT", "walk_to OBJECT", "center_on OBJECT", "reach_for OBJECT", "grasp OBJECT", "carry OBJECT", "put_down OBJECT", "return_to_start".
   - These are not tool names; they are internal planning steps.

3) Convert that high-level plan into concrete tool calls in the JSON "plan":
   - Use "Propagate Action" for high-level movement and predefined behaviors e.g., walking forward, moving left/right, kicking, catching, putting down.
   - Use "Control Servo" when you need fine-grained head/servo motion, for example to adjust where the head is looking.
   - Use "Capture Image" again if you need to re-check the scene during a longer plan. Use this as many times as needed to ensure the robot knows where objects are within its view while working towards completion of plan.
   - Use "Summarize Scene" again if the user asks you to report what you currently see.
   - use "Pick Object" if you need to pick up an object during the plan execution.

4) Keep the "response" field short and user-friendly, and put the detailed execution sequence in the "plan" array. The "plan" should reflect the steps you intend to execute in order.

Example for a simple wave:
{
    "response": "I will wave hello.",
    "plan": [
        {
            "step": 1,
            "tool": "Propagate Action",
            "params": {"Action": "wave"}
        }
    ]
}

Example for "move right twice":
{
    "response": "I will move right twice.",
    "plan": [
        {
            "step": 1,
            "tool": "Propagate Action",
            "params": {"Action": "right_move"}
        },
        {
            "step": 2, 
            "tool": "Propagate Action",
            "params": {"Action": "right_move"}
        }
    ]
}

DO NOT include any explanations or reasoning outside of the JSON. Only return the JSON.
"""
list_of_all_actions = [
    "back", "back_end", "back_fast", "back_one_step", "bow",
    "go_forward", "go_forward_end", "go_forward_fast", "go_forward_one_small_step",
    "go_forward_one_step", "go_forward_start", "go_forward_start_fast",
    "left_kick", "left_move_10", "left_move_20", "left_move_30", "left_move",
    "left_move_fast", "left_shot", "left_shot_fast", "left_uppercut", 
    "right_kick", "right_move_10", "right_move_20", "right_move_30",
    "right_move", "right_move_fast", "right_shot", "right_shot_fast",
    "right_uppercut", "sit_ups", "squat", "squat_down", "squat_up",
    "stand", "stand_slow", "stand_up_back", "stand_up_front",
    "put_down", "wave", "wing_chun", "catch_ball", "catch_ball_up", 
    "catch_ball_go", "catch_ball_left_move", "catch_ball_right_move",
]


previous_action = lambda msg : f"previous action: '{msg}'"

class MCPClient:
    def __init__(self):
        self.session = None
        self.stdio = None
        self.write = None
        
        self.previous_plan = "None."

    async def run(self):
        """Connect to MCP server"""
        try:
            server_params = StdioServerParameters(
                command="python", args=["main_mcp_server.py"]
            )
            print("Connecting to robot MCP server...")

            async with stdio_client(server_params) as (stdio, write):
                self.stdio = stdio
                self.write = write

                async with ClientSession(self.stdio, self.write) as session:
                    self.session = session
                    await session.initialize()

                    # List available tools
                    tools_response = await session.list_tools()
                    print("\nAvailable Tools:")
                    for tool in tools_response.tools:
                        print(f"  - {tool.name}: {tool.description}")

                    await self.interactive_chat()
                    # return True

        except Exception as e:
            print(f"Connection failed: {e}")
            # return False

    def get_ollama_plan(self, user_input: str):
        """Get planning and tools sequence from qwen3:1.7b"""
        # repetition_patterns = [
        #     (r'twice', 2),
        #     (r'thrice', 3),
        #     (r'three times', 3),
        #     (r'four times', 4),
        #     (r'five times', 5),
        #     (r'(\d+)\s*times', lambda m: int(m.group(1))),
        #     (r'(\d+)\s*', lambda m: int(m.group(1))),
        #     (r'^(\d+)\s*', lambda m: int(m.group(1)))
        # ]
        # repetitions = 1
        # clean_input = user_input

        # for pattern, rep in repetition_patterns:
        #     match = re.search(pattern, user_input.lower())
        #     if match:
        #         if callable(rep):
        #             repetitions = rep(match)
        #         else:
        #             repetitions = rep

        url = f"{OLLAMA_SERVER}/api/chat"
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_input},
                # {"role": "user", "content": previous_action(self.previous_plan) + "\ncurrent input:" + user_input},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            print("Sending request to Ollama...")
            response = requests.post(url, json=payload, timeout=30)
            print(payload['messages'][-1])
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "{}")
            # print(f"Raw content: {content}")

            # Parse JSON response
            try:
                plan_data = json.loads(content)
                self.previous_plan = plan_data # rl  mem 
                # what happens if there are repeated tasks? create syntax or update system prompt
                return plan_data
            except json.JSONDecodeError as e:
                print("JSON parsing error: {e}")
                print("Raw Ollamacontent:", repr(content))
                return {"response": "Planning failed", "plan": []}

        except Exception as e:
            print(f"Ollama request failed: {e}")
            return {"response": "Planning failed", "plan": []}

    async def check_and_replan(
        self, step_result: str, current_plan: list, step_index: int, user_input: str
    ) -> tuple[bool, list]:
        """Check if any step failed and attempt corrective re-planning"""
        error_keywords = [
            "error",
            "failed",
            "invalid",
            "not one of",
            "validation",
            "validation error",
        ]
        if any(keyword.lower() in step_result.lower() for keyword in error_keywords):
            print(f" [Replanning] Detected error in step: {step_result}")

            try:
                scene_result = await self.session.call_tool("Summarize Scene", {})
                scene_text = (
                    scene_result.content[0].text
                    if scene_result.content
                    else "No scene available"
                )
            except:
                scene_text = "Unable to capture scene"

            replan_prompt = f"""
            The previous plan failed with error: {step_result}. You are correcting a failed robot plan.
            Current scene: {scene_text}
            Original user command: {user_input}
            
            IMPORTANT REPETITION RULES:
            - If the user said "twice", create 2 identical steps
            - If the user said "3 times", create 3 identiical steps
            - If the user said "five times", create 5 identiical steps
            - For "move right twice", create TWO "right_move" actions
            - For "walk forward 3 times", create THREE "go_forward" actions
            
            Create a corrected plan using ONLY VALID tools and actions from the allowed list.
            CRITICAL:Use ONLY these action names: {list_of_all_actions}
            Return ONLY JSON with a corrected plan.
            Original system prompt: {SYS_PROMPT}
            """

            # Corrected plan
            corrected_plan_data = self.get_ollama_plan(replan_prompt)
            if corrected_plan_data and "plan" in corrected_plan_data:
                corrected_plan = corrected_plan_data.get(
                    "plan", []
                )  # replace remaining steps with new plan
                executed_steps = current_plan[
                    :step_index
                ]  # keep executed steps and just replace futre ones
                new_steps = []
                for i, step in enumerate(corrected_plan):
                    new_steps.append(
                        {
                            "step": step_index + i + 1,
                            "tool": step["tool"],
                            "params": step["params"],
                        }
                    )

                updated_plan = executed_steps + new_steps
                return True, updated_plan
        return False, current_plan

    async def execute_plan(self, plan_data: dict, user_input: str):
        """Execute the planned tool sequence"""
        if not plan_data or "plan" not in plan_data:
            return "No plan to execute"

        response_text = plan_data.get("response", "")
        plan = plan_data.get("plan", [])
        execution_log = []
        execution_log.append(f"Initial response: {response_text}")
        execution_log.append(f"Execution plan with {len(plan)} steps:")

        # Execute each step in plan
        i = 0
        while i < len(plan):
            step = plan[i]
            step_num = step.get("step", i + 1)
            tool_name = step.get("tool", "")
            params = step.get("params", {})
            execution_log.append(f"Step {step_num}: {tool_name} with params {params}")
            # for step in plan:
            #     step_num = step.get("step", 0)
            #     tool_name = step.get("tool", "")
            #     params = step.get("params", {})
            #     execution_log.append(f"Step {step_num}: {tool_name} with params {params}")

            try:
                # Executing tool via MCP server tool call
                result = await self.session.call_tool(tool_name, params)
                if result and result.content:
                    tool_result = result.content[0].text
                    execution_log.append(f"  Result: {tool_result}")

                    # check for errors & replan when needed
                    should_replan, new_plan = await self.check_and_replan(
                        tool_result, plan, i, user_input
                    )
                    if should_replan:
                        execution_log.append(
                            f" [Replanning] Generating corrected plan..."
                        )
                        plan = new_plan  # corrected plan
                        continue

                else:
                    execution_log.append(f"  Result: No response from tool")

            except Exception as e:
                execution_log.append(f"  Error: {repr(e)}")
                # if exception error needs replanning
                should_replan, new_plan = await self.check_and_replan(
                    repr(e), plan, i, user_input
                )
                if should_replan:
                    execution_log.append(f" [Replanning] Generating corrected plan...")
                    plan = new_plan  # corrected plan
                    continue

            # Make a brief pause between each step and only execute next step if no replanning occurs
            i += 1
            await asyncio.sleep(0.5)

        execution_summary = "\n".join(execution_log)
        final_analysis = await self.get_final_analysis(user_input, execution_summary)
        execution_log.append(f"\n{final_analysis}")
        return "\n".join(execution_log)

    async def get_final_analysis(self, user_input: str, execution_summary: str):
        """Get final analysis from LLM about the execution results"""
        url = f"{OLLAMA_SERVER}/api/chat"
        prompt = f"""
        User asked: "{user_input}"
        The robot executed this plan: {execution_summary}
        Based on the executed results, provide a concise final response to the user about what was accomplished and what was found.
        Example: 
        Final analysis: The action 'wave' was executed successfully, and the user's request to wave was completed.
        """
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            result = response.json()
            content = result.get("message", {}).get("content", "No analysis available")
            filtered_analysis = re.sub(
                r"<think>.*?</think>", "", content, flags=re.DOTALL
            ).strip()
            return filtered_analysis
        except Exception as e:
            return f"Analysis unavailable: {str(e)}"

    async def interactive_chat(self):
        """Main interactive chat loop"""
        print("\n" + "=" * 40)
        print("ROBOT PLANNING AND CONTROL INTERFACE")
        print("=" * 40)
        print("Give commands to the robot. The system will:")
        print("1. Plan the sequence of tools needed")
        print("2. Execute each tool in order")
        print("3. Report the execution results")
        print("Type 'quit' to exit")
        print("=" * 40)

        while True:
            try:
                user_input = input("\nYour command: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if not user_input:
                    continue

                print(f"\nProcessing: '{user_input}'")

                # Step 1: get plan from Ollama
                print("Getting execution plan from Ollama...")
                plan_data = self.get_ollama_plan(user_input)
                
                
                print(plan_data)
                # verify_compliance(plan_data, list_of_all_actions, ) 

                
                # continue

                # Step 2: execute the plan 
                print("Executing plan...")
                execution_result = await self.execute_plan(plan_data, user_input)

                # Step 3: display results
                print(f"\nExecution Results:")
                print(execution_result)

            except KeyboardInterrupt:
                print("\nSession interrupted")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
async def automated_chat(self, task_category = "task_a"):
        """Main interactive chat loop"""
        print("\n" + "=" * 40)
        print("ROBOT PLANNING AND CONTROL INTERFACE")
        print("=" * 40)
        print("Give commands to the robot. The system will:")
        print("1. Plan the sequence of tools needed")
        print("2. Execute each tool in order")
        print("3. Report the execution results")
        print("Type 'quit' to exit")
        print("=" * 40)

        task = {
            "task_a":[
                "Wave.",
                "Move right.",
                "Move left.",
                "Move forward one step.",    
            ],
            "task_b": [
                "Move 3 steps forward and move right.",
                "Move left twice.",
                "Move forward 2 steps, move right 2 steps, then wave.",
                "Move backward one step, then move forward one step.",
            ],
            "task_c": [
                "Grab the pen.",
                "Pick up the bottle.",
                "Pick up the cup.",
                "Fetch the red block.",
            ],
            "task_d": [  
                "Grab the pen, then move right.",
                "Grab the bottle, move forward one step, then put it down.",
                "Pick up the cup, move left twice, then put it down.",
                "Grab the red block, wave, then put it down.",
            ]
        }
        task_a = task[task_category]
        task_count = 0 
        total_prompts = len(task_a) * 6 
        while True and (task_count != total_prompts):
            try:
                user_input = task_a[task_count % len(task_a)] # input("\nYour command: ").strip()
                task_count += 1 
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                if not user_input:
                    continue

                print(f"\nProcessing: '{user_input}'")

                # Step 1: get plan from Ollama
                print("Getting execution plan from Ollama...")
                plan_data = self.get_ollama_plan(user_input)
                
                
                print(plan_data)
                # verify_compliance(plan_data, list_of_all_actions, ) 

                
                # continue

                # Step 2: execute the plan 
                print("Executing plan...")
                execution_result = await self.execute_plan(plan_data, user_input)

                # Step 3: display results
                print(f"\nExecution Results:")
                print(execution_result)

            except KeyboardInterrupt:
                print("\nSession interrupted")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

async def main():
    print("Starting MCP Client...")
    client = MCPClient()
    await client.run()
    print("Session completed")


if __name__ == "__main__":
    asyncio.run(main())
