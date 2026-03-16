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

### ASR ###
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np

MODEL_WHISPER = os.getenv("MODEL_WHISPER", "tiny.en")

### TTS ###
# TTS Config
import io
import soundfile as sf
import emoji
import pyaudio
import wave

try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("Kokoro-ONNX not found. Run: pip install kokoro-onnx sounddevice soundfile")

KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "./ignore/kokoro_tts/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "./ignore/kokoro_tts/voices-v1.0.bin")
KOKORO_VOICE_NAME = os.getenv("KOKORO_VOICE_NAME", "af_sarah")

# --- GLOBAL STATE ---
kokoro_tts = None
whisper = None

# --- INITIALIZATION ---
def init_resources():
    global kokoro_tts, whisper
    
    print("Initializing resources...")
    
    # 1. Kokoro TTS
    if KOKORO_AVAILABLE and os.path.exists(KOKORO_MODEL_PATH) and os.path.exists(KOKORO_VOICES_PATH):
        print("Loading Kokoro TTS...")
        kokoro_tts = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
    else:
        print("Kokoro TTS not available or paths missing.")

    # 2. Whisper ASR
    print(f"Loading Whisper model: {MODEL_WHISPER}...")
    whisper = WhisperModel(MODEL_WHISPER, device="cpu", compute_type="int8")
    
    print("Initialization complete.")

# Run initialization on startup
init_resources()

# --- HELPER FUNCTIONS ---
def record_audio(duration=5, sample_rate=16000) -> np.ndarray:
    """Record audio from microphone."""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def record_until_silence(sample_rate=16000, silence_threshold=0.01, silence_duration=1.5, max_duration=10) -> np.ndarray:
    """Record audio until silence is detected."""
    chunk_size = int(sample_rate * 0.1)  # 100ms chunks
    audio_chunks = []
    silent_chunks = 0
    silent_chunks_needed = int(silence_duration / 0.1)

    # print("Listening...")
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            chunk_flat = chunk.flatten()
            audio_chunks.append(chunk_flat)

            # Check volume
            volume = np.abs(chunk_flat).mean()
            if volume < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0  # reset on speech

            # Stop if silence detected after speech, or max duration hit
            total_duration = len(audio_chunks) * 0.1
            if silent_chunks >= silent_chunks_needed and total_duration > 1.0:
                break
            if total_duration >= max_duration:
                break

    return np.concatenate(audio_chunks)

def transcribe_audio(audio: np.ndarray, sample_rate=16000) -> str:
    """Transcribe audio using Whisper."""
    if not whisper:
        return ""
    segments, _ = whisper.transcribe(audio, beam_size=5)
    return " ".join(segment.text for segment in segments).strip()

def clean_for_tts(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[\*\#\_\`]', '', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def generate_audio_response(text):
    """Generates audio bytes from text using Kokoro."""
    if not kokoro_tts:
        return None
    
    try:
        # Run in executor to avoid blocking the async loop
        loop = asyncio.get_event_loop()
        samples, sample_rate = await loop.run_in_executor(
            None, 
            lambda: kokoro_tts.create(text, voice=KOKORO_VOICE_NAME, speed=1.0, lang="en-us")
        )
        
        # Convert numpy float32 array to 16-bit PCM bytes (WAV format)
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def play_audio(audio_bytes: bytes):
    buffer = io.BytesIO(audio_bytes)
    with wave.open(buffer, 'rb') as wf:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        stream.write(wf.readframes(wf.getnframes()))
        stream.stop_stream()
        stream.close()
        p.terminate()

# Planner configuration 
OLLAMA_SERVER = "http://100.67.254.11:11434"
# ollama run smollm2:1.7b
MODEL_NAME = "qwen3:1.7b" # "deepseek-r1:1.5b" # "smollm2:1.7b" #       
IS_THINKING = True
# MCP Server Path
mcp_server = "mcp-server-updated-jl.py"

# Robot Variables
# Describe to the model that it is a robot. 
# This is where you give the robot it's name as well as 
# a general description of it's bodily composition.
robot_description = """
You are a robot with a physical body: a camera (head), legs, and hands. 
Your body is bipedal, and you MUST use the tools available to control it. 
You are addressed as Erza.
"""

SYS_PROMPT = f"""
{robot_description}

Formatting rules:
- CRITICAL: You MUST respond with ONLY valid JSON. No other text, no explanations, no markdown, and no thinking tags.
- All tool names and action names MUST exactly match the allowed list.
- If the JSON is invalid or uses unknown tool names, the system will fail.

For each user input, reply ONLY with valid JSON in the form:
{json.dumps({
    "response": "string reply to user",
    "plan": [
        {
            "step": 1,
            "tool": "Tool Name",
            "params": {"ParameterName": "ParameterValue"}
        }
    ]
})}

REPETITION HANDLING:
- If the user says "twice", "two times", "2 times", create 2 identical steps in the plan
- If the user says "three times", "3 times", "thrice", create 3 identical steps
- If the user says "four times", "4 times", create 4 identical steps
- If the user says "five times", "5 times", create 5 identical steps

VisionLanguageInterpreter-style behavior:
When the user asks you to understand a scene or execute a complex, multi-step task:
1) First gather context.
2) Form a high-level plan in your head like a symbolic planner.
3) Convert that high-level plan into concrete tool calls in the JSON "plan".
4) Keep the "response" field short and user-friendly, and put the detailed execution sequence in the "plan" array. The "plan" should reflect the steps you intend to execute in order.

DO NOT include any explanations or reasoning outside of the JSON. Only return the JSON.
"""

previous_action = lambda msg : f"previous action: '{msg}'"

class MCPClient:
    def __init__(self):
        self.session = None
        self.stdio = None
        self.write = None
        
        self.previous_plan = "None."
        self.system_prompt = ""

    async def run(self):
        """Connect to MCP server"""
        try:
            server_params = StdioServerParameters(
                command="python", args=[mcp_server]
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
                    tools_summary = json.dumps([tool.model_dump() for tool in tools_response.tools], indent=2)
                    self.system_prompt = SYS_PROMPT + "\n\nAvailable Tools:\n" + tools_summary

                    print(f"System Prompt: \n{self.system_prompt}")
                    await self.interactive_chat()
                    # await self.automated_chat("task_b")

        except Exception as e:
            print(f"Connection failed: {e}")
            import traceback
            traceback.print_exc()  # ✅ shows the real error

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
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input},
                # {"role": "user", "content": previous_action(self.previous_plan) + "\ncurrent input:" + user_input},
            ],
            "stream": False,
            "format": "json",
        }

        try:
            print("Sending request to Ollama...")
            response = requests.post(url, json=payload, timeout=30)
            print(payload['messages'][-1:])
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
                scene_result = await self.session.call_tool("Summarize_Scene", {})
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
            CRITICAL:Use ONLY the provided action names.
            Return ONLY JSON with a corrected plan.
            Original system prompt: {self.system_prompt}
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
        print("Type 'q', 'quit', or 'exit' to exit")
        print("Listening automatically — just speak!")
        print("=" * 40)

        while True:
            try:
                # Step 1: Record until silence
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, record_until_silence
                )

                # Step 2: Transcribe
                print("Listening...")
                user_input = transcribe_audio(audio)
                if not user_input:
                    # print("Nothing detected, listening again...")
                    continue
                print(f"\nHeard: '{user_input}'")

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

                # Step 2. Generate TTS & play back
                clean_text = clean_for_tts(plan_data["response"])
                print(clean_text)
                audio_data = await generate_audio_response(clean_text)
                if audio_data:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: play_audio(audio_data))
                else:
                    print("No audio generated")
                
                # continue

                # # Step 3: execute the plan 
                # print("Executing plan...")
                # execution_result = await self.execute_plan(plan_data, user_input)

                # # Step 4: display results
                # print(f"\nExecution Results:")
                # print(execution_result)

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