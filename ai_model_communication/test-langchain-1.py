# pip install langchain langchain-ollama pydantic requests

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import requests, json, time

from langchain_ollama.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ====== CONFIG: swap these with your endpoints ======
DETECT_URL = "http://<pi-or-server>:8000/detect"   # your "find image" API
ROBOT_URL  = "http://<pi-ip>:9030"                 # Hiwonder JSON-RPC (Werkzeug) root
TIMEOUT_S  = 8
RETRIES    = 2

# ====== Tool input schemas ======
class ObjQuery(BaseModel):
    """tool in put schemas"""
    object_name_attributes: str = Field(
        ...,
        description='Single object query like "red foam block" or "blue rectangular foam".'
    )

class RobotActions(BaseModel):
    """
    robot tools 
    """
    actions: List[str] = Field(
        ..., description='List of discrete robot actions to execute in order'
    )
    repeat: int = Field(1, ge=1, description="Times to repeat the whole sequence")

class CityWeather(BaseModel):
    city: str

# ====== Tools ======
@tool(parse_docstring=True, args_schema=ObjQuery)
def get_object_coordinates(object_name_attributes: str) -> Dict[str, Any]:
    """Fetch object coordinates (xmin,ymin,xmax,ymax,score) for a single described object."""
    payload = {"query": object_name_attributes}
    for attempt in range(1, RETRIES+1):
        try:
            r = requests.post(DETECT_URL, json=payload, timeout=TIMEOUT_S)
            r.raise_for_status()
            data = r.json()
            # normalize minimal contract
            return {
                "query": object_name_attributes,
                "bbox": {
                    "xmin": data.get("xmin"), "ymin": data.get("ymin"),
                    "xmax": data.get("xmax"), "ymax": data.get("ymax"),
                },
                "score": data.get("score")
            }
        except Exception as e:
            if attempt == RETRIES:
                return {"error": f"detect_failed: {e}", "query": object_name_attributes}
            time.sleep(0.3)

@tool(parse_docstring=True, args_schema=RobotActions)
def control_robot(actions: List[str], repeat: int = 1) -> Dict[str, Any]:
    """Send a sequence of actions to the robot over JSON-RPC RunAction (Hiwonder TonyPi)."""
    results = []
    for _ in range(repeat):
        for act in actions:
            payload = {
                "jsonrpc": "2.0",
                "method": "RunAction",
                "params": [act, 1],     # TonyPi expects [action_name, times]
                "id": int(time.time())
            }
            try:
                r = requests.post(ROBOT_URL, json=payload, timeout=TIMEOUT_S)
                ok = (r.status_code == 200)
                results.append({"action": act, "status": r.status_code, "body": r.text})
                if not ok:
                    return {"error": "robot_action_failed", "detail": results}
                # optional: short dwell between discrete actions
                time.sleep(0.15)
            except Exception as e:
                return {"error": f"robot_http_error: {e}", "partial": results}
    return {"ok": True, "executed": results}

@tool(parse_docstring=True, args_schema=CityWeather)
def get_weather(city: str) -> Dict[str, str]:
    """Return weather for a city (dummy example)."""
    # plug a real weather API here if you want
    return {city: "sunny and humid"}

# ====== LLM and loop ======
def main():
    sys_prompt = """You are a robot planner. Use tools to see and act.
Return *only* a JSON like:
{
  "response": "...",
  "action": ["..."],               // may be empty []
  "next_step": "..."               // omit this field when task is complete
}
Prefer calling get_object_coordinates before moving when the user asks to find/approach objects.
When you need to execute body motions, call control_robot with a list of discrete actions.
"""

    messages = [
        SystemMessage(sys_prompt),
        HumanMessage("Find the red foam block, approach, pick it up, then put it down.")
    ]

    tools = [get_object_coordinates, control_robot, get_weather]
    model = ChatOllama(
        model="llama3.1:8b",
        temperature=0.3,
        num_predict=512,
        validate_model_on_init=True,
    ).bind_tools(tools)

    tools_by_name = {t.name: t for t in tools}

    # tool loop
    for step in range(12):  # hard stop to avoid infinite loops
        ai: AIMessage = model.invoke(messages)
        messages.append(ai)

        # if the model emitted a final JSON answer (no tool calls), we’re done
        if not getattr(ai, "tool_calls", None):
            print(ai.content)  # should be your JSON {response, action, maybe next_step}
            break

        # otherwise execute each tool call and append ToolMessage back
        for tc in ai.tool_calls:
            name = tc["name"]
            args = tc.get("args", {})  # LangChain hands you parsed args already
            tool_fn = tools_by_name.get(name)
            if tool_fn is None:
                # echo back an error so the LLM can recover
                tm = ToolMessage(
                    name=name,
                    content=json.dumps({"error": f"unknown_tool:{name}", "args": args}),
                    tool_call_id=tc["id"]
                )
                messages.append(tm)
                continue

            result = tool_fn.invoke(args)  # runs your Python tool
            tm = ToolMessage(
                name=name,
                content=json.dumps(result),
                tool_call_id=tc["id"]
            )
            messages.append(tm)

    else:
        print(json.dumps({"response": "Planner reached step limit.", "action": []}))

if __name__ == "__main__":
    main()
