# SLM-MCP Tonypi: Language-Guided Humanoid Robot Control
A modular humanoid robot control system that integrates Small Language Models (SLMs) and Vision-Language Models (VLMs) through the Model Context Protocol (MCP) for natural language command executions.

# Overview
This project enables intuitive control of a Hiwonder Tonypi humanoid robot using natural language. The system translates high-level user commands into executable robot actions through a distributed architecture that separates planning, perception, and execution components.

# System Architecture
1. MCP Server: Python-based server exposing robot control tools (movement, perception, manipulation).
2. MCP Client: Interactive interface for issuing commands and executing planned actions.
3. AI Models: Remote Ollama-hosted SLMs for task planning and VLMs for visual grounding.
4. Robot: Raspberry Pi 5-based Hiwonder Tonypi executing physical actions.

# Quick Start
1. Start the vision and language services by running the Ollama FastAPI server script in the ai_model_communication `ollama-server.sh` 
```bash
cd ai_model_communication

bash ollama-server.sh
```

2. For the mcp implementation, run these two scripts in different terminals: 

a) Run the MCP server in the first terminal
```bash
cd mcp-implement 

python main_mcp_server.py
```

b) In another terminal, run the MCP Client 
```bash
cd mcp-implement 

python main_mcp_client.py
```

# Features
1. Natural Language Control: issue commands like "wave hello" or "pick up the red block".
2. Visual Perception: object detection using GroundingDINO and scene description via VLM.
3. Modular Design: swap planners, perception models, or robot harware independently.
4. Distributed Execution: remote AI processing with local robot actuation over VPN.

# Requirements
1. Python 3.9+
2. Ollama with Qwen3 (1.7B) and Qwen3-VL (2B) models installed
3. Grounding Dino tiny 
4. Hiwonder Tonypi robot with Raspberry Pi 5
5. VPN connection between workstation and robot

# Citation
If you use this work, please reference our IEEE SoutheastCon 2026 paper: "From Language to Action: Small Language Model–Driven Humanoid Control with Visual Grounding Under the Model Context Protocol"
