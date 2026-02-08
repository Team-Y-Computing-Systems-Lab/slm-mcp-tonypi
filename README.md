# slm-mcp-tonypi
small language controlled tonypi robot using mcp orchestration.


# run the vision and slm planner 

run the script in the ai_model_communication `ollama-server.sh` 
```bash
bash ollama-server.sh
```

and for the mcp implement run two scirpt in different terminals 

MCP server 
```bash
cd mcp-implement 

python main_mcp_server.py
```

and in another terminal 


MCP Client 
```bash
cd mcp-implement 

python main_mcp_client.py
```