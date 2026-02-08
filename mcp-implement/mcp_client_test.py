import asyncio
#import json
#import os
#import requests
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class SimpleMCPClient:
    def __init__(self):
        self.session = None
        self.stdio = None
        self.write = None
        
    async def connect_and_test(self):
        try:
            server_params = StdioServerParameters(command="python", args=["mcp-implement/mcp_server_new.py"])
            print("Attempting to connect to server...")
            
            async with stdio_client(server_params) as (stdio, write):
                self.stdio = stdio
                self.write = write
                
                async with ClientSession(self.stdio, self.write) as session:
                    print("Session initialized successfully")
                    
                    await asyncio.sleep(1.0)
                    print("Initializing session...")
                    init_result = await session.initialize()
                    print(f"Session initialized result: {init_result}")        
                  
                    # List tool calls
                    tools_response = await session.list_tools()
                    print("\n=== Available Tools ===")
                    for tool in tools_response.tools:
                        print(f" - {tool.name}: {tool.description}")
        
                    # Test a tool call
                    print("\n Testing wave action")
                    movement_result = await session.call_tool("execute_movement", {"action": "wave", "times": 1})
                    print(f"Movement Result: {movement_result.content[0].text}")
            
        except FileNotFoundError:
            print("ERROR: Server script not found")
            print("Check that mcp_server_new.py is in the right directory")
        except Exception as e:
            print(f"Connection Error: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            traceback.print_exc()

async def main():
    print('Starting MCP Client Test...')
    print('='*40)
    client = SimpleMCPClient()
    await client.connect_and_test()
    print('='*40)
    print('Test complete!')
    
if __name__ == "__main__":
    asyncio.run(main())