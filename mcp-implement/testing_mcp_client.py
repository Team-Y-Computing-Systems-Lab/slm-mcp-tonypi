import asyncio
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class SimpleMCPClient:
    def __init__(self):
        self.session = None
        self.stdio = None
        self.write = None
        
    async def connect_and_chat(self):
        """Connect to MCP server and handle free-form user input"""
        try:
            # Connect to MCP server
            server_params = StdioServerParameters(command="python", args=["mcp_server_new.py"])
            print("Attempting to connect to robot MCP server...")
            
            async with stdio_client(server_params) as (stdio, write):
                self.stdio = stdio
                self.write = write
                
                async with ClientSession(self.stdio, self.write) as session:
                    print("Session initialized successfully")
                    
                    await asyncio.sleep(1.0)
                    init_result = await session.initialize()
                    print(f"Session initialized: {init_result}")
                    
                    # List available tools
                    tools_response = await session.list_tools()
                    print("\nAvailable Robot Tools:")
                    for tool in tools_response.tools:
                        print(f"  - {tool.name}: {tool.description}")
                    
                    # Start interactive chat
                    await self.interactive_chat(session)
            
        except FileNotFoundError:
            print("ERROR: Server script not found")
            print("   Check that mcp_server_new.py is in the right directory")
        except Exception as e:
            print(f"Connection Error: {type(e).__name__}")
            print(f"   Error details: {str(e)}")
            traceback.print_exc()

    async def interactive_chat(self, session):
        """Handle free-form user input and use robot_decide_and_act tool"""
        print("\n" + "="*20)
        print("ROBOT CONTROL INTERFACE")
        print("="*30)
        print("You can give free-form commands like:")
        print("  1. 'Walk towards the green ball and pick it up'")
        print("  2. 'Wave hello and then bow'")
        print("  3. 'Look around and describe what you see'")
        print("  4. 'Stand up and do a little dance'")
        print("  5. 'Find the red object and move towards it'")
        print("="*30)

        while True:
            try:
                # Get user input
                user_input = input("\nYour command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Ending robot session...")
                    break
                
                if not user_input:
                    continue
                
                print(f"Processing: '{user_input}'")
                
                # Decision tool that handles reasoning and action sequencing from the MCP server
                result = await session.call_tool("robot_decide_and_act", {"query": user_input})
                #print("\nRobot Tool Used:")
                #print(f"Tool used: '{user_input}'")
                                    
                # Display the result
                if result and result.content:
                    response_text = result.content[0].text
                    print(f"\nRobot Response:")
                    print(f"   {response_text}")
                else:
                    print("No response from robot")
                    
            except KeyboardInterrupt:
                print("\nSession interrupted by user")
                break
            except Exception as e:
                print(f"Error processing command: {str(e)}")
                print("   Please try a different command")

    async def execute_specific_action(self, session, action_name, params=None):
        """Execute a specific action directly to test or use our predefined specific commands"""
        try:
            if params is None:
                params = {}
            
            result = await session.call_tool(action_name, params)
            
            if result and result.content:
                return result.content[0].text
            return "No response"
        except Exception as e:
            return f"Error: {str(e)}"

    async def get_robot_status(self, session):
        """Get comprehensive robot status"""
        try:
            result = await session.call_tool("get_robot_status", {"detailed": True})
            if result and result.content:
                return result.content[0].text
            return "No status information"
        except Exception as e:
            return f"Error getting status: {str(e)}"

async def main():
    print('Starting MCP Robot Client...')
    print('='*25)
    client = SimpleMCPClient()
    await client.connect_and_chat()
    print('='*25)
    print('Session completed!')

if __name__ == "__main__":
    asyncio.run(main())