def generate_plan(user_input, handle_tools=handle_tools):
    # Check if server is running
    if client.health_check():
        tools = [mcp_server]
        # Generate a response
        full_response = client.generate_response(
            model_name=config.MODEL_NAME,
            user_message=f"user: {user_input}",
            system_prompt="You are an agentic robot. Decide which of these tools should be used. Tools: {tools}",
            think=config.IS_THINKING
        )

        # Get the reasoning if available
        thinking = full_response['message'].get('thinking', '')

        # Get the message content 
        content_str = full_response['message']['content']

        # Parse the JSON content
        content_data = json.loads(content_str)

        # Extract components
        response_text = content_data.get('response', '')
        tools = content_data.get('Tools', [])

        #print("Full response:", full_response)
        print("Message:", response_text)
        print("Tools:", tools)

        ### Handle Tool Call ###
        handle_tools(tools)

    else:
        print("Ollama server is not running or not accessible")
        
def generate_response(user_input, handle_tools=handle_tools):
    # Check if server is running
    if client.health_check():
        tools = [mcp_server]
        # Generate a response
        full_response = client.generate_response(
            model_name=config.MODEL_NAME,
            messages=f"user: {user_input}; tool: {tool_response}; robot:",
            system_prompt="You are an agentic robot.",
            think=config.IS_THINKING
        )

        # Get the reasoning if available
        thinking = full_response['message'].get('thinking', '')

        # Get the message content 
        content_str = full_response['message']['content']

        # Parse the JSON content
        content_data = json.loads(content_str)

        # Extract components
        response_text = content_data.get('response', '')
        tools = content_data.get('Tools', [])

        #print("Full response:", full_response)
        print("Message:", response_text)
        print("Tools:", tools)

        ### Handle Tool Call ###
        handle_tools(tools)

    else:
        print("Ollama server is not running or not accessible")

