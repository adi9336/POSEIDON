# src/agent/react_agent.py
import os
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, create_react_agent, tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ReactAgent:
    def __init__(self):
        """Initialize the ReactAgent with tools and history."""
        self.history = []
        self.tools = self._setup_tools()
        self.agent_executor = self._create_agent()

    def _setup_tools(self) -> List[Tool]:
        """Set up the tools available to the agent."""

        # Create a bound method for the tool function
        def retrieve_data_tool(query: str) -> str:
            return self._retrieve_data(query)

        return [
            Tool(
                name="retrieve_data",
                func=retrieve_data_tool,
                description=(
                    "Useful for retrieving oceanographic data based on natural language queries. "
                    "Input should be a natural language question about ocean data. "
                    "The tool can answer questions about: "
                    "water temperature, salinity levels, ocean currents, depth measurements, "
                    "time series data, and other oceanographic measurements. "
                    "Examples: 'temperature at Mumbai in January 2024', 'salinity trend last month'"
                ),
                return_direct=False,
            )
        ]

    def _retrieve_data(self, query: str) -> str:
        """Wrapper function to retrieve data using the existing workflow."""
        from src.agent.Retrieving_Agent import run_argo_workflow

        try:
            print(f"🔍 Retrieving data for: {query}")
            result = run_argo_workflow(query)
            answer = result.get("final_answer", "No data found for the query.")
            print("✓ Data retrieved successfully")
            return answer
        except Exception as e:
            error_msg = f"Error retrieving data: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg

    def _create_agent(self) -> AgentExecutor:
        """Create and return a ReAct agent with proper prompt template."""
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. Please set it in your .env file."
            )

        llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key)

        # Create the ReAct prompt template
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        # Create the prompt with all required variables
        prompt = PromptTemplate(
            template=template,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        )

        # Get tool descriptions and names
        tool_strings = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        tool_names = ", ".join([tool.name for tool in self.tools])

        # Create the agent with the prompt and tools
        agent = create_react_agent(llm=llm, tools=self.tools, prompt=prompt)

        # Create and return the agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=5,  # Limit iterations to prevent infinite loops
            max_execution_time=60,  # Timeout after 60 seconds
        )

    def run(self, query: str, use_history: bool = False) -> str:
        """Run the agent with the given query.

        Args:
            query: The user's question
            use_history: Whether to include chat history (for future enhancement)

        Returns:
            The agent's response
        """
        try:
            print(f"\n{'='*60}")
            print(f"🤖 Processing Query: {query}")
            print(f"{'='*60}")

            result = self.agent_executor.invoke(
                {"input": query, "agent_scratchpad": ""}
            )

            # Store in history if needed
            if use_history:
                self.history.append({"query": query, "response": result["output"]})

            return result["output"]

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"\n❌ {error_msg}")
            return error_msg

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
        print("✓ Chat history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.history


def main():
    """Main function to run the interactive agent."""
    try:
        print("\n" + "=" * 60)
        print("🌊 Oceanographic Data Analysis Agent")
        print("=" * 60)
        print("Initializing ReAct Agent...")

        # Initialize the agent
        agent = ReactAgent()
        print("✓ Agent ready!")

        print("\nCommands:")
        print("  - Type your question to query oceanographic data")
        print("  - Type 'history' to see conversation history")
        print("  - Type 'clear' to clear history")
        print("  - Type 'exit' or 'quit' to exit")
        print("=" * 60)

        while True:
            try:
                query = input("\n🔵 Your question: ").strip()

                if not query:
                    continue

                if query.lower() in ("exit", "quit", "q"):
                    print("\n👋 Goodbye!")
                    break

                if query.lower() == "history":
                    print("\n📝 Conversation History:")
                    for i, item in enumerate(agent.get_history(), 1):
                        print(f"\n{i}. Q: {item['query']}")
                        print(f"   A: {item['response'][:100]}...")
                    continue

                if query.lower() == "clear":
                    agent.clear_history()
                    continue

                # Process the query
                response = agent.run(query, use_history=True)
                print(f"\n{'='*60}")
                print(f"🤖 Final Answer:")
                print(f"{'='*60}")
                print(response)
                print(f"{'='*60}")

            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"\n❌ Failed to initialize agent: {str(e)}")
        print("Please make sure you have:")
        print("  1. Set OPENAI_API_KEY in your .env file")
        print("  2. Installed required packages: langchain, langchain-openai")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
