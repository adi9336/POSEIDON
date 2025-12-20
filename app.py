import streamlit as st
import sys
import os

# Add the src directory to the path so we can import from agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.react_agent import ReactAgent


def initialize_agent():
    """Initialize the ReactAgent and store it in session state."""
    if "agent" not in st.session_state:
        try:
            with st.spinner("Initializing agent..."):
                st.session_state.agent = ReactAgent()
            st.success("✓ Agent initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.info("Please make sure OPENAI_API_KEY is set in your .env file")
            st.stop()


def main():
    st.set_page_config(
        page_title="🌊 Oceanographic Data Analysis Agent",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize the agent
    initialize_agent()

    # Initialize session state for chat history if it doesn't exist
    if "history" not in st.session_state:
        st.session_state.history = []

    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

    # Sidebar for controls and information
    with st.sidebar:
        st.title("🌊 Controls")

        # Clear chat history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.agent.clear_history()
            st.session_state.history = []
            st.session_state.message_count = 0
            st.rerun()

        st.markdown("---")

        # Statistics
        st.markdown("### 📊 Statistics")
        st.metric("Total Messages", st.session_state.message_count)
        st.metric(
            "Conversations",
            len([m for m in st.session_state.history if m["role"] == "user"]),
        )

        st.markdown("---")

        # About section
        st.markdown("### 📖 About")
        st.markdown("""
        This AI agent can answer questions about:
        
        **Measurements:**
        - 🌡️ Water temperature
        - 🧂 Salinity levels
        - 🌊 Ocean currents
        - 📏 Depth measurements
        - 🧪 Nitrate levels
        
        **Analysis:**
        - 📈 Trends over time
        - 📊 Statistical summaries
        - 🗺️ Location-based queries
        - ⏰ Time series data
        """)

        st.markdown("---")

        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - "What's the temperature near Mumbai in January 2024?"
            - "Show me salinity trends from last month"
            - "Temperature at 500m depth in the Arabian Sea"
            - "Nitrate levels for the last 30 days"
            """)

        st.markdown("---")
        st.caption("Powered by LangChain & OpenAI")

    # Main chat interface
    st.title("🌊 Oceanographic Data Analysis Agent")
    st.markdown("Ask me anything about oceanographic data!")

    # Display chat history
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about oceanographic data..."):
        # Add user message to chat history
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.message_count += 1

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking and retrieving data..."):
                try:
                    # Run the agent
                    response = st.session_state.agent.run(prompt, use_history=True)
                    st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.session_state.message_count += 1

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.history.append(
                        {"role": "assistant", "content": error_msg}
                    )
                    st.session_state.message_count += 1

        # Rerun to update the display
        st.rerun()


if __name__ == "__main__":
    main()
