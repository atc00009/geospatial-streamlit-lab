# Imports at top of app.py
from google import genai
from google.genai import types

# Initialize Gemini Client
gemini_key = st.secrets.get("GEMINI_API_KEY", None)
client = genai.Client(api_key=gemini_key) if gemini_key else None

# 7. AI ASSISTANT, STORYTELLING & REPORT GENERATOR
st.divider()
st.subheader("🤖 AI Spatial Assistant & Report Builder (Gemini Powered)")

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat with AI Assistant", 
    "✨ Auto-Generate AI Report", 
    "🧠 Lab Student Activities", 
    "🔍 View Raw Dataset"
])

# TAB 1: INTERACTIVE CHAT INTERFACE
with tab1:
    st.markdown("Ask the AI assistant any questions about analyzing spatial data, interpreting map hotspots, or structuring your lab report.")
    
    if not client:
        st.warning("⚠️ Gemini API key not detected. Please add `GEMINI_API_KEY` to your Streamlit secrets.")
    else:
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "model", 
                    "content": "Hello! I am your Spatial Data Science AI assistant powered by Gemini. Ask me how to interpret your active map data, analyze geographic hotspots, or write policy recommendations!"
                }
            ]

        for message in st.session_state.chat_messages:
            role = "assistant" if message["role"] == "model" else "user"
            with st.chat_message(role):
                st.markdown(message["content"])

        if user_prompt := st.chat_input("Ask a question about your spatial analysis or report structure..."):
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                system_context = f"""
                You are an expert Spatial Analytics Teaching Assistant. 
                The student is currently analyzing Chicago incident data with these parameters:
                - Selected Severities: {selected_severity}
                - Total Filtered Incidents: {len(filtered_df)}
                - Average Response Time: {filtered_df['Response_Time_Min'].mean():.1f} min if not filtered_df.empty else 0
                - Active Map Representation: {view_mode}

                Help the student understand spatial concepts (vector vs. raster/TIFF, point clusters vs boundary polygons), 
                and guide them on how to write professional analytical insights and policy recommendations.
                """

                try:
                    # Request streaming response from Gemini 2.5 Flash
                    response_stream = client.models.generate_content_stream(
                        model='gemini-2.5-flash',
                        contents=f"System Context: {system_context}\n\nUser Question: {user_prompt}"
                    )

                    full_response = ""
                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.chat_messages.append({"role": "model", "content": full_response})

                except Exception as e:
                    st.error(f"⚠️ Gemini API Call Failed: {e}")

# TAB 2: AUTOMATED REPORT GENERATOR
with tab2:
    st.markdown("Click below to generate a structured academic report based on your currently filtered dataset.")
    
    if not client:
        st.warning("⚠️ Gemini API key not detected. Please add `GEMINI_API_KEY` to your Streamlit secrets.")
    else:
        if st.button("🚀 Generate Spatial Story & Insights Report"):
            with st.spinner("Analyzing spatial patterns with Gemini..."):
                try:
                    avg_time = filtered_df['Response_Time_Min'].mean() if not filtered_df.empty else 0
                    total_incidents = len(filtered_df)
                    top_areas = filtered_df['Community_Area'].value_counts().head(3).to_dict() if not filtered_df.empty else {}
                    severity_counts = filtered_df['Severity'].value_counts().to_dict() if not filtered_df.empty else {}

                    report_prompt = f"""
                    You are a Senior Spatial Data Science Professor. Analyze the following Chicago geospatial incident data summary and write an engaging data story and report section for students.

                    DATA SUMMARY:
                    - Total Incidents Analyzed: {total_incidents}
                    - Selected Severities: {selected_severity}
                    - Breakdown by Severity: {severity_counts}
                    - Average Emergency Response Time: {avg_time:.1f} minutes
                    - Top 3 Community Area Hotspots (IDs): {top_areas}

                    Please structure your output using Markdown with these explicit sections:
                    1. 📖 **The Data Story (Context & Narrative)**
                    2. 📊 **Key Analytical Insights**
                    3. 💡 **Strategic Recommendations**
                    4. 📝 **Report Writing Tip for Students**
                    """

                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=report_prompt
                    )
                    
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"⚠️ Report Generation Failed: {e}")
