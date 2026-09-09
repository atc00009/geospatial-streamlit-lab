import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np
from openai import OpenAI

# 1. PAGE SETUP
st.set_page_config(
    page_title="Chicago Geospatial Analytics & AI Lab",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Chicago Spatial Analytics & AI Storytelling Lab")
st.markdown("Explore spatial point vectors, choropleth polygon aggregations, and interact with an AI Spatial Assistant for insights and report guidance.")

# Safe initialization of OpenAI Client using Streamlit Secrets
openai_key = st.secrets.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=openai_key) if openai_key else None

# 2. CACHED GEOSPATIAL DATA GENERATOR
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1200
    
    # Generate coordinates around Chicago center
    lats = np.random.normal(loc=41.8781, scale=0.06, size=n)
    lons = np.random.normal(loc=-87.6298, scale=0.05, size=n)
    community_ids = np.random.choice(range(1, 78), size=n)
    severity_levels = np.random.choice(["Low", "Medium", "High", "Critical"], size=n, p=[0.4, 0.3, 0.2, 0.1])
    response_times = np.random.exponential(scale=15, size=n).round(1)

    df = pd.DataFrame({
        "Incident_ID": [f"INC-{10000+i}" for i in range(n)],
        "Latitude": lats,
        "Longitude": lons,
        "Community_Area": community_ids,
        "Severity": severity_levels,
        "Response_Time_Min": response_times
    })

    # Convert to GeoPandas GeoDataFrame (Point Vector Object)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    )
    return df, gdf

df_incidents, gdf_incidents = load_data()

# 3. INTERACTIVE SIDEBAR CONTROL PANEL
st.sidebar.header("🎛️ Student Control Panel")

selected_severity = st.sidebar.multiselect(
    "Filter Incident Severity:",
    options=["Low", "Medium", "High", "Critical"],
    default=["High", "Critical"]
)

view_mode = st.sidebar.radio(
    "Select Visualization Method:",
    ["Scatter Point Vector Layer", "Aggregated Choropleth Polygon Map", "Interactive Folium Map"]
)

# Filter Data based on user inputs
filtered_df = df_incidents[df_incidents["Severity"].isin(selected_severity)]

# 4. METRIC KPIS
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Visible Incidents", f"{len(filtered_df):,}")
kpi2.metric("Average Response Time", f"{filtered_df['Response_Time_Min'].mean():.1f} min" if not filtered_df.empty else "N/A")
kpi3.metric("Critical Incidents Ratio", f"{(filtered_df['Severity'] == 'Critical').mean() * 100:.1f}%" if not filtered_df.empty else "N/A")

st.divider()

# 5. VISUALIZATION CANVAS
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader(f"Geospatial View — {view_mode}")
    
    if filtered_df.empty:
        st.warning("No data points available for the selected filters.")
    elif view_mode == "Scatter Point Vector Layer":
        fig_point = px.scatter_map(
            filtered_df,
            lat="Latitude",
            lon="Longitude",
            color="Severity",
            size="Response_Time_Min",
            color_discrete_map={"Low": "green", "Medium": "blue", "High": "orange", "Critical": "red"},
            zoom=9.5,
            center={"lat": 41.8781, "lon": -87.6298},
            map_style="carto-positron",
            hover_data=["Incident_ID", "Response_Time_Min"]
        )
        fig_point.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
        st.plotly_chart(fig_point, use_container_width=True)

    elif view_mode == "Aggregated Choropleth Polygon Map":
        counts = filtered_df.groupby("Community_Area").size().reset_index(name="Incident_Count")
        fig_choro = px.choropleth_map(
            counts,
            geojson="https://raw.githubusercontent.com/chicago/chicago-gis/master/Community_Areas.geojson",
            featureidkey="properties.area_num_1",
            locations="Community_Area",
            color="Incident_Count",
            color_continuous_scale="Reds",
            zoom=9.5,
            center={"lat": 41.8781, "lon": -87.6298},
            map_style="carto-positron"
        )
        fig_choro.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
        st.plotly_chart(fig_choro, use_container_width=True)

    else:
        folium_map = folium.Map(location=[41.8781, -87.6298], zoom_start=10, tiles="CartoDB positron")
        for _, row in filtered_df.head(100).iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=5,
                color="red" if row["Severity"] == "Critical" else "blue",
                fill=True,
                popup=f"ID: {row['Incident_ID']} | Time: {row['Response_Time_Min']} min"
            ).add_to(folium_map)
        st_folium(folium_map, width="100%", height=500)

with right_col:
    st.subheader("Statistical Analysis")
    if not filtered_df.empty:
        fig_hist = px.histogram(
            filtered_df, 
            x="Response_Time_Min", 
            color="Severity",
            title="Response Time Distribution",
            color_discrete_map={"Low": "green", "Medium": "blue", "High": "orange", "Critical": "red"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# 6. AI ASSISTANT, STORYTELLING & REPORT GENERATOR
st.divider()
st.subheader("🤖 AI Spatial Assistant & Report Builder")

tab1, tab2, tab3, tab4 = st.tabs([
    " Chat with AI Assistant", 
    " Auto-Generate AI Report", 
    " Lab Reflection Questions", 
    " View Raw Dataset"
])

# TAB 1: INTERACTIVE CHAT INTERFACE
with tab1:
    st.markdown("Ask the AI assistant any questions about analyzing spatial data, interpreting map hotspots, or structuring your lab report.")
    
    if not client:
        st.warning(" OpenAI API key not detected. Please add `OPENAI_API_KEY` to your Streamlit secrets to enable live chat.")
    else:
        # Initialize chat history in session state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [
                {
                    "role": "assistant", 
                    "content": "Hello! I am your Spatial Data Science AI assistant. Ask me how to interpret your active map data, analyze geographic hotspots, or write policy recommendations!"
                }
            ]

        # Display conversation history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input Box
        if user_prompt := st.chat_input("Ask a question about your spatial analysis or report structure..."):
            st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Generate streaming response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                system_context = f"""
                You are an expert Spatial Analytics Teaching Assistant. 
                The student is currently analyzing Chicago incident data with these parameters:
                - Selected Severities: {selected_severity}
                - Total Filtered Incidents: {len(filtered_df)}
                - Average Response Time: {filtered_df['Response_Time_Min'].mean():.1f} min if dataset not empty else 0
                - Active Map Representation: {view_mode}

                Help the student understand spatial concepts (vector vs. polygon/choropleth, MAUP, point clusters), 
                and guide them on how to write professional analytical insights and policy recommendations.
                """
                
                messages_for_api = [{"role": "system", "content": system_context}] + [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.chat_messages
                ]

                response_stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_for_api,
                    stream=True
                )

                full_response = ""
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            
            st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

# TAB 2: AUTOMATED REPORT GENERATOR
with tab2:
    st.markdown("Click below to generate a structured academic report based on your currently filtered dataset.")
    
    if not client:
        st.warning("⚠️ OpenAI API key not detected. Please add `OPENAI_API_KEY` to your Streamlit secrets.")
    else:
        if st.button(" Generate Spatial Story & Insights Report"):
            with st.spinner("Analyzing spatial patterns and drafting story..."):
                avg_time = filtered_df['Response_Time_Min'].mean() if not filtered_df.empty else 0
                total_incidents = len(filtered_df)
                top_areas = filtered_df['Community_Area'].value_counts().head(3).to_dict() if not filtered_df.empty else {}
                severity_counts = filtered_df['Severity'].value_counts().to_dict() if not filtered_df.empty else {}

                prompt = f"""
                You are a Senior Spatial Data Science Professor. Analyze the following Chicago geospatial incident data summary and write an engaging data story and report section for students.

                DATA SUMMARY:
                - Total Incidents Analyzed: {total_incidents}
                - Selected Severities: {selected_severity}
                - Breakdown by Severity: {severity_counts}
                - Average Emergency Response Time: {avg_time:.1f} minutes
                - Top 3 Community Area Hotspots (IDs): {top_areas}

                Please structure your output using Markdown with these explicit sections:
                1.  **The Data Story (Context & Narrative)**: Set the scene in Chicago for city managers and policymakers.
                2.  **Key Analytical Insights**: Interpret the numbers, hotspots, and response times. Explain spatial risks clearly.
                3.  **Strategic Recommendations**: Provide 3 concrete, actionable recommendations for city resource allocation.
                4.  **Report Writing Tip for Students**: Explain briefly why this structure works well in technical academic writing.
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                
                ai_report = response.choices[0].message.content
                st.markdown(ai_report)

# TAB 3: GUIDED REFLECTION QUESTIONS
with tab3:
    st.markdown("""
    **Guided Questions for Manual Student Analysis:**
    1. **Spatial Aggregation (MAUP):** Toggle between *Scatter Point Vector* and *Choropleth Polygon Map*. Does polygon aggregation conceal localized spatial hotspots?
    2. **Resource Distribution:** Analyze the Response Time histogram across severity levels. Do high-severity incidents exhibit shorter response times?
    3. **Spatial Bias:** How might non-random reporting patterns introduce bias when analyzing city incident distributions?
    """)

# TAB 4: RAW DATASET
with tab4:
    st.dataframe(filtered_df, use_container_width=True)
