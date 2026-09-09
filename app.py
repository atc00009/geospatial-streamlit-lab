import streamlit as st  # MUST BE AT THE VERY TOP OF APP.PY
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np
from google import genai

# 1. PAGE SETUP
st.set_page_config(
    page_title="Chicago Geospatial Analytics & AI Lab",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Chicago Spatial Analytics & AI Storytelling Lab")
st.markdown("Explore spatial point vectors, choropleth polygon aggregations, and interact with an AI Spatial Assistant for insights and report guidance.")

# 2. INITIALIZE GEMINI CLIENT SAFELY
gemini_key = st.secrets.get("GEMINI_API_KEY", None)
client = genai.Client(api_key=gemini_key) if gemini_key else None

# 3. GEOSPATIAL FOUNDATIONS (EDUCATIONAL CONCEPT MODULE)
with st.expander("📚 **Geospatial Foundations: Why Spatial Data, Raster, & TIFF Files Matter**", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Why Use Geospatial Data?**
        * **Beyond Static Summary Tables:** Summary numbers tell you *how much*, but spatial mapping reveals *where* events cluster and how they relate across urban space.
        * **Latitude ($\phi$) & Longitude ($\lambda$):** Spherical coordinates used to pinpoint exact positions on Earth. Chicago is centered near $\sim 41.8781^\circ\text{ N}, -87.6298^\circ\text{ W}$.
        * **Layered Context:** Spatial frameworks let you layer discrete events (points) over administrative boundaries (neighborhood polygons) and environmental basemaps.
        """)
    with col_b:
        st.markdown("""
        **Data Formats: Vector vs. Raster (TIFF & GeoTIFF)**
        * **Vector Data (Point, Line, Polygon):** Represents discrete features using explicit coordinates.
          * *Points:* Incident locations (`Latitude`, `Longitude`).
          * *Polygons:* Neighborhood boundary shapes saved in **GeoJSON** format.
        * **Raster Data (Pixel Grids & TIFF / `.tif`):** Represents continuous surfaces where every square cell holds a value (e.g., satellite imagery, urban heat islands, elevation models).
        * **Why GeoTIFF (.tif)?** Used when data varies continuously across space rather than stopping cleanly at administrative borders.
        """)

# 4. CACHED GEOSPATIAL DATA GENERATOR
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

# 5. INTERACTIVE SIDEBAR CONTROL PANEL
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

# 6. METRIC KPIS
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Visible Incidents", f"{len(filtered_df):,}")
kpi2.metric("Average Response Time", f"{filtered_df['Response_Time_Min'].mean():.1f} min" if not filtered_df.empty else "N/A")
kpi3.metric("Critical Incidents Ratio", f"{(filtered_df['Severity'] == 'Critical').mean() * 100:.1f}%" if not filtered_df.empty else "N/A")

st.divider()

# 7. VISUALIZATION CANVAS (MAPS + HISTOGRAM CHARTS)
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

# 8. AI ASSISTANT, STORYTELLING & REPORT GENERATOR (GEMINI POWERED)
st.divider()
st.subheader("🤖 AI Spatial Assistant & Report Builder")

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
                    response_stream = client.models.generate_content_stream(
                        model='gemini-3.6-flash',
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
                    1. 📖 **The Data Story (Context & Narrative)**: Set the scene in Chicago for city managers and policymakers.
                    2. 📊 **Key Analytical Insights**: Interpret the numbers, hotspots, and response times. Explain spatial risks clearly.
                    3. 💡 **Strategic Recommendations**: Provide 3 concrete, actionable recommendations for city resource allocation.
                    4. 📝 **Report Writing Tip for Students**: Explain briefly why this structure works well in technical academic writing.
                    """

                    response = client.models.generate_content(
                        model='gemini-3.6-flash',
                        contents=report_prompt
                    )
                    
                    st.markdown(response.text)

                except Exception as e:
                    st.error(f"⚠️ Report Generation Failed: {e}")

# TAB 3: GUIDED STORY & DECISION WORKSHOP
with tab3:
    st.markdown("""
    ### 🧠 Data Story & Decision Workshop
    **Goal:** build your analysis *from the map and chart alone* — no raw table, no AI report yet.
    Write your own story first. You'll compare it against the AI-generated report afterward.
    """)

    with st.expander("🔎 What to look for in each view (read this before you start)", expanded=False):
        st.markdown("""
        **Scatter Point Vector Layer** — each dot is one incident (hover for ID & response time).
        Look for spatial *clustering* (do dots bunch up in specific neighborhoods?) and the
        *color × size interaction* (large red dots = Critical severity **and** slow response —
        a flag worth naming explicitly).

        **Aggregated Choropleth Polygon Map** — each shaded polygon is a community area, colored
        by incident count (hover for area ID & count). This is the unit a city planner actually
        allocates budget against. Check whether the areas that light up here match the clusters
        you saw on the scatter map — agreement strengthens your case, disagreement is worth
        explaining.

        **Interactive Folium Map** — click individual markers (red = Critical) to drill into
        specific cases and spot-check outliers the aggregate views smoothed over.

        **Response Time Histogram** — compare the *shape and tail* of each severity color. A
        right-shifted tail for Critical vs. Low is strong, citable evidence that response
        prioritization isn't matching severity.
        """)

    st.divider()
    st.markdown("#### ✍️ Your Analysis")

    if "student_story" not in st.session_state:
        st.session_state.student_story = {"observations": "", "hypothesis": "", "decision": ""}

    st.session_state.student_story["observations"] = st.text_area(
        "1️⃣ Observations — what patterns do you see on the active map and histogram right now?",
        value=st.session_state.student_story["observations"],
        placeholder="e.g. Incidents cluster near community areas 24 and 33; Critical incidents show a longer response-time tail than Low..."
    )

    st.session_state.student_story["hypothesis"] = st.text_area(
        "2️⃣ Hypothesis — why might this pattern exist?",
        value=st.session_state.student_story["hypothesis"],
        placeholder="e.g. These areas may be farther from existing stations, or have higher call volume overall..."
    )

    st.session_state.student_story["decision"] = st.text_area(
        "3️⃣ Decision — what would you recommend a city manager do, based only on what you've observed?",
        value=st.session_state.student_story["decision"],
        placeholder="e.g. Prioritize a new response unit near area 24; audit dispatch times for Critical calls..."
    )

    st.divider()
    st.info(
        "✅ Once your story is written, generate the AI report in the **✨ Auto-Generate AI Report** tab "
        "and compare: What did it notice that you missed? What did you catch that it didn't? "
        "Where do your recommendations agree or disagree?"
    )

# TAB 4: RAW DATASET (verification, not a shortcut)
with tab4:
    st.caption(
        "💡 Use this to **verify** the story you already wrote — not to skip straight to it. "
        "If you're seeing this before filling out the Workshop tab, go back and write your "
        "observations from the map first."
    )
    st.dataframe(filtered_df, use_container_width=True)
