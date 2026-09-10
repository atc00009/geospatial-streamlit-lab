import streamlit as st  # MUST BE AT THE VERY TOP OF APP.PY
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
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

# 2b. RESILIENT MODEL FALLBACK LIST
# Ordered from most-preferred to least-preferred. If Google retires the first
# model, the code automatically tries the next one instead of erroring out.
FLASH_MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

def generate_with_fallback(client, contents, stream=False):
    """
    Try each model in FLASH_MODEL_CANDIDATES in order until one works.
    Returns (response_or_stream, model_name_used).
    Raises the last error if every candidate fails.
    """
    last_error = None
    for model_name in FLASH_MODEL_CANDIDATES:
        try:
            if stream:
                result = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents
                )
            else:
                result = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
            return result, model_name
        except Exception as e:
            last_error = e
            continue
    raise last_error

# 3. INTERACTIVE GEOSPATIAL DATA FORMAT EXPLORER
st.subheader("📚 Interactive Geospatial Data Explorer")
st.caption("Click a data format below to dynamically load its corresponding interactive visual example and theoretical explanation.")

# Interactive Selector for Data Formats
selected_format = st.radio(
    "Select Data Format to Explore:",
    [
        "📍 Vector Points (Incidents)",
        "🗺️ Vector Polygons (GeoJSON Boundaries)",
        "🖼️ Raster Data (GeoTIFF Pixel Matrix)",
        "🌐 Coordinates (Latitude & Longitude)"
    ],
    horizontal=True
)

expl_col, vis_col = st.columns([1, 1])

if selected_format == "📍 Vector Points (Incidents)":
    with expl_col:
        st.markdown("### 📍 Vector Point Data")
        st.markdown("""
        * **Structure:** Discrete $X, Y$ coordinate pairs ($\text{Longitude}, \text{Latitude}$) stored with attributes.
        * **When to Use:** Representing distinct, localized events or objects like incident locations, fire hydrants, or transit stops.
        * **Why it Matters:** Allows exact spatial point pattern analysis, distance measurement, and nearest-neighbor calculations.
        """)
        st.info("💡 **Interpretation:** Notice how each point represents a distinct individual event with specific properties (Severity, Response Time).")
    
    with vis_col:
        sample_df = pd.DataFrame({
            "Lat": [41.8781, 41.8850, 41.8700, 41.8900],
            "Lon": [-87.6298, -87.6350, -87.6200, -87.6400],
            "Incident": ["INC-1", "INC-2", "INC-3", "INC-4"],
            "Severity": ["Critical", "High", "Medium", "Low"]
        })
        fig_pt = px.scatter_map(
            sample_df, lat="Lat", lon="Lon", color="Severity", hover_name="Incident",
            zoom=11.5, center={"lat": 41.8781, "lon": -87.6298},
            color_discrete_map={"Low": "green", "Medium": "blue", "High": "orange", "Critical": "red"},
            map_style="carto-positron", title="Interactive Vector Points View"
        )
        fig_pt.update_layout(margin={"r":0, "t":30, "l":0, "b":0}, height=300)
        st.plotly_chart(fig_pt, use_container_width=True)

elif selected_format == "🗺️ Vector Polygons (GeoJSON Boundaries)":
    with expl_col:
        st.markdown("### 🗺️ Vector Polygon Data (GeoJSON)")
        st.markdown("""
        * **Structure:** Closed loops of connected coordinate pairs defining bounded geographical areas.
        * **When to Use:** Representing administrative boundaries, community areas, voting districts, or census tracts.
        * **Why it Matters:** Essential for choropleth mapping and aggregating point counts into policy-relevant administrative units.
        """)
        st.info("💡 **Interpretation:** Polygons group spatial information inside defined administrative zones so decision-makers can allocate localized budgets.")

    with vis_col:
        poly_lat = [41.875, 41.885, 41.885, 41.875, 41.875]
        poly_lon = [-87.635, -87.635, -87.620, -87.620, -87.635]
        fig_poly = px.line_map(
            lat=poly_lat, lon=poly_lon,
            zoom=12, center={"lat": 41.880, "lon": -87.627},
            map_style="carto-positron", title="Interactive GeoJSON Boundary Polygon View"
        )
        fig_poly.update_layout(margin={"r":0, "t":30, "l":0, "b":0}, height=300)
        st.plotly_chart(fig_poly, use_container_width=True)

elif selected_format == "🖼️ Raster Data (GeoTIFF Pixel Matrix)":
    with expl_col:
        st.markdown("### 🖼️ Raster Grid Data (.TIFF / .GeoTIFF)")
        st.markdown("""
        * **Structure:** Continuous grid matrix made of square pixels where every cell holds a numeric surface value.
        * **When to Use:** Satellite imagery, elevation models (DEM), heat maps, or urban heat island surface measurements.
        * **Why GeoTIFF (.tif)?** Used when phenomena vary continuously across space without stopping cleanly at neighborhood borders.
        """)
        st.info("💡 **Interpretation:** Heatmaps and satellite images don't have borders; they store continuous cell values across a spatial grid.")

    with vis_col:
        x_grid, y_grid = np.meshgrid(np.linspace(-2, 2, 35), np.linspace(-2, 2, 35))
        z_raster = np.exp(-x_grid**2 - y_grid**2)
        fig_rast = px.imshow(
            z_raster, color_continuous_scale="Hot",
            title="Continuous GeoTIFF Raster Grid Heatmap Simulation"
        )
        fig_rast.update_layout(margin={"r":0, "t":30, "l":0, "b":0}, height=300)
        st.plotly_chart(fig_rast, use_container_width=True)

else:
    with expl_col:
        st.markdown("### 🌐 Coordinates (Latitude & Longitude)")
        st.markdown("""
        * **Latitude ($\mathbf{\phi}$):** Measures angular distance North/South of the Equator ($0^\circ$). Chicago is centered at $\sim 41.8781^\circ\text{ N}$.
        * **Longitude ($\mathbf{\lambda}$):** Measures angular distance East/West of the Prime Meridian ($0^\circ$). Chicago is centered at $\sim -87.6298^\circ\text{ W}$.
        * **CRS (EPSG:4326):** Standard global datum (WGS84) used by GPS systems to project spherical coordinates onto flat screens.
        """)
        st.info("💡 **Interpretation:** Every point on Earth requires both a latitude and longitude value to be uniquely identified.")

    with vis_col:
        fig_coord = go.Figure(go.Scattergeo(
            lat=[41.8781], lon=[-87.6298],
            mode='markers+text',
            text=["Chicago (41.8781° N, -87.6298° W)"],
            textposition="top center",
            marker=dict(size=12, color='red')
        ))
        fig_coord.update_layout(
            title="Coordinate Spatial Marker",
            geo=dict(projection_scale=3, center=dict(lat=41.8781, lon=-87.6298), showland=True),
            margin={"r":0, "t":30, "l":0, "b":0}, height=300
        )
        st.plotly_chart(fig_coord, use_container_width=True)

st.divider()

# 4. CACHED GEOSPATIAL DATA GENERATOR
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 1200
    
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat with AI Assistant", 
    "✨ Auto-Generate AI Report", 
    "🧠 Story & Decision Workshop", 
    "🔍 View Raw Dataset",
    "🆚 Compare: You vs. AI"
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
                    response_stream, used_model = generate_with_fallback(
                        client,
                        contents=f"System Context: {system_context}\n\nUser Question: {user_prompt}",
                        stream=True
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

                    response, used_model = generate_with_fallback(
                        client,
                        contents=report_prompt,
                        stream=False
                    )

                    st.session_state.ai_report = response.text

                except Exception as e:
                    st.error(f"⚠️ Report Generation Failed: {e}")

        if st.session_state.get("ai_report"):
            st.markdown(st.session_state.ai_report)

# TAB 3: GUIDED STORY & DECISION WORKSHOP
with tab3:
    st.markdown("""
    ### 🧠 Data Story & Decision Workshop
    **Goal:** Build your analysis *from the map and chart alone* — write your own story first before reviewing the AI report.
    """)

    with st.expander("🔎 What to look for in each view (Read before writing)", expanded=False):
        st.markdown("""
        * **Scatter Point Vector Layer:** Look for spatial clustering and color/size interactions (e.g., large red markers indicate Critical severity with long response delays).
        * **Aggregated Choropleth Polygon Map:** Observe aggregated totals per neighborhood area. Compare whether polygon totals align with scatter density.
        * **Response Time Histogram:** Check right-shifted distribution tails across severity levels to evaluate resource distribution efficiency.
        """)

    st.divider()
    st.markdown("#### ✍️ Your Analysis")

    if "student_story" not in st.session_state:
        st.session_state.student_story = {"observations": "", "hypothesis": "", "decision": ""}

    st.session_state.student_story["observations"] = st.text_area(
        "1️⃣ Observations — what patterns do you see on the active map and histogram right now?",
        value=st.session_state.student_story["observations"],
        placeholder="e.g., Incidents cluster heavily near area 24; Critical incidents exhibit longer response tails..."
    )

    st.session_state.student_story["hypothesis"] = st.text_area(
        "2️⃣ Hypothesis — why might this pattern exist?",
        value=st.session_state.student_story["hypothesis"],
        placeholder="e.g., Distance from existing dispatch stations or higher call volumes in specific zones..."
    )

    st.session_state.student_story["decision"] = st.text_area(
        "3️⃣ Decision — what would you recommend a city manager do?",
        value=st.session_state.student_story["decision"],
        placeholder="e.g., Position emergency units closer to identified high-density clusters..."
    )

# TAB 4: RAW DATASET
with tab4:
    st.caption("💡 Use raw tabular data to verify observations made on the spatial maps.")
    st.dataframe(filtered_df, use_container_width=True)

# TAB 5: COMPARE — STUDENT STORY VS. AI REPORT
with tab5:
    st.markdown("### 🆚 Compare Your Story vs. the AI Report")

    story = st.session_state.get("student_story", {"observations": "", "hypothesis": "", "decision": ""})
    ai_report = st.session_state.get("ai_report", "")

    story_written = any(story.values())
    report_generated = bool(ai_report)

    if not story_written or not report_generated:
        missing = []
        if not story_written:
            missing.append("your notes in **🧠 Story & Decision Workshop**")
        if not report_generated:
            missing.append("a report from **✨ Auto-Generate AI Report**")
        st.warning("Fill in " + " and ".join(missing) + " to unlock the comparison.")
    else:
        col_you, col_ai = st.columns(2)

        with col_you:
            st.markdown("#### 🧑‍🎓 Your Story")
            st.markdown(f"**Observations**\n\n{story['observations']}")
            st.markdown(f"**Hypothesis**\n\n{story['hypothesis']}")
            st.markdown(f"**Decision**\n\n{story['decision']}")

        with col_ai:
            st.markdown("#### 🤖 AI Report")
            st.markdown(ai_report)

        st.divider()
        st.markdown("#### 📝 Reflection")
        st.session_state.setdefault("reflection_notes", "")
        st.session_state.reflection_notes = st.text_area(
            "Where did your analysis align or differ from the AI report?",
            value=st.session_state.reflection_notes,
            height=150
        )
