import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import st_folium
import numpy as np

# 1. PAGE SETUP
st.set_page_config(
    page_title="Chicago Geospatial Lab",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ Interactive Spatial Analytics & Student Lab")
st.markdown("Explore spatial point vector representations, choropleth polygon aggregations, and performance metrics.")

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

# 3. INTERACTIVE CONTROL PANEL
st.sidebar.header("🎛️ Student Controls")

selected_severity = st.sidebar.multiselect(
    "Filter Incident Severity:",
    options=["Low", "Medium", "High", "Critical"],
    default=["High", "Critical"]
)

view_mode = st.sidebar.radio(
    "Select Visualization Method:",
    ["Scatter Point Vector Layer", "Aggregated Choropleth Polygon Map", "Interactive Folium Map"]
)

# Filter Data based on inputs
filtered_df = df_incidents[df_incidents["Severity"].isin(selected_severity)]

# 4. METRIC KPIS
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Visible Incidents", f"{len(filtered_df):,}")
kpi2.metric("Average Response Time", f"{filtered_df['Response_Time_Min'].mean():.1f} min")
kpi3.metric("Critical Incidents Ratio", f"{(filtered_df['Severity'] == 'Critical').mean() * 100:.1f}%")

st.divider()

# 5. VISUALIZATION CANVAS
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader(f"Geospatial View — {view_mode}")
    
    if view_mode == "Scatter Point Vector Layer":
        # Updated method for modern Plotly versions
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
        # Updated method for modern Plotly versions
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
    fig_hist = px.histogram(
        filtered_df, 
        x="Response_Time_Min", 
        color="Severity",
        title="Response Time Distribution",
        color_discrete_map={"Low": "green", "Medium": "blue", "High": "orange", "Critical": "red"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# 6. GUIDED STUDENT LAB EXERCISES
st.divider()
st.subheader("🧠 Student Hypothesis Testing")

tab1, tab2 = st.tabs(["Analysis Exercises", "Raw Dataset View"])

with tab1:
    st.markdown("""
    **Guided Lab Questions:**
    1. **Spatial Aggregation (MAUP):** Toggle between *Scatter Point Vector* and *Choropleth Polygon Map*. Does polygon aggregation conceal localized spatial hotspots?
    2. **Resource Distribution:** Analyze the Response Time histogram across severity levels. Do high-severity incidents exhibit shorter response times?
    3. **Spatial Bias:** How might non-random reporting patterns introduce bias when analyzing city incident distributions?
    """)

with tab2:
    st.dataframe(filtered_df, use_container_width=True)
