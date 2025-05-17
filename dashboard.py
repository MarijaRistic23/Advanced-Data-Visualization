import geopandas as gpd
from sqlalchemy import create_engine
import plotly.graph_objects as go
from shapely import wkt
import pandas as pd
import numpy as np
import configparser
import glob
import os
import base64
from pathlib import Path
import dash
from dash import dcc, html, Input, Output, callback, State
import dash_daq as daq
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import time
import math
import warnings

config = configparser.ConfigParser()
config.read('config.cfg')
db_name = "nyc_taxi_adv"
username = config.get('credentials', 'username')
password = config.get('credentials', 'password')

COLORS = {
    "background": '#153E67', #"#F5F5F5",   # Gris claro y suave
    "accent": "#FFD700",       # Amarillo taxi para títulos y detalles
    "text": "#222222",         # Texto principal
    "blue": "#0099C6",         # Azul para gráficos
    "purple": "#990099",       # Alternativa vibrante
    "red": "#DC3912"           # Rojo para alertas o extremos
}
CRS = "EPSG:4326"

db_url = f"postgresql://postgres:{password}@localhost:5432/nyc_taxi_adv"
engine = create_engine(db_url)

query = 'SELECT "LocationID", zone, borough, geom FROM taxi_zones;'
gdf = gpd.read_postgis(query, engine, geom_col="geom")

query = '''SELECT 
    borough,
	ST_AsText(ST_Centroid(ST_Union(geom))) AS centroid,
    ST_AsText(ST_Union(geom)) AS geom
FROM taxi_zones
GROUP BY borough'''

df_borough_centroid = pd.read_sql(query, engine)
df_borough_centroid["centroid"] = df_borough_centroid["centroid"].apply(wkt.loads)
df_borough_centroid["geom"] = df_borough_centroid["geom"].apply(wkt.loads)

gdf_borough = gpd.GeoDataFrame(df_borough_centroid, geometry="geom", crs=CRS)
gdf_borough["longitude"] = gdf_borough["centroid"].apply(lambda p: p.x)
gdf_borough["latitude"] = gdf_borough["centroid"].apply(lambda p: p.y)
gdf_borough.head()

borough_centroids = {row['borough']: (row['latitude'], row['longitude']) 
                    for _, row in gdf_borough.iterrows()}

query = '''SELECT 
    zone,
    ST_AsText(ST_Centroid(geom)) AS centroid
FROM 
    taxi_zones
WHERE 
    zone IN ('JFK Airport', 
            'LaGuardia Airport', 
            'Newark Airport')'''
df_airport_centroid = pd.read_sql(query, engine)
df_airport_centroid["centroid"] = df_airport_centroid["centroid"].apply(wkt.loads)

gdf_airport_centroid = gpd.GeoDataFrame(df_airport_centroid, geometry="centroid", crs=CRS) 
gdf_airport_centroid["longitude"] = gdf_airport_centroid["centroid"].x
gdf_airport_centroid["latitude"] = gdf_airport_centroid["centroid"].y

flowmap_title = None
heatmap_tittle = None
streamgraph_title = None

def fetch_airport(date, start_time, end_time, airport, selected_borough, toggle_value):
    where_condition = ''
    if toggle_value == 'dropoff':
        where_condition = f"dz.zone = '{airport}' AND pz.borough = '{selected_borough}'"
    else:
        where_condition = f"pz.zone = '{airport}' AND dz.borough = '{selected_borough}'"
        
    query = f"""
    SELECT 
        ST_AsText(ST_Centroid(pz.geom)) as centroid_pickup, 
        ST_AsText(ST_Centroid(dz.geom)) as centroid_dropoff,
        COUNT(*) AS trip_count
    FROM taxi t
    JOIN taxi_zones pz ON ST_Contains(pz.geom, t.geom_pickup)
    JOIN taxi_zones dz ON ST_Contains(dz.geom, t.geom_dropoff)
    WHERE {where_condition}
    AND t.tpep_pickup_datetime between '{date} {start_time}' and '{date} {end_time}'
    GROUP BY pz.geom, dz.geom
    limit 50;
    """
    df_taxi = pd.read_sql(query, engine)
    df_taxi["centroid_pickup"] = df_taxi["centroid_pickup"].apply(wkt.loads)
    df_taxi["centroid_dropoff"] = df_taxi["centroid_dropoff"].apply(wkt.loads)
    
    gdf_airport = gpd.GeoDataFrame(df_taxi, geometry="centroid_pickup", crs=CRS) 
    
    gdf_airport["pickup_longitude"] = gdf_airport["centroid_pickup"].x
    gdf_airport["pickup_latitude"] = gdf_airport["centroid_pickup"].y
    
    gdf_airport["dropoff_longitude"] = gdf_airport["centroid_dropoff"].apply(lambda p: p.x)
    gdf_airport["dropoff_latitude"] = gdf_airport["centroid_dropoff"].apply(lambda p: p.y)
    return gdf_airport

def fetch_airport_borough(date, start_time, end_time, airport, toggle_value):
    where_condition = ''
    group_by_condition = ''
    select_condition = ''
    if toggle_value == 'dropoff':
        where_condition = f"dz.zone = '{airport}'"
        group_by_condition = "pz.borough, dz.geom"
        select_condition = "pz.borough, ST_AsText(ST_Centroid(dz.geom)) as centroid_dropoff"
    else:
        where_condition = f"pz.zone = '{airport}'"
        group_by_condition = "dz.borough, pz.geom"
        select_condition = "dz.borough, ST_AsText(ST_Centroid(pz.geom)) as centroid_dropoff"
    query = f"""
    SELECT
        {select_condition},
        COUNT(*) AS trip_count
    FROM taxi t
    JOIN taxi_zones pz ON ST_Contains(pz.geom, t.geom_pickup)
    JOIN taxi_zones dz ON ST_Contains(dz.geom, t.geom_dropoff)
    WHERE {where_condition}
    AND t.tpep_pickup_datetime between '{date} {start_time}' and '{date} {end_time}'
    GROUP BY {group_by_condition}
    """
    #print("Upit: ", query)
    df_borough_airport = pd.read_sql(query, engine)
    df_borough_airport["centroid_dropoff"] = df_borough_airport["centroid_dropoff"].apply(wkt.loads)

    gdf_borough_airport = gpd.GeoDataFrame(df_borough_airport, geometry="centroid_dropoff", crs=CRS) 
    gdf_borough_airport["longitude"] = gdf_borough_airport["centroid_dropoff"].x
    gdf_borough_airport["latitude"] = gdf_borough_airport["centroid_dropoff"].y
    gdf_borough_airport.head()
    
    return gdf_borough_airport

def generate_large_arc(start, end, num_points=30, arc_height_factor=0.7):
    lat1, lon1 = start
    lat2, lon2 = end
    
    arc_lats, arc_lons = [], []
    
    for i in np.linspace(0, 1, num_points):
        # Pravimo interpoliranu tačku između start i end
        interpolated_lat = lat1 + i * (lat2 - lat1)
        interpolated_lon = lon1 + i * (lon2 - lon1)
        
        # Podižemo tačku "iznad" kako bi luk bio izraženiji
        height_adjustment = np.sin(i * np.pi) * arc_height_factor * abs(lat1 - lat2)
        interpolated_lat += height_adjustment  

        arc_lats.append(interpolated_lat)
        arc_lons.append(interpolated_lon)

    return arc_lats, arc_lons

# Putanja do vašeg icons foldera
icon_files = glob.glob('./mapbox-maki-8.2.0-0-g6ab50f3/mapbox-maki-6ab50f3/icons/*.svg')

# Ekstrakcija imena ikonica za Plotly
plotly_icons = []
for file_path in icon_files:
    # Dobijanje imena fajla bez ekstenzije
    filename = os.path.basename(file_path).replace('.svg', '')
    
    # Uklanjanje brojeva i crtica (npr. "airport-15" postaje "airport")
    icon_name = filename.split('-')[0]
    plotly_icons.append(icon_name)

# Uklanjanje duplikata
plotly_icons = list(set(plotly_icons))
#plotly_icons


svg_icon = Path('./mapbox-maki-8.2.0-0-g6ab50f3/mapbox-maki-6ab50f3/icons/arrow.svg').read_text()
encoded_image = base64.b64encode(svg_icon.encode('utf-8')).decode('utf-8')
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])#, suppress_callback_exceptions=True)

@app.server.errorhandler(404)
def page_not_found(e):
    return html.Div("Custom 404 Error Page"), 404

query_dates = "select distinct tpep_pickup_datetime::date from taxi order by tpep_pickup_datetime::date;"
available_dates = pd.read_sql(query_dates, engine)["tpep_pickup_datetime"].tolist()

marks={i: {'label': available_dates[i].strftime('%Y-%m-%d'), 'style': {'color': 'red'}} 
        for i in range(0, len(available_dates), 10)}  # Oznake na svakih 10 dana
last_idx = len(available_dates) - 1
if last_idx not in marks:
    # Dodaj poslednji datum kao posebnu oznaku
    last_date = available_dates[last_idx].strftime('%Y-%m-%d')
    marks[last_idx] = {
        'label': last_date,
        'style': {'color': 'red'}  # Opciono: istakni drugom bojom
    }
    
def plot_places(fig, gdf_airport, color):
    fig.add_trace(go.Scattermapbox(
        lon=gdf_airport["pickup_longitude"],
        lat=gdf_airport["pickup_latitude"],
        hoverinfo = 'text',
        text = gdf_airport["trip_count"],
        mode='markers',
        marker=dict(size=5, color=color),
        showlegend=False
    ))
    return fig

def get_arrow_angle(lon1, lat1, lon2, lat2):
    dx = lon2 - lon1
    dy = lat2 - lat1
    return math.degrees(math.atan2(dy, dx))

def plot_lines(fig, gdf_airport, color):
    for i in range(len(gdf_airport)):
        start_point = (gdf_airport['pickup_latitude'][i], gdf_airport['pickup_longitude'][i])
        end_point = (gdf_airport['dropoff_latitude'][i], gdf_airport['dropoff_longitude'][i]) 
        arc_lats, arc_lons = generate_large_arc(start_point, end_point, arc_height_factor= 0.5)
        fig.add_trace(
            go.Scattermapbox(
                lon = arc_lons,
                lat = arc_lats,
                mode = 'lines',
                line = dict(width = 1, color = color),
                opacity = float(gdf_airport['trip_count'][i]) / float(gdf_airport['trip_count'].max()),
                showlegend=False
            )
        )
        mid_idx = len(arc_lons) // 2  # Indeks srednje tačke
        fig.add_trace(
            go.Scattermapbox(
                lon=[arc_lons[mid_idx]],
                lat=[arc_lats[mid_idx]],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    symbol=f"image://data:image/svg+xml;base64,{encoded_image}",  # Strelica kao marker
                    angle=get_arrow_angle(arc_lons[mid_idx-1], arc_lats[mid_idx-1], 
                                    arc_lons[mid_idx+1], arc_lats[mid_idx+1]) # Funkcija za izračun ugla
                ),
                showlegend=False
            )
        )
        
        
    return fig

def plot_lines_borough(fig, gdf_airport_borough , color, toggle_value):
    global borough_centroids, gdf_airport_centroid

    for i, row in gdf_airport_borough.iterrows():
        borough_name = row['borough']
        centroid = gdf_borough[gdf_borough['borough'] == borough_name]
        start_point = (centroid.iloc[0]['latitude'], centroid.iloc[0]['longitude'])
        end_point = (gdf_airport_borough['latitude'][i], gdf_airport_borough['longitude'][i]) 
        arc_lats, arc_lons = generate_large_arc(start_point, end_point)
        for j in range(len(arc_lons)-1):
            if toggle_value == "dropoff":
                width = max(0.5, 7 * (j / (len(arc_lons)-1))) #linearno opada
            else:
                width = max(0.5, 9*(1 - j/10))
            fig.add_trace(
                go.Scattermapbox(
                    lon=[arc_lons[j], arc_lons[j+1]],
                    lat=[arc_lats[j], arc_lats[j+1]],
                    mode='lines',
                    line=dict(width=width, color=color),  # Width decreases
                    hoverinfo='none',
                    showlegend=False
                )
            )
    return fig

def show_polygons(fig, gdf_polygons, name ):
    for idx, row in gdf_polygons.iterrows():
        geom = row.geom
        if geom.geom_type == "MultiPolygon":
            for polygon in geom.geoms:
                lon, lat = polygon.exterior.xy
                text_values = [row[name]] * len(lon) 
                fig.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=list(lon),
                    lat=list(lat),
                    line=dict(width=0.8, color="black"),
                    opacity=0.5,
                    name=row[name],
                    text=text_values,
                    hoverinfo="name",
                    showlegend=False,
                    fill = "toself",
                    fillcolor="rgba(0,0,0,0)"
                ))
            if name == 'borough':
                centroid = geom.centroid
                fig.add_trace(go.Scattermapbox(
                    lat=[centroid.y],
                    lon=[centroid.x],
                    mode="markers",
                    marker=dict(size=10, color="red", opacity=0.5),  # nevidljiv marker koji prima klik
                    text=text_values,
                    name = row[name],
                    customdata=[[row[name]]],
                    hoverinfo="name",
                    showlegend=False
                ))
        elif geom.geom_type == "Polygon":
            lon, lat = geom.exterior.xy
            text_values = [row[name]] * len(lon)
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=list(lon),
                lat=list(lat),
                line=dict(width=0.8, color="black"),
                opacity=0.5,
                name=row[name],
                text=text_values,
                hoverinfo="name",
                showlegend=False,
                fill = "toself",
                fillcolor="rgba(0,0,0,0)",
                ))
            if name == 'borough':
                centroid = geom.centroid
                fig.add_trace(go.Scattermapbox(
                        lat=[centroid.y],
                        lon=[centroid.x],
                        mode="markers",
                        marker=dict(size=10, color="red", opacity=0.5),  # nevidljiv marker koji prima klik
                        text=text_values,
                        name = row[name],
                        customdata=[[row[name]]],
                        hoverinfo="name",
                        showlegend=False
                    ))
        else:
            print(f"Unsuported geom type: {geom.geom_type}")
    return fig

def update_map(selected_date, time_range, selected_borough, toggle_value):
    global gdf_airport_centroid
    print(selected_date)
    
    print("time range", time_range)
    start_hour, end_hour = time_range
    start_time = time(hour=int(start_hour), minute=int((start_hour % 1) * 60))
    end_time = time(hour=int(end_hour), minute=int((end_hour % 1) * 60))
    print(start_time, end_time)

    airports = ['JFK Airport', 'LaGuardia Airport','Newark Airport']
    color_list = [COLORS["red"], COLORS["purple"], COLORS["blue"]]
    
    fig = go.Figure()
        
    if selected_borough:
        zones_in_borough = gdf[gdf['borough'] == selected_borough]
        fig = show_polygons(fig, zones_in_borough, "zone")

        for i, airport in enumerate(airports):
            gdf_airport_zone = fetch_airport(selected_date, start_time, end_time, airport, selected_borough, toggle_value)
            fig = plot_places(fig, gdf_airport_zone, color_list[i])
            fig = plot_lines(fig, gdf_airport_zone, color_list[i])
        
    else:
        fig = show_polygons(fig, gdf_borough, "borough")
        for i, airport in enumerate(airports):
            print(f"Iscrava se za aerodrom {airport}")
            gdf_airport_borough = fetch_airport_borough(selected_date, start_time, end_time, airport, toggle_value)
            fig = plot_lines_borough(fig, gdf_airport_borough , color_list[i], toggle_value)

    fig.update_layout(
        uirevision="constant",
        mapbox=dict(
            accesstoken="pk.eyJ1IjoibWFyaWphcmlzdGljMjMiLCJhIjoiY21hZjZpeTc4MDIzZjJqcjFjcWhvMTRyNiJ9.V7dv1K-HL_i3asRs3aKmfg", 
            style="light",  #"light" "dark", "satellite", "streets"
            center=dict(lat=40.7128, lon=-74.0060),  # Centar NYC
            zoom=9.2,
            bearing=-20
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        plot_bgcolor="white",
    )
    
    return fig

def get_taxi_frequency_by_hour(selected_date, selected_borough, toggle_value):
    extra_condition = ''
    cond = 'dz.zone'
    if toggle_value == 'dropoff':
        
        if selected_borough:
            extra_condition = f"AND pz.borough = '{selected_borough}'"
            print("desio se ekstra condition")
    else:
        if selected_borough:
            extra_condition = f"AND dz.borough = '{selected_borough}'"
    query = f'''
    SELECT 
        {cond} as borough,
        EXTRACT(HOUR FROM t.tpep_pickup_datetime) AS hour,
        COUNT(*) AS drives
    FROM taxi t
    JOIN taxi_zones pz ON ST_Contains(pz.geom, t.geom_pickup)
    JOIN taxi_zones dz ON ST_Contains(dz.geom, t.geom_dropoff)
    WHERE {cond} IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport') {extra_condition}
    AND DATE(t.tpep_pickup_datetime) = '{selected_date}'
    GROUP BY {cond}, EXTRACT(HOUR FROM t.tpep_pickup_datetime)
    ORDER BY hour, borough;
    '''
    return pd.read_sql(query, engine)

def update_streamgraph(selected_date, selected_borough, toggle_value):
    
    df_taxi_frequency = get_taxi_frequency_by_hour(selected_date, selected_borough, toggle_value)
    df_taxi_frequency_pivot = df_taxi_frequency.pivot_table(
        index='hour', 
        columns='borough', 
        values='drives',
        fill_value=0  # Ako neki borough nema podatke za neki sat
    ).reset_index()

    stream_fig = go.Figure()
    boroughs_sorted = df_taxi_frequency_pivot.drop(columns='hour').sum().sort_values(ascending=False).index
    color_list = [COLORS["purple"], COLORS["red"], COLORS["blue"]]
    
    for i, borough in enumerate(boroughs_sorted):
        stream_fig.add_trace(go.Scatter(
            x=df_taxi_frequency_pivot['hour'],
            y=df_taxi_frequency_pivot[borough],
            name=borough,
            mode='lines',
            stackgroup='one',  # Ključno za streamgraph!
            line=dict(width=0.5, shape='spline', color=color_list[i]),  # Glatke krivine
            hoverinfo='x+y+name',
            hovertemplate=f'<b>{borough}</b><br>Hour: %{{x}}<br>Num of drives: %{{y}}<extra></extra>'
        ))

    stream_fig.update_layout(
        xaxis_title='Hour in day',
        yaxis_title='Num of drives',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(tickvals=list(range(24)), ticktext=[f'{h}:00' for h in range(24)]),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        legend=dict(
            orientation="v",          # Horizontal
            yanchor="bottom",
            y=1.1,                    # Por encima del gráfico
            xanchor="center"
        )
    )
    print("zavrseno pravljenje stream figure")
    return stream_fig

def get_tip_amount_borough(selected_date, time_range, toggle_value):
    start_hour, end_hour = time_range
    start_time = time(hour=int(start_hour), minute=int((start_hour % 1) * 60))
    end_time = time(hour=int(end_hour), minute=int((end_hour % 1) * 60))
    
    where_cond = ''
    cond = ''
    if toggle_value == 'dropoff':
        cond = 'pz.borough'
        where_cond = 'dz.zone'
    else:
        cond = 'dz.borough'
        where_cond = 'pz.zone'

    query = f'''
    SELECT 
        {cond} as borough,
        AVG(t.tip_amount) AS average_tip
    FROM taxi t
    JOIN taxi_zones pz ON ST_Contains(pz.geom, t.geom_pickup)
    JOIN taxi_zones dz ON ST_Contains(dz.geom, t.geom_dropoff)
    WHERE {where_cond} IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport')
    AND t.tpep_pickup_datetime between '{selected_date} {start_time}' and '{selected_date} {end_time}'
    GROUP BY {cond}
    '''

    df_pom = pd.read_sql(query, engine)
    return df_pom

def get_tip_amount(selected_date, time_range, selected_borough, toggle_value):
    start_hour, end_hour = time_range
    start_time = time(hour=int(start_hour), minute=int((start_hour % 1) * 60))
    end_time = time(hour=int(end_hour), minute=int((end_hour % 1) * 60))
    
    where_cond = 'dz.zone'
    cond = ''
    if toggle_value == 'dropoff':
        cond = 'pz'
        
    else:
        cond = 'dz'

    query = f'''
    SELECT 
        {cond}.zone as zone,
        ST_AsText({cond}.geom) as geometry,
        AVG(t.tip_amount) AS average_tip
    FROM taxi t
    JOIN taxi_zones pz ON ST_Contains(pz.geom, t.geom_pickup)
    JOIN taxi_zones dz ON ST_Contains(dz.geom, t.geom_dropoff)
    WHERE {where_cond} IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport')
    AND {cond}.borough = '{selected_borough}'
    AND t.tpep_pickup_datetime between '{selected_date} {start_time}' and '{selected_date} {end_time}'
    GROUP BY {cond}.zone, {cond}.geom 
    '''

    df_pom = pd.read_sql(query, engine)
    return df_pom

def update_heatmap(selected_date, time_range, selected_borough, toggle_value):
    if selected_borough:
        query = f'''SELECT zone, borough, ST_AsText(geom) AS geom
        FROM taxi_zones
        WHERE borough = '{selected_borough}';
        '''
        df_zone = pd.read_sql(query, engine)
        df_zone["geom"] = df_zone["geom"].apply(wkt.loads)
        df_tip_amount = get_tip_amount(selected_date, time_range, selected_borough, toggle_value)
        df_merge = df_zone.merge(df_tip_amount, on=['zone'], how='left', validate=None)
        gdf_tip_amount =  gpd.GeoDataFrame(df_merge, geometry="geom", crs=CRS)

        gdf_tip_amount['hover_text'] = gdf_tip_amount.apply(
            lambda x: f"{x['zone']}<br>Average tip: {x['average_tip']:.2f}" 
                if pd.notna(x['average_tip']) else f"{x['zone']}<br>No drives",
                axis=1
        )

        heatmap_tittle = f"Average tip amount by zone in {selected_borough}"

    else:
        global df_boroug_centroid
        df_tip_amount = get_tip_amount_borough(selected_date, time_range, toggle_value)
        df_merge = df_borough_centroid.merge(df_tip_amount, on='borough', how='left')
        gdf_tip_amount =  gpd.GeoDataFrame(df_merge, geometry="geom", crs=CRS)

        gdf_tip_amount['hover_text'] = gdf_tip_amount.apply(
            lambda x: f"{x['borough']}<br>Average tip: {x['average_tip']:.2f}"
                if pd.notna(x['average_tip']) else f"{x['borough']}<br>No drives",
                axis=1
        )
        heatmap_tittle = "Average tip amount by borough in NYC"

    heatmap_fig = go.Figure(go.Choroplethmapbox(
        geojson=gdf_tip_amount.geometry.__geo_interface__,
        locations=gdf_tip_amount.index,
        z=gdf_tip_amount['average_tip'].where(gdf_tip_amount['average_tip'].notna()),
        text=gdf_tip_amount['hover_text'],
        colorscale=[[0, COLORS['red']], [0.5, COLORS['purple']], [1, COLORS['blue']]],
        hoverinfo='text',
        marker_line_width=1,
        marker_opacity=0.8,
        marker_line_color='white',
        showscale=True
    ))
    heatmap_fig.add_trace(go.Choroplethmapbox(
    geojson=gdf_tip_amount.geometry.__geo_interface__,
    locations=gdf_tip_amount.index,
    z=np.where(gdf_tip_amount['average_tip'].isna(), 0, np.nan),
    text=gdf_tip_amount['hover_text'],
        hoverinfo = 'text',
    colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Potpuno providno
    marker_line_width=1,
    marker_opacity=1,
    marker_line_color='gray',  # Kontura za None vrednosti
    showscale=False,  # Sakrij legendu za ovaj sloj
    below=''  # Postavi ispod obojenih borough-a
    ))

    # Konfiguracija layout-a
    heatmap_fig.update_layout(
        uirevision=None,
        mapbox={
        "accesstoken": "pk.eyJ1IjoibWFyaWphcmlzdGljMjMiLCJhIjoiY21hZjZpeTc4MDIzZjJqcjFjcWhvMTRyNiJ9.V7dv1K-HL_i3asRs3aKmfg",
        "style": "light",
        "center": {"lat": 40.7259855, "lon": -74.0346485},
        "zoom": 8.9
        },
        title_x=0.5,
        margin={"r":0,"t":40,"l":0,"b":0},
        coloraxis_colorbar={
            'title': 'Iznos ($)',
            'thickness': 15,
            'len': 0.5
        }
    )
    return heatmap_fig

initial_index = len(available_dates) // 2  
initial_date = available_dates[initial_index] 

initial_time_range = [8.0, 12.0]
figure_initial = update_map(initial_date, initial_time_range, None, 'dropoff')
streamgraph_figure_initial = update_streamgraph(initial_date, None, 'dropoff')
heatmap_figure_initial = update_heatmap(initial_date, initial_time_range, None, 'dropoff')

white_square_shadow_box = '0 0 10px rgba(0,0,0,0.1)'

app.layout = html.Div([

    # Contenedor flotante unificado
    html.Div([

        # Subfila 1: Logo, toggle, date picker
        html.Div([
            # Logo a la izquierda
            html.Img(
                src="/assets/Nyctlc_logo.webp",
                style={
                    'height': '50px',
                    'marginRight': '20px'
                }
            ),

            html.Div(style={'flexGrow': '0.35'}),
            
            # Toggle antes del centro
            html.Div([
                html.Div(id='toggle-label', style={
                    'marginRight': '10px',
                    'fontFamily': 'Gotham',
                    'color': COLORS['background'],
                    'fontWeight': 'bold'
                }),

                daq.ToggleSwitch(
                    id='location-toggle',
                    value=False,
                    vertical=False,
                    color=COLORS['accent'],
                    className='custom-toggle',
                )
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginRight': '20px'
            }),

            # Separador flexible que empuja el DatePicker hacia la derecha del centro
            html.Div(style={'flexGrow': '0.25'}),

            # Date Picker después del centro
            dcc.DatePickerSingle(
                id='date-picker',
                className='custom-date',
                min_date_allowed=min(available_dates),
                max_date_allowed=max(available_dates),
                date=available_dates[initial_index],
                first_day_of_week=1,
                show_outside_days=False,
                display_format='DD-MM-YYYY'
            ),
        ], style={
            'position': 'sticky',
            'top': '0',
            'zIndex': '1000',
            'backgroundColor': '#D0E7F9',
            'boxShadow': white_square_shadow_box,
            'width': '100vw',
            'left': '0',
            'right': '0',
            'margin': '0 auto',
            'display': 'flex',
            'alignItems': 'center',
            'padding': '10px 30px'
        }),


        # Subfila 2: Slider
        html.Div([
            dcc.RangeSlider(
                id='time-range-slider',
                min=0,
                max=24,
                step=1,
                marks={i: f"{i}:00h" for i in range(0, 25, 4)},
                value=initial_time_range,
                tooltip={"placement": "bottom", "always_visible": False},
                className='horizontal-slider'
            )
        ], style={
            'padding': '10px 20px 10px 20px',
        }),

    ], style={
        'position': 'sticky',
        'top': '0',
        'zIndex': '1000',
        'backgroundColor': '#D0E7F9',
        'boxShadow': white_square_shadow_box,
        'margin': '0',
        'width': '100%',
    }),
    
    html.Div([
    # Contenedor de instrucciones
        html.Div([
            html.Button("X", id="close-instructions", n_clicks=0, style={
                'position': 'absolute',
                'top': '10px',
                'right': '10px',
                'background': 'none',
                'border': 'none',
                'fontSize': '20px',
                'cursor': 'pointer'
            }),
            html.H4("How to use this dashboard", style={
                'marginBottom': '10px',
                'color': COLORS['background'],
                'fontFamily': 'Gotham'
            }),
        html.P("""
        - Use the toggle to switch between trips from or to the airports.
        - Pick a specific day using the date picker.
        - Filter trips by hours with the slider.

        Visualizations:

        - Flowmap: Shows trips between airports and boroughs. Click to drill down or back.
        - Heatmap: Average tips per zone. Click to drill down or back.
        - Streamgraph: Trips over time. Hover for details.
        """, style={
                    'whiteSpace': 'pre-line',
                    'lineHeight': '1.5',
                    'color': '#153E67',
                    'fontFamily': 'Gotham'
                })
            ], id='instructions-box', style={
                'position': 'fixed',
                'top': '300px',
                'left': '50%',
                'right': '50%',
                'transform': 'translateX(-50%)',
                'backgroundColor': 'white',
                'padding': '20px 30px',
                'borderRadius': '12px',
                'boxShadow': white_square_shadow_box,
                'zIndex': '2000',
                'width': '90%',
                'maxWidth': '600px'
            })
        ]),
    
    # Fila 3 - Título
    html.Div([
        html.H1("Taxi trips in NYC", style={
            'textAlign': 'center',
            'color': COLORS['accent'],
            'fontFamily': 'Gotham',
            'margin': '20px 0'
        })
    ]),

    # Fila 4 - Flowmap
    html.Div([
        html.H3("Trips to/from airports", style={
            'textAlign': 'center',
            'color': COLORS['background'],
            'fontFamily': 'Gotham',
            'marginBottom': '10px'
        }),
        dcc.Graph(id='map-graph', figure=figure_initial)
    ], style={
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'padding': '20px',
        'margin': '0 20px 20px 20px',
        'boxShadow': white_square_shadow_box
    }),

    # Fila 5 - Heatmap + Streamgraph
    html.Div([
        html.Div([
            html.H3(heatmap_tittle, style={
                'textAlign': 'center',
                'color': COLORS['background'],
                'fontFamily': 'Gotham',
                'marginBottom': '10px'
            }),
            dcc.Graph(id='heatmap-graph', figure=heatmap_figure_initial)
        ], style={
            'width': '50%',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'marginRight': '10px',
            'boxShadow': white_square_shadow_box
        }),

        html.Div([
            html.H3("Number of taxi trips per airport", style={
                'textAlign': 'center',
                'color': COLORS['background'],
                'fontFamily': 'Gotham',
                'marginBottom': '10px'
            }),
            dcc.Graph(id='stream-graph', figure=streamgraph_figure_initial)
        ], style={
            'width': '50%',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': white_square_shadow_box
        }),
    ], style={'display': 'flex', 'margin': '0 20px 20px 20px'})

], style={
    'backgroundColor': COLORS['background'],
    'paddingBottom': '40px'
})


@app.callback(
    Output('instructions-box', 'style'),
    Input('close-instructions', 'n_clicks'),
    prevent_initial_call=True
)
def hide_instructions(n_clicks):
    if n_clicks:
        return {'display': 'none'}
    return dash.no_update


@app.callback(
    Output('toggle-label', 'children'),
    Input('location-toggle', 'value')
)
def update_toggle_label(toggle_value):
    return "From airport" if toggle_value is False else "To airport"

@callback(
    [Output("map-graph", "figure"), Output('time-range-slider', 'value'), Output('stream-graph', 'figure')],
    [Input("date-picker", "date"), Input('time-range-slider', 'value'), Input('map-graph', 'clickData'), Input('location-toggle', 'value' )],
    [State('stream-graph', 'figure')] 
)
def normalize_time_range(time_range):
    start, end = time_range
    if end >= 24:
        end = 23.9833
    return start, end

def get_triggered_id():
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    print(f"okinuo se dogadjaj {triggered}")
    return triggered

def normalize_toggle_value(toggle_value):
    return 'pickup' if toggle_value else 'dropoff'

def extract_selected_borough(click_data):
    if not click_data or 'points' not in click_data:
        return None

    print("clickData:", click_data)  # Debug
    for point in click_data['points']:
        borough = point.get('text')
        if borough and borough in gdf_borough['borough'].values:
            print("clicked okrug:", borough)
            return borough
        else:
            print("Nisam nasao customdata")
    return None

def update_streamgraph_conditionally(triggered, selected_date, selected_borough, toggle_value, curr_stream_figure):
    if (
        triggered == 'date-picker.date'
        or selected_borough
        or triggered == 'location-toggle.value'
        or (triggered == 'map-graph.clickData' and selected_borough is None)
    ):
        return update_streamgraph(selected_date, selected_borough, toggle_value)
    return curr_stream_figure

def combined_callback(selected_date, time_range, click_data, toggle_value, curr_stream_figure):
    start, end = normalize_time_range(time_range)
    triggered = get_triggered_id()
    toggle_value = normalize_toggle_value(toggle_value)
    selected_borough = extract_selected_borough(click_data)

    fig = update_map(selected_date, [start, end], selected_borough, toggle_value)
    streamgraph_fig = update_streamgraph_conditionally(
        triggered, selected_date, selected_borough, toggle_value, curr_stream_figure
    )

    return fig, [start, end] if end >= 24 else dash.no_update, streamgraph_fig


if __name__ == "__main__":
    app.run(debug=False)