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
    "blue": "#153E67",         # Azul para gráficos
    "purple": "#28a745",       # Alternativa vibrante
    "red": "#FFD700"#"#DC3912"           # Rojo para alertas o extremos
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
    select_condition = ''
    if toggle_value == 'dropoff':
        where_condition = f"dz.zone = '{airport}' AND pz.borough = '{selected_borough}'"
        select_condition = 'pz.zone'
    else:
        where_condition = f"pz.zone = '{airport}' AND dz.borough = '{selected_borough}'"
        select_condition = 'dz.zone'
        
    query = f"""
    SELECT 
        {select_condition} as zone,
        ST_AsText(ST_Centroid(pz.geom)) as centroid_pickup, 
        ST_AsText(ST_Centroid(dz.geom)) as centroid_dropoff,
        COUNT(*) AS trip_count
    FROM taxi t
    JOIN taxi_zones pz ON ST_Contains(pz.geom, t.geom_pickup)
    JOIN taxi_zones dz ON ST_Contains(dz.geom, t.geom_dropoff)
    WHERE {where_condition}
    AND t.tpep_pickup_datetime between '{date} {start_time}' and '{date} {end_time}'
    GROUP BY pz.geom, dz.geom, {select_condition}
    limit 50;
    """
    df_taxi = pd.read_sql(query, engine)
    df_taxi["centroid_pickup"] = df_taxi["centroid_pickup"].apply(wkt.loads)
    df_taxi["centroid_dropoff"] = df_taxi["centroid_dropoff"].apply(wkt.loads)
    
    gdf_airport = gpd.GeoDataFrame(df_taxi, geometry="centroid_pickup", crs="EPSG:4326")
    
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

def generate_large_arc(start, end, num_points=30, arc_height_factor=0.7, peak_position=0.5):
    lat1, lon1 = start
    lat2, lon2 = end
    
    arc_lats, arc_lons = [], []
    
    for i in np.linspace(0, 1, num_points):
        # Pravimo interpoliranu tačku između start i end
        interpolated_lat = lat1 + i * (lat2 - lat1)
        interpolated_lon = lon1 + i * (lon2 - lon1)  

        if i <= peak_position:
            x = i / peak_position  # Normalizujemo na [0,1] za prvu polovinu
            height = np.sin(x * np.pi/2)  # Raste do pika
        else:
            x = (i - peak_position) / (1 - peak_position)  # Normalizujemo za drugu polovinu
            height = np.cos(x * np.pi/2)  # Opada od pika
        
        interpolated_lat += height * arc_height_factor * abs(lat2 - lat1)
        
        arc_lats.append(interpolated_lat)
        arc_lons.append(interpolated_lon)

       # widths.append(10 * (1 - i) + 1)  # Linearno smanjenje od 10 do 1

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
    
def plot_airports(fig, gdf_airport):
    fig.add_trace(go.Scattermapbox(
        lon=gdf_airport["longitude"],
        lat=gdf_airport["latitude"],
        hovertext=gdf_airport["zone"],  # Tekst za hover (može biti lista/Series)
        hoverinfo='text',  # Prikazuje samo hovertext
        mode='text',
        text = 'X',
        textfont=dict(
            size=20,  # Povećajte veličinu fonta
            color='black',
            weight='bold',
        ),
        textposition="middle center",
        showlegend=False
    ))
    return fig
    
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

def plot_lines(fig, gdf_airport, color, toggle_value, airport, arc_height_factor):
    for i in range(len(gdf_airport)):
        start_point = (gdf_airport['pickup_latitude'][i], gdf_airport['pickup_longitude'][i])
        end_point = (gdf_airport['dropoff_latitude'][i], gdf_airport['dropoff_longitude'][i])
        
        if toggle_value == 'dropoff':
            arc_lats, arc_lons = generate_large_arc(start_point, end_point, num_points=8, arc_height_factor= arc_height_factor, peak_position=0.8)
            text_value = f' {gdf_airport['zone'][i]} -> {airport}: {gdf_airport['trip_count'][i]}'
            
        else:
            arc_lats, arc_lons = generate_large_arc(start_point,end_point,  num_points=8, arc_height_factor=arc_height_factor, peak_position=0.5)
            text_value = f' {gdf_airport['zone'][i]} <- {airport}: {gdf_airport['trip_count'][i]}'

        max_opacity = float(gdf_airport['trip_count'].max())
        
        fig.add_trace(
            go.Scattermapbox(
                    lon = arc_lons,
                    lat = arc_lats,
                    mode = 'lines',
                    line = dict(width = 1, color = color),
                    opacity = float(gdf_airport['trip_count'][i]) / max_opacity,
                    text = text_value,
                    hoverinfo='text',
                    showlegend=False
            )
        )        
    return fig

def plot_lines_borough(fig, gdf_airport_borough , color, toggle_value, airport):
    global borough_centroids, gdf_airport_centroid
    
    for i, row in gdf_airport_borough.iterrows():
        borough_name = row['borough']
        centroid = gdf_borough[gdf_borough['borough'] == borough_name]
        if toggle_value == "dropoff":
            start_point = (centroid.iloc[0]['latitude'], centroid.iloc[0]['longitude'])
            end_point = (gdf_airport_borough['latitude'][i], gdf_airport_borough['longitude'][i]) 
            arc_lats, arc_lons = generate_large_arc(start_point, end_point)
            text_value = f' {gdf_airport_borough['borough'][i]} -> {airport}: {gdf_airport_borough['trip_count'][i]}'
        else:
            end_point = (centroid.iloc[0]['latitude'], centroid.iloc[0]['longitude'])
            start_point = (gdf_airport_borough['latitude'][i], gdf_airport_borough['longitude'][i]) 
            arc_lats, arc_lons = generate_large_arc(start_point, end_point)
            text_value = f'{gdf_airport_borough['borough'][i]} <- {airport}: {gdf_airport_borough['trip_count'][i]}'
        
        for j in range(len(arc_lons)-1):
            width = max(0.5, 7 * (j / (len(arc_lons)-1)))
            fig.add_trace(
                go.Scattermapbox(
                    lon=[arc_lons[j], arc_lons[j+1]],
                    lat=[arc_lats[j], arc_lats[j+1]],
                    mode='lines',
                    line=dict(width=width, color=color),  # Width decreases
                    text = text_value,
                    hoverinfo='text',
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
                    mode="markers+text",
                    marker=dict(size=10, color="red", opacity=0.5),  # nevidljiv marker koji prima klik
                    text=text_values,
                    textposition="top center",
                    name = row[name],
                    customdata=[[row[name]]],
                    hoverinfo="none",
                    showlegend=False
                ))
        elif geom.geom_type == "Polygon":
            lon, lat = geom.exterior.xy
            text_values = [row[name]] * len(lon)
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=list(lon),
                lat=list(lat),
                line=dict(width=0.8,  color="black"),
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
                        mode="markers+text",
                        marker=dict(size=10, color="red", opacity=0.5),  # nevidljiv marker koji prima klik
                        text=text_values,
                        textposition="middle left",
                        name = row[name],
                        customdata=[[row[name]]],
                        hoverinfo="none",
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
    arc_height_factor = [0.5, 0.7, 0.9]
    color_list = [COLORS["red"], COLORS["purple"], COLORS["blue"]]
    
    fig = go.Figure()
    flowmap_title = ""
    
    if toggle_value == "dropoff":
        # to airport
        flowmap_title += "Trips to airports"
    else:
        flowmap_title += "Trips from airports"        
    
    if selected_borough:
        flowmap_title += f" to {selected_borough}"
        zones_in_borough = gdf[gdf['borough'] == selected_borough]
        fig = show_polygons(fig, zones_in_borough, "zone")
        fig = plot_airports(fig, gdf_airport_centroid)
        
        for i, airport in enumerate(airports):
            gdf_airport_zone = fetch_airport(selected_date, start_time, end_time, airport, selected_borough, toggle_value)
            fig = plot_places(fig, gdf_airport_zone, color_list[i])
            fig = plot_lines(fig, gdf_airport_zone, color_list[i], toggle_value, airport,arc_height_factor[i])
    else:
        fig = show_polygons(fig, gdf_borough, "borough")
        fig = plot_airports(fig, gdf_airport_centroid)
        for i, airport in enumerate(airports):
            print(f"Iscrava se za aerodrom {airport}")
            gdf_airport_borough = fetch_airport_borough(selected_date, start_time, end_time, airport, toggle_value)
            fig = plot_lines_borough(fig, gdf_airport_borough , color_list[i], toggle_value, airport)
        

    fig.update_layout(
        uirevision=None,
        mapbox=dict(
            accesstoken="pk.eyJ1IjoibWFyaWphcmlzdGljMjMiLCJhIjoiY21hZjZpeTc4MDIzZjJqcjFjcWhvMTRyNiJ9.V7dv1K-HL_i3asRs3aKmfg", 
            style="light",  #"light" "dark", "satellite", "streets"
            center=dict(lat=40.7128, lon=-74.0060),  # Centar NYC
            zoom=9.2,
            bearing=-20
        ),
        margin=dict(t=40, b=0, l=0, r=0),
        plot_bgcolor="white",
        title={
            'text': flowmap_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'family': 'Gotham',
                'size': 25,
                'color': COLORS['background']
            }
        },
    )
    
    return fig

def get_taxi_frequency_by_hour(selected_date, selected_borough, toggle_value):
    ekstra_condition = ''
    cond = ''
    if toggle_value == 'dropoff':
        cond = 'dz.zone'
        if selected_borough:
            ekstra_condition = f"AND pz.borough = '{selected_borough}'"
            print("desio se ekstra condition")
    else:
        cond = 'pz.zone'
        if selected_borough:
            ekstra_condition = f"AND dz.borough = '{selected_borough}'"
    query = f'''
    SELECT 
        {cond} as borough,
        EXTRACT(HOUR FROM t.tpep_pickup_datetime) AS hour,
        COUNT(*) AS drives
    FROM taxi t
    JOIN taxi_zones pz ON ST_Contains(pz.geom, t.geom_pickup)
    JOIN taxi_zones dz ON ST_Contains(dz.geom, t.geom_dropoff)
    WHERE {cond} IN ('JFK Airport', 'LaGuardia Airport', 'Newark Airport') {ekstra_condition}
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
    print(f"borughs sorted: {boroughs_sorted}")
    
    for i, borough in enumerate(boroughs_sorted):
        stream_fig.add_trace(go.Scatter(
            x=df_taxi_frequency_pivot['hour'],
            y=df_taxi_frequency_pivot[borough],
            name=borough,
            mode='lines',
            stackgroup='one',  # Ključno za streamgraph!
            line=dict(width=0.5, shape='spline', color = color_list[i % len(color_list)]),  # Glatke krivine
            hoverinfo='x+y+name',
            hovertemplate=f'<b>{borough}</b><br>Hour: %{{x}}<br>Num of drives: %{{y}}<extra></extra>'
        ))

    stream_fig.update_layout(
        xaxis_title='Hour in day',
        yaxis_title='Num of drives',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(
            tickvals=list(range(24)),
            ticktext=[f'{h}:00' for h in range(24)]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        margin=dict(t=80, b=10, l=40, r=20),  # ← más espacio arriba y abajo
        title={
            'text': f"Number of taxi trips {'to' if toggle_value == 'dropoff' else 'from'} airports{f' in {selected_borough}' if selected_borough else ''}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'family': 'Gotham',
                'size': 22,
                'color': COLORS['background']
            }
        },
        legend=dict(
            orientation="h",       # ← horizontal para que no se superponga
            yanchor="bottom",
            y=1.02,                # ← justo debajo del título
            xanchor="center",
            x=0.5,
            font=dict(
                family='Gotham',
                size=12
            )
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
    cond = 'pz'
    #if toggle_value == 'dropoff':
    #    cond = 'pz'
    #    
    #else:
    #    cond = 'dz'

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
        gdf_tip_amount =  gpd.GeoDataFrame(df_merge, geometry="geom", crs="EPSG:4326")

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
        gdf_tip_amount =  gpd.GeoDataFrame(df_merge, geometry="geom", crs="EPSG:4326")

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
        title={
            'text': heatmap_tittle,
            'x': 0.5,
            'xanchor': 'center',
            'font': {
                'family': 'Gotham',
                'size': 25,
                'color': COLORS['background']
            }
        },
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Iznos ($)',
            'thickness': 15,
            'len': 0.5
        }
    )
    return heatmap_fig

initial_index = len(available_dates) // 2  
initial_date = available_dates[initial_index] 

initial_time_range = [8, 12]
figure_initial = update_map(initial_date, initial_time_range, None, 'dropoff')
streamgraph_figure_initial = update_streamgraph(initial_date, None, 'dropoff')
heatmap_figure_initial = update_heatmap(initial_date, initial_time_range, None, 'dropoff')

white_square_shadow_box = '0 0 10px rgba(0,0,0,0.1)'
options_start_time = [{'label': f'{i}:00h', 'value': i} for i in range(0, 24)]
options_final_time = [{'label': f'{i}:00h', 'value': i} for i in range(0, 24)]
options_final_time.append({'label': '23:59h', 'value': 23})

app.layout = html.Div([

    # Mensaje inicial flotante
    html.Div([
        html.Button("X", id="close-instructions", n_clicks=0, style={
            'position': 'absolute',
            'top': '10px',
            'right': '10px',
            'background': 'none',
            'border': 'none',
            'fontSize': '20px',
            'cursor': 'pointer',
            'color': 'white'
        }),
        html.H4("How to use this dashboard", style={
            'marginBottom': '10px',
            'color': 'white',
            'fontFamily': 'Gotham'
        }),
        html.P("""- Use the toggle to switch between trips from or to the airports.
- Pick a specific day using the date picker.
- Filter trips by hours with the dropdowns.

Visualizations:
- Flowmap: Shows trips between airports and boroughs. Click to drill down or back.
- Heatmap: It will change with the flowmap changes, showing borough or zone level selected in the flowmap.
- Streamgraph: Trips over time. Hover for details.""", style={
            'whiteSpace': 'pre-line',
            'lineHeight': '1.5',
            'color': 'white',
            'fontFamily': 'Gotham'
        })
    ], id='instructions-box', style={
        'position': 'fixed',
        'top': '300px',
        'left': '50%',
        'transform': 'translateX(-50%)',
        'backgroundColor': '#153E67',
        'padding': '20px 30px',
        'borderRadius': '12px',
        'boxShadow': white_square_shadow_box,
        'zIndex': '2000',
        'width': '90%',
        'maxWidth': '600px'
    }),

    # Fila 1 - Título
    html.Div([
        html.H1("Taxi trips in NYC", style={
            'textAlign': 'center',
            'color': COLORS['accent'],
            'fontFamily': 'Gotham',
            'margin': '20px 0'
        })
    ]),

    # Fila 2 - Menú a la izquierda y Flowmap a la derecha
    html.Div([
        # Columna izquierda: Menú con fondo blanco
        html.Div([
            html.Div([
                html.Img(src="/assets/Nyctlc_logo.webp", style={
                    'height': '100px',
                    'marginBottom': '20px'
                }),

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
                    'marginBottom': '20px'
                }),

                # Etiqueta para el DatePicker
                html.Div([
                    html.Label("Date:", style={
                        'color': COLORS['background'],
                        'fontFamily': 'Gotham',
                        'fontSize': '14px',
                        'fontWeight': 'bold',
                        'marginBottom': '5px'
                    }),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        className='custom-date',
                        min_date_allowed=min(available_dates),
                        max_date_allowed=max(available_dates),
                        date=available_dates[initial_index],
                        first_day_of_week=1,
                        show_outside_days=False,
                        display_format='DD-MM-YYYY',
                        style={'marginBottom': '20px'}
                    )
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'alignItems': 'center'
                }),

                html.Div([
                    html.Div([
                        html.Label("Start hour:", style={
                            'marginBottom': '5px',
                            'color': COLORS['background'],
                            'fontFamily': 'Gotham',
                            'fontWeight': 'bold',
                            'fontSize': '14px'
                        }),
                        dcc.Dropdown(
                            id='start-time-dropdown',
                            options=options_start_time,
                            value=initial_time_range[0],
                            clearable=False,
                            style={
                                'width': '100%',
                                'marginBottom': '15px',
                                'color': COLORS['background'],
                                'fontFamily': 'Gotham',
                                'fontWeight': 'bold',
                                'fontSize': '14px'
                            }
                        ),
                    ], style={'flex': '1', 'marginRight': '10px'}),

                    html.Div([
                        html.Label("Final hour:", style={
                            'marginBottom': '5px',
                            'color': COLORS['background'],
                            'fontFamily': 'Gotham',
                            'fontWeight': 'bold',
                            'fontSize': '14px'
                        }),
                        dcc.Dropdown(
                            id='end-time-dropdown',
                            options=options_final_time,
                            value=initial_time_range[1],
                            clearable=False,
                            style={
                                'width': '100%',
                                'color': COLORS['background'],
                                'fontFamily': 'Gotham',
                                'fontWeight': 'bold',
                                'fontSize': '14px'
                            }
                        ),
                    ], style={'flex': '1'})
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'flex-start',
                    'justifyContent': 'flex-start',
                    'width': '100%',
                    'maxWidth': '250px'
                })
            ], style={
                'margin': 'auto 0',  # Centrado vertical
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center'
            })
        ], style={
            'width': '30%',
            'marginRight': '15px',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': white_square_shadow_box,
            'height': '100%',
            'boxSizing': 'border-box',
            'display': 'flex',
            'flexDirection': 'column'
        }),


        # Columna derecha: Flowmap
        html.Div([
            dcc.Graph(id='map-graph', figure=figure_initial, style={'height': '100%'})
        ], style={
            'width': '70%',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': white_square_shadow_box,
            'boxSizing': 'border-box',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'space-between'
        })
    ], style={
        'display': 'flex',
        'margin': '0 20px 15px 20px',  # <-- margen inferior vertical igualado a 15px
        'alignItems': 'stretch',
        'height': '55vh'
    }),

    # Fila 3 - Heatmap y Streamgraph
    html.Div([
        html.Div([
            dcc.Graph(id='heatmap-graph', figure=heatmap_figure_initial)
        ], style={
            'width': '50%',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'marginRight': '15px',  # <-- separación horizontal igualada a 15px
            'boxShadow': white_square_shadow_box
        }),

        html.Div([
            dcc.Graph(id='stream-graph', figure=streamgraph_figure_initial)
        ], style={
            'width': '50%',
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': white_square_shadow_box
        }),
    ], style={
        'display': 'flex',
        'margin': '0 20px 15px 20px'  # <-- margen inferior vertical igualado a 15px
    })

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
    Output('end-time-dropdown', 'options'),
    Output('end-time-dropdown', 'value'),
    Input('start-time-dropdown', 'value'),
    prevent_initial_call=True
)
def update_end_time_options(start_hour):
    if start_hour is None or start_hour >= 24:
        return [], None

    end_options = [{'label': f'{i}:00h', 'value': i} for i in range(start_hour + 1, 25)]

    # Por defecto, selecciona la primera opción válida
    new_value = end_options[0]['value'] if end_options else None

    return end_options, new_value

@app.callback(
    Output('toggle-label', 'children'),
    Input('location-toggle', 'value')
)
def update_toggle_label(toggle_value):
    return "To airports" if toggle_value is False else "From airports"


@app.callback(
    [Output("map-graph", "figure"),
    Output("stream-graph", "figure"),
    Output("heatmap-graph", "figure")],
    [Input("date-picker", "date"),
    Input("start-time-dropdown", "value"),
    Input("end-time-dropdown", "value"),
    Input("map-graph", "clickData"),
    Input("location-toggle", "value")],
    [State("stream-graph", "figure")]
)
def combined_callback(selected_date, start_hour, end_hour, click_data, toggle_value, curr_stream_figure):
    if start_hour is None or end_hour is None:
        return dash.no_update, dash.no_update, dash.no_update

    if end_hour >= 24:
        end_hour = 23.9833

    # Determina qué input activó el callback
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id']
    print(f"Trigger recibido: {triggered}")

    # Normaliza el valor del toggle
    toggle_value = 'dropoff' if toggle_value is False else 'pickup'

    # Determina el borough si hay clickData
    selected_borough = None
    if click_data and 'points' in click_data:
        for point in click_data['points']:
            borough = point.get('text')  # Verifica que sea el mismo atributo que usas en update_map()
            if borough and borough.strip() in gdf_borough['borough'].values:
                selected_borough = borough.strip()
                print("Clicked borough:", selected_borough)

    # Rango de tiempo
    time_range = [start_hour, end_hour]

    # Actualiza gráficos
    map_fig = update_map(selected_date, time_range, selected_borough, toggle_value)
    heatmap_fig = update_heatmap(selected_date, time_range, selected_borough, toggle_value)

    # Condición inteligente para no sobreescribir streamgraph vacío
    if triggered in ['date-picker.date', 'location-toggle.value', 'map-graph.clickData'] or selected_borough:
        streamgraph_fig = update_streamgraph(selected_date, selected_borough, toggle_value)
    else:
        streamgraph_fig = curr_stream_figure

    return map_fig, streamgraph_fig, heatmap_fig



if __name__ == "__main__":
    app.run(debug=False)