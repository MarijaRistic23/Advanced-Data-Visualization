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
from dash import dcc, html, Input, Output, callback
import dash_daq as daq
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import time
import warnings
import dash_bootstrap_components as dbc
from dash import Dash


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

config = configparser.ConfigParser()
config.read('config.cfg')
db_name = "nyc_taxi_adv"
username = config.get('credentials', 'username')
password = config.get('credentials', 'password')
db_url = f"postgresql://postgres:{password}@localhost:5432/nyc_taxi_adv"
engine = create_engine(db_url)
CRS = "EPSG:4326"

query = 'SELECT "LocationID", zone, borough, geom FROM taxi_zones;'
gdf = gpd.read_postgis(query, engine, geom_col="geom")
gdf.head()

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

borough_centroids = {row['borough']: (row['latitude'], row['longitude']) for _, row in gdf_borough.iterrows()}

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
gdf_airport_centroid.head()

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
    #gdf_Newark.head()
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
#available_dates

marks={i: {'label': available_dates[i].strftime('%Y-%m-%d'), 'style': {'color': 'red'}} 
        for i in range(0, len(available_dates), 10)}  # Oznake na svakih 10 dana
last_idx = len(available_dates) - 1
if last_idx not in marks:
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
    import math
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
                line=dict(width=0.8, 
                            color="black"),
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
    color = ['red', 'blue', 'green']
    
    fig = go.Figure()
        
    if selected_borough:
        zones_in_borough = gdf[gdf['borough'] == selected_borough]
        fig = show_polygons(fig, zones_in_borough, "zone")
        for i, airport in enumerate(airports):
            gdf_airport_zone = fetch_airport(selected_date, start_time, end_time, airport, selected_borough, toggle_value)
            fig = plot_places(fig, gdf_airport_zone, color[i])
            fig = plot_lines(fig, gdf_airport_zone, color[i])
        
    else:
        fig = show_polygons(fig, gdf_borough, "borough")
        for i, airport in enumerate(airports):
            print(f"Iscrava se za aerodrom {airport}")
            gdf_airport_borough = fetch_airport_borough(selected_date, start_time, end_time, airport, toggle_value)
            fig = plot_lines_borough(fig, gdf_airport_borough , color[i], toggle_value)

    fig.update_layout(
        uirevision="constant",
        mapbox=dict(
            accesstoken="pk.eyJ1IjoibWFyaWphcmlzdGljMjMiLCJhIjoiY21hZjZpeTc4MDIzZjJqcjFjcWhvMTRyNiJ9.V7dv1K-HL_i3asRs3aKmfg", 
            style="light",  #"light" "dark", "satellite", "streets"
            center=dict(lat=40.7128, lon=-74.0060),  # Centar NYC
            zoom=9.5,
            bearing=-20
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        plot_bgcolor="white",
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

    
    for borough in boroughs_sorted:
        stream_fig.add_trace(go.Scatter(
            x=df_taxi_frequency_pivot['hour'],
            y=df_taxi_frequency_pivot[borough],
            name=borough,
            mode='lines',
            stackgroup='one',  # Ključno za streamgraph!
            line=dict(width=0.5, shape='spline'),  # Glatke krivine
            hoverinfo='x+y+name',
            hovertemplate=f'<b>{borough}</b><br>Hour: %{{x}}<br>Num of drives: %{{y}}<extra></extra>'
        ))

    stream_fig.update_layout(
        title='Number of taxi rides per airport for the selected day',
        xaxis_title='Hour in day',
        yaxis_title='Num of drives',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(tickvals=list(range(24)), ticktext=[f'{h}:00' for h in range(24)]),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
    )
    print("zavrseno pravljenje stream figure")
    return stream_fig

initial_index = len(available_dates) // 2  
initial_date = available_dates[initial_index] 

initial_time_range = [8.0, 12.0]

figure_initial = update_map(initial_date, initial_time_range, None, 'dropoff')

app.layout = dbc.Container([
    dbc.Row([
        # Columna 1: menú lateral izquierdo
        dbc.Col([
            # Aquí pones tu menú, botones, etc.
            html.Hr(),
            html.Label("Visualizations"),
            dbc.ButtonGroup([
                dbc.Button("FlowMap", id="tab-map", n_clicks=0),
                dbc.Button("Heatmap", id="tab-bar", n_clicks=0),
                dbc.Button("Otro", id="tab-other", n_clicks=0),
            ], vertical=True, className="w-100")
        ], md=3, className="bg-light p-3 rounded shadow-sm"),

        # Columna 2: contenido principal dividido en 4 filas
        dbc.Col([
            # Fila 1: título
            dbc.Row([
                html.H1("Airports Traffic in NYC", className="text-center mb-4", style={"width": "100%"})
            ]),

            # Fila 2: toggle + date picker
            dbc.Row([
                dbc.Col([
                    # DatePicker
                    html.Label("Select a Date"),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        min_date_allowed=min(available_dates),
                        max_date_allowed=max(available_dates),
                        date=available_dates[initial_index],
                        display_format='YYYY-MM-DD',
                        first_day_of_week=1,
                        show_outside_days=False,
                        style={"width": "100%"}
                    )
                ], md=8), 
                dbc.Col([
                    # Toggle con textos a los lados
                    dbc.Row([
                        dbc.Col(html.Div("from airport", className="text-end pe-2"), width="auto", style={"alignSelf": "center"}),
                        dbc.Col(daq.ToggleSwitch(
                            id='location-toggle',
                            value=False,
                            vertical=False,
                            color="#FFD700"
                        ), width="auto"),
                        dbc.Col(html.Div("to airport", className="text-start ps-2"), width="auto", style={"alignSelf": "center"})
                    ], justify="center", align="center")
                ], md=4)
            ], className="mb-3"),

            # Fila 3: slider
            dbc.Row([
                html.Label("Time Range"),
                dcc.RangeSlider(
                    id='time-range-slider',
                    min=0,
                    max=24,
                    step=1,
                    marks={i: f"{i}:00h" for i in range(0, 25, 4)},
                    value=initial_time_range,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className="mb-3"),

            # Fila 4: gráfico flowmap
            dbc.Row([
                dcc.Graph(
                    id="map-graph",
                    figure=figure_initial,
                    style={"height": "600px", "width": "100%"}
                )
            ])
        ], md=9)
    ])
], fluid=True)



def combined_callback(selected_date, time_range, click_data, toggle_value):
    time_range = sanitize_time_range(time_range)
    toggle_value = 'dropoff' if not toggle_value else 'pickup'
    selected_borough = extract_borough_from_click(click_data)

    fig = update_map(selected_date, time_range, selected_borough, toggle_value)
    return fig, time_range if time_range[1] >= 24 else dash.no_update

def sanitize_time_range(time_range):
    start, end = time_range
    if end >= 24:
        end = 23.9833
    return [start, end]

def extract_borough_from_click(click_data):
    if not (click_data and 'points' in click_data):
        return None

    print("clickData:", click_data)  # Debug

    for point in click_data['points']:
        borough = point.get('text')
        if borough and borough in gdf_borough['borough'].values:
            print("clicked okrug:", borough)
            return borough
        elif borough is None:
            print("Nisam nasao customdata")

    return None

    
if __name__ == "__main__":
    app.run(debug=False)