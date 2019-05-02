""" Visualization of Results

# TODO: This should run from MLFlow dumps.
"""


import json
import os
from glob import glob

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from pandas.io.sql import DatabaseError

from source.config import sql
from source.utils import run_path_to_datetime

app = dash.Dash(__name__)

app.css.append_css({"external_url": "https://codepen.io/thoward27/pen/aKzpEZ.css"})

# colors
# noinspection SpellCheckingInspection
COLORS = {
    'STATIC': '5cbae6',
    'DYNAMIC': 'b6d957',
    'HYBRID': 'fac364',
}


def extract_model_type(path):
    with open(path + '/details.json', 'r') as f:
        details = json.load(f)['details']
    try:
        return details['MODEL'].upper()
    except KeyError:
        return ""


def serve_layout():
    return html.Div([
        # Title
        html.H1('COREL Results Explorer'),
        html.Hr(),

        # Global Filters
        html.Div([
            html.H4('Select a run to visualize.'),
            dcc.Dropdown(
                id='dd-run',
                value=(sorted([r for r in glob('./runs/*') if os.path.exists(r + '/data.db')])[-1]),
                options=[{
                    'label': "{} {}".format(run_path_to_datetime(r), extract_model_type(r)),
                    'value': r,
                    'disabled': not os.path.exists(r + '/data.db')
                } for r in sorted(glob('./runs/*'))],
                clearable=False,
                placeholder="Select a run",
            ),
        ]),
        html.Hr(),

        html.Div([
            html.H4('Full Table'),
            html.Div(children=[
                html.Label("Aggregate data on:"),
                dcc.Dropdown(
                    id='dd-data-table-agg',
                    placeholder='Select values to aggregate data.',
                    multi=True,
                    value='feature_set'
                ),
                html.Label("Hide columns:"),
                dcc.Dropdown(
                    id='dd-data-table-hide',
                    placeholder='Select values to hide.',
                    multi=True,
                    value='epoch'
                ),
                dt.DataTable(
                    rows=[{}],
                    row_selectable=False,
                    filterable=True,
                    sortable=True,
                    id='data-table'
                ),
            ]),
        ]),
        html.Hr(),

        # Results by Program
        html.Div([
            html.H4('Results by Program'),
            html.Button('Toggle Show Datasets', id='btn-toggle-dataset', n_clicks=1),
            html.Div(children=[
                dcc.Graph(id='g-xprogram'),
                html.Label(children='Select a metric for the Y axis.'),
                dcc.Dropdown(
                    id='dd-g-xprogram-yselector',
                    placeholder='Select an element for the Y axis',
                    value='wrt_opt',
                    multi=True
                ),
            ]),
        ]),

        # Results by Epoch
        html.Div([
            html.H4('Results by Epoch'),
            html.Div(children=[
                dcc.Graph(id='g-xepoch'),
                html.Label(children='Select a metric for the Y axis.'),
                dcc.Dropdown(
                    id='dd-g-xepoch-yselector',
                    placeholder='Select an element for the Y axis.',
                    value='wrt_opt',
                    multi=True
                ),
            ]),
        ]),

        # Other graphs for data exploration.
        html.Div([
            html.H4("Surface of program runtimes"),
            dcc.Graph(id='g-surface-runtimes'),
        ]),
        html.Hr(),

        # Run metadata
        html.Div([
            html.H4('Run Details'),
            html.P(id='txt-run-details'),
        ]),
        html.Hr(),

        # Caches
        html.Div(id='program-metrics-cache', style={'display': 'none'}),
        html.Div(id='model-metrics-cache', style={'display': 'none'})
    ])


app.layout = serve_layout


def safe_query(run, query):
    try:
        conn = sql.connect("file:{}?mode=ro".format(run + '/data.db'), uri=True)
        data = pd.read_sql(query, conn)
        conn.close()
    except DatabaseError:
        conn = sql.connect(run + '/data.db')
        conn.execute(
            'create view if not exists vProgramMetrics '
            'as select *, o3 / one_shot as wrt_03_1, o3 / five_shot as wrt_03_5, o3 / ten_shot as wrt_03_10, '
            'opt / ten_shot as wrt_opt from ProgramMetric'
        )
        conn.commit()
        data = pd.read_sql(query, conn)
        conn.close()

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(axis=1, how='all', inplace=True)
    return data.round(3)


@app.callback(Output('txt-run-details', 'children'), [Input('dd-run', 'value')])
def update_txt_run_details(run):
    """ Displays metadata about the selected run. """
    with open(run + '/details.json', 'r') as f:
        details = json.load(f)['details']
    return '; '.join(['({}: {})'.format(k, v) for k, v in details.items() if type(v) is not list])


@app.callback(Output('dd-data-table-agg', 'options'), [Input('dd-run', 'value')])
def update_dd_data_table_agg(run):
    """ Updates the aggregation options for the dynamic table. """
    df = safe_query(run, 'select * from vProgramMetrics limit 1;')
    return [{'label': col, 'value': col} for col in df.columns]


@app.callback(Output('dd-data-table-hide', 'options'), [Input('dd-data-table-agg', 'options')])
def update_dd_data_table_hide(cols):
    return cols


@app.callback(Output('data-table', 'rows'), [
    Input('dd-run', 'value'),
    Input('dd-data-table-agg', 'value'),
    Input('dd-data-table-hide', 'value'),
])
def update_data_table(run, agg, drop):
    """ Updates the dynamic table according to global and local options. """
    df = safe_query(run, 'select * from vProgramMetrics;')

    if agg:
        df = df.groupby(agg, as_index=False).mean().round(2)

    if drop:
        df = df.drop(drop, axis=1, errors='ignore')

    return [{k: v for (k, v) in d.items()} for d in df.to_dict('records')]


@app.callback(Output('dd-g-xepoch-yselector', 'options'), [Input('dd-run', 'value')])
def update_g_xepoch_yselector(run):
    """ Loads the valid yaxis options for the xepoch graph. """
    df = safe_query(run, 'select * from vProgramMetrics limit 1;')
    return [{'label': col, 'value': col} for col in df.columns]


@app.callback(Output('g-xepoch', 'figure'), [
    Input('dd-run', 'value'),
    Input('dd-g-xepoch-yselector', 'value'),
])
def update_g_xepoch(run, yaxis):
    if type(yaxis) is not list:
        yaxis = [yaxis]

    df = safe_query(run, 'select * from vProgramMetrics;')
    df = df.groupby(['feature_set', 'epoch'], as_index=False).mean().round(2)

    return {
        'data': sorted([{
            'x': df['epoch'][df['feature_set'] == fs],
            'y': df[y][df['feature_set'] == fs],
            'name': ' '.join([fs, y]),
            'text': ' '.join([fs, y]),
            'marker': {'color': COLORS[fs]},
            'line': {'color': COLORS[fs]},
        } for fs in df['feature_set'].unique() for y in yaxis], key=lambda d: d['name']),
        'layout': {
            'title': 'model Performance by epoch',
            'xaxis': {'title': 'epoch'},
            'yaxis': {'title': ', '.join(yaxis)},
        }
    }


@app.callback(Output('dd-g-xprogram-yselector', 'options'), [Input('dd-run', 'value')])
def update_g_xprogram_yselector(run):
    df = safe_query(run, 'select * from vProgramMetrics limit 1;')
    return [{'label': col, 'value': col} for col in df.columns]


@app.callback(Output('g-xprogram', 'figure'), [
    Input('dd-run', 'value'),
    Input('dd-g-xprogram-yselector', 'value'),
    Input('btn-toggle-dataset', 'n_clicks')
])
def update_g_xprogram(run, yaxis, dataset_toggle_n_clicks):
    if type(yaxis) is not list:
        yaxis = [yaxis]

    df = safe_query(run, 'select * from vProgramMetrics;')
    if dataset_toggle_n_clicks % 2 == 0:
        df['name'] = df[['name', 'dataset']].apply(lambda x: '_'.join(str(b) for b in x), axis=1)
    df = df.groupby(['feature_set', 'name'], as_index=False).max().round(2)

    return {
        'data': sorted([{
            'x': df['name'][df['feature_set'] == fs],
            'y': df[y][df['feature_set'] == fs],
            'name': ' '.join([fs, y]),
            'text': ' '.join([fs, y]),
            'mode': 'lines',
            'marker': {'color': COLORS[fs]},
        } for fs in df['feature_set'].unique() for y in yaxis], key=lambda d: d['name']),
        'layout': {
            'title': 'Model Performance by Program',
            'xaxis': {'title': 'Programs'},
            'yaxis': {'title': ', '.join(yaxis)},
            'margin': {'b': 200}
        }
    }


@app.callback(Output('g-surface-runtimes', 'figure'), [Input('dd-run', 'value')])
def update_g_surface_runtimes(run):
    """ Displays the runtimes of all programs as a surface graph. """
    with open(run + '/programs.json', 'r') as f:
        runtimes = pd.DataFrame(
            [[p['name'], p['dataset'], *p['runtimes']] for p in json.load(f)['programs']],
            columns=['name', 'dataset', *[i for i in range(128)]],
            dtype=np.float64
        ).groupby('name').mean().drop('dataset', axis=1).round(2)
    runtimes['average'] = runtimes.mean(axis=1, numeric_only=True)
    runtimes.sort_values('average', axis=0, inplace=True, ascending=False)
    runtimes.drop('average', axis=1)
    return {
        'data': [{
            'type': 'surface',
            'x': [i for i in range(128)],
            'y': runtimes.index,
            'z': runtimes.values.tolist(),
        }],
        'layout': {
            'title': 'Surface of runtimes',
            'autosize': True,
        }
    }


if __name__ == '__main__':
    app.run_server(debug=True)
