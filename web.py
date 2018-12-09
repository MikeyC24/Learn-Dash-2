from dash import Dash
from werkzeug.wsgi import DispatcherMiddleware
import flask
from werkzeug.serving import run_simple
import dash_html_components as html
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import os, sys
import plotly.graph_objs as go
import DataDisplay
from flask import request, jsonify, send_file, render_template

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
server = flask.Flask(__name__)
dash_app1 = Dash(__name__, server = server, url_base_pathname='/dashboard/' )
dash_app_infl = Dash(__name__, server = server, url_base_pathname='/Inflation/',external_stylesheets=external_stylesheets)
dash_app_gdp = Dash(__name__, server = server, url_base_pathname='/GDP/',external_stylesheets=external_stylesheets)
dash_app_ir = Dash(__name__, server = server, url_base_pathname='/Interest Rates/',external_stylesheets=external_stylesheets)
dash_app1.layout = html.Div([html.H1('Hi there, I am app1 for dashboards')])
dash_app_infl.layout = DataDisplay.run_inflation_summary(dash_app_infl)
dash_app_gdp.layout = DataDisplay.run_gdp_summary(dash_app_gdp)
dash_app_ir.layout = DataDisplay.run_ir_summary(dash_app_ir)

@server.route('/')
@server.route('/hello')
def hello():
    return 'hello world!'

@server.route('/dashboard')
def render_dashboard():
    return flask.redirect('/dash1')


@server.route('/reports')
def render_reports():
    return flask.redirect('/dash1')


@server.route('/Interest Rates')
def render_IR():
    return flask.redirect('/dash4')

@server.route('/Inflation')
def render_Inflation():
    return flask.redirect('/dash2')

@server.route('/GDP')
def render_GDP():
    return flask.redirect('/dash3')

@server.route('/home')
def update_output():
    return render_template('home.html')


@server.route('/app')
def render_reports1():
    return 'Model'


app = DispatcherMiddleware(server, {
    '/dash1': dash_app1.server,
    '/dash2': dash_app_infl.server,
    '/model':server,
    '/dash3': dash_app_gdp.server,
    '/dash4': dash_app_ir.server
})

run_simple('0.0.0.0', 8080, app, use_reloader=True, use_debugger=True)




