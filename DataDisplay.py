import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import os, sys
import plotly.graph_objs as go
from Dashboard import return_combine_df_for_graph,get_dfs_for_display_inflation, get_gdp_graphs

"""
not a class, maybe will not be turned into one

"""

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#dash_app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

text_dict = {
	'Interest Rates':"""
	lets
	SEE
	if this 
	works
	[click](href="/Interest Rates")
	[Assemble](http://assemble.io)
	<http://foo.bar.baz>
	""",
	'Inflation':"""
	inflation 


	space 
	""",
	'GDP':"""
	inflation 


	space 
	"""
}




def df_convert_for_multi_display(df, drop_cols):
	"""
	#this method takes a df and arranges it so it can be graphed on an interact display where the values can
	#be click on and off
	"""
	df = df.drop(drop_cols, axis=1)
	df_for_dash = pd.DataFrame(index=df.index)
	name_array = []
	value_array = []
	for col in df.columns:
		col_name = [col] * len(df[col])
		name_array = name_array + col_name
		value_array = value_array + list(df[col]) 
		df_for_dash = pd.DataFrame.from_dict({'Spread_Names':name_array, 'Spread_Values':value_array})
	return df_for_dash



def display_interactive_line_graph(dash_app):

	df = return_combine_df_for_graph()
	df_for_dash = df_convert_for_multi_display(df, ['Time Period'])
	spread_range = 1
	dash_app.layout = html.Div([
		dcc.Graph(
			id='life-exp-vs-gdp',
			figure={
				'data': [
					go.Scatter(
						x=df.index,
						y=df_for_dash[df_for_dash['Spread_Names'] == i]['Spread_Values'],
						text=df_for_dash[df_for_dash['Spread_Names'] == i]['Spread_Values'],
						mode='lines',
						opacity=0.7,
						#fill='tozeroy',
						marker={
							'size': 15,
							'line': {'width': 0.5, 'color': 'white'}
						},
						name=i
					) for i in df_for_dash.Spread_Names.unique()
				],
				'layout': go.Layout(
					xaxis={'title': 'Dates'},
					yaxis={'title': 'Values'},
					margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
					legend={'x': 0, 'y': 1},
					hovermode='closest'
				)
			}
		)
	])
	return dash_app.layout

def get_df_from_data(topic):
	if topic == 'Interest Rates':
		text = text_dict[topic]
		df, spread_df = return_combine_df_for_graph()
		df_for_dash = df_convert_for_multi_display(df, ['Time Period'])
		df_spread_for_dash = df_convert_for_multi_display(spread_df, [])
		index = df.index
	elif topic == 'Inflation':
		text = text_dict[topic]
		df, spread_df = get_dfs_for_display_inflation()
		df_for_dash = df_convert_for_multi_display(df, ['Time Period'])
		df_spread_for_dash = df_convert_for_multi_display(spread_df, [])
		index = df.index
	elif topic == 'GDP':
		text = text_dict[topic]
		gdp_percent, gdp_change = get_gdp_graphs() 
		df_for_dash = df_convert_for_multi_display(gdp_percent, ['Time Period'])
		df_spread_for_dash = df_convert_for_multi_display(gdp_change, ['Time Period'])
		index = gdp_percent.index
	else:
		pass
	return index, df_for_dash, text, df_spread_for_dash


def build_interact_template(dash_app, index, df_for_dash, text, df_spread_for_dash, title, title2):
	markdown_text = text
	#df = return_combine_df_for_graph()
	#df_for_dash = df_convert_for_multi_display(df, ['Time Period'])
	spread_range = 1
	dash_app.layout = html.Div([
		dcc.Graph(
			id='values',
			figure={
				'data': [
					go.Scatter(
						x=index,
						y=df_for_dash[df_for_dash['Spread_Names'] == i]['Spread_Values'],
						text=df_for_dash[df_for_dash['Spread_Names'] == i]['Spread_Values'],
						mode='lines',
						opacity=0.7,
						#fill='tozeroy',
						marker={
							'size': 15,
							'line': {'width': 0.5, 'color': 'white'}
						},
						name=i
					) for i in df_for_dash.Spread_Names.unique()
				],
				'layout': go.Layout(
					title=title,
					xaxis={'title': 'Dates'},
					yaxis={'title': 'Values'},
					margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
					legend={'x':0, 'y': -1},
					hovermode='closest'
				)
			}
		),

		dcc.Graph(
			id='spreads',
			figure={
				'data': [
					go.Scatter(
						x=index,
						y=df_spread_for_dash[df_spread_for_dash['Spread_Names'] == i]['Spread_Values'],
						text=df_spread_for_dash[df_spread_for_dash['Spread_Names'] == i]['Spread_Values'],
						mode='lines',
						opacity=0.7,
						#fill='tozeroy',
						marker={
							'size': 15,
							'line': {'width': 0.5, 'color': 'white'}
						},
						name=i
					) for i in df_spread_for_dash.Spread_Names.unique()
				],
				'layout': go.Layout(
					title=title2,
					xaxis={'title': 'Dates'},
					yaxis={'title': 'Spread Values'},
					margin={'l': 40, 'b': 40, 't': 30, 'r': 10},
					legend={'x': 0, 'y': -1},
					hovermode='closest'
				)
			}
		),
	dcc.Markdown(children=markdown_text)
	])
	return dash_app.layout


def drop_down(dash_app):
	dash_app.layout = html.Div([
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='NYC'
    ),
    html.Div(id='output-container')
])
	return dash_app



def run_app(dash_app):
	pass
	#data = display_interactive_line_graph(dash_app)
	#return data

def run_app2(dash_app):
	pass
	#data = build_interact_template(dash_app)
	#return data

def run_ir_summary(dash_app):
	data = get_df_from_data('Interest Rates')
	# index, df_for_dash, text
	return_data = build_interact_template(dash_app, data[0], data[1], data[2], data[3], 'Inerest Rates', 'Inerest Rates Spreads')
	return return_data

def run_inflation_summary(dash_app):
	data = get_df_from_data('Inflation')
	# index, df_for_dash, text
	return_data = build_interact_template(dash_app, data[0], data[1], data[2], data[3], 'Inflation', 'Inflation Spreads')
	return return_data
	
def run_gdp_summary(dash_app):
	data = get_df_from_data('GDP')
	# index, df_for_dash, text
	return_data = build_interact_template(dash_app, data[0], data[1], data[2], data[3], 'GDP as percent', 'GDP as percent change')
	return return_data

#if __name__ == '__main__':
#	app.run_server(debug=True)


"""
resources 
https://stackoverflow.com/questions/18967441/add-a-prefix-to-all-flask-routes 
https://github.com/plotly/dash/issues/214
https://stackoverflow.com/questions/45845872/running-a-dash-app-within-a-flask-app
http://werkzeug.pocoo.org/docs/0.14/middlewares/#werkzeug.wsgi.DispatcherMiddleware 


also this page doesnt load idk why 

"""