from flask import Flask, render_template
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo

app = Flask(__name__)

@app.route('/')
def dashboard():
    df = pd.read_csv('data/stress_predictions.csv')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['current_stress'], mode='lines+markers', name='Current Stress'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted_stress'], mode='lines', name='Predicted Stress'))
    chart_html = pyo.plot(fig, output_type='div')
    return render_template('dashboard.html', chart=chart_html)
