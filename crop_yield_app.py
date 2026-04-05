import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go

st.set_page_config(
    page_title="AgroAI · Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=DM+Sans:wght@400;500&family=DM+Mono:wght@500&display=swap');
  
  :root {
    --primary-color: #2e7d32;
    --background-color: #f7f6f2;
    --secondary-background-color: #ffffff;
    --text-color: #1a2e1a;
    --font: 'DM Sans', sans-serif;
  }

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1a2e1a; }
  .stApp { background: #f7f6f2; }
  
    div[data-testid="stWidgetLabel"],
    div[data-testid="stWidgetLabel"] * {
        color: #1a2e1a !important;
        opacity: 1 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    label {
        color: #1a2e1a !important;
        opacity: 1 !important;
    }

  .stSelectbox div[data-baseweb="select"], 
  .stNumberInput input, 
  .stSlider div[data-baseweb="slider"] {
      color: #1a2e1a !important;
      background-color: #ffffff !important;
  }
  
  .app-header {
      display: flex; align-items: center; gap: 14px;
      padding: 28px 0 20px; border-bottom: 1px solid #e0ddd5; margin-bottom: 28px;
  }
  .app-header-icon { font-size: 32px; line-height: 1; }
  .app-header h1 {
      font-family: 'Playfair Display', serif !important;
      font-size: 26px !important; font-weight: 600 !important;
      color: #1a2e1a !important; margin: 0 !important; padding: 0 !important;
  }
  .app-header p { font-size: 13px; color: #7a7a70; margin: 2px 0 0; }
  
  .model-badge {
      margin-left: auto; font-size: 11px; font-family: 'DM Mono', monospace;
      padding: 4px 12px; border-radius: 999px;
      background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9;
  }
  
  .card { background: #fff; border: 1px solid #e8e5de; border-radius: 14px; padding: 22px 24px; color: #1a2e1a; }
  .card-title {
      font-size: 11px; font-weight: 500; letter-spacing: 0.08em;
      text-transform: uppercase; color: #9a9a8e; margin-bottom: 18px;
  }
  
  .result-box { background: #fff; border: 1px solid #e8e5de; border-radius: 14px; padding: 24px; color: #1a2e1a; }
  .result-label {
      font-size: 11px; font-weight: 500; letter-spacing: 0.08em;
      text-transform: uppercase; color: #9a9a8e; margin-bottom: 6px;
  }
  .result-number {
      font-family: 'Playfair Display', serif;
      font-size: 52px; font-weight: 600; color: #2e7d32; line-height: 1;
  }
  .result-unit { font-family: 'DM Mono', monospace; font-size: 16px; color: #7a7a70; margin-left: 8px; }
  
  .metric-mini { flex: 1; background: #f7f6f2; border-radius: 10px; padding: 12px 14px; }
  .metric-mini .mv { font-family: 'DM Mono', monospace; font-size: 15px; font-weight: 500; color: #1a2e1a; }
  
  .section-title {
      font-family: 'Playfair Display', serif;
      font-size: 17px; font-weight: 500; color: #1a2e1a;
      margin: 32px 0 16px; border-bottom: 1px solid #e8e5de; padding-bottom: 10px;
  }

  .stButton > button {
      background: #2e7d32 !important; color: #fff !important;
      border: none !important; border-radius: 10px !important;
      padding: 12px 24px !important; font-size: 14px !important; font-weight: 500 !important; width: 100% !important;
  }
  
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Color constants ────────────────────────────────────────────────────────────
G_DARK  = "#1b5e20"
G_MID   = "#2e7d32"
G_LIGHT = "#c8e6c9"
G_PALE  = "#e8f5e9"
AMBER   = "#f59e0b"
CORAL   = "#ef4444"
INDIGO  = "#6366f1"
GRID    = "#f0ede5"
FONT    = dict(family="DM Sans, sans-serif", color="#4a4a40")

def base_layout(t=36, b=20, l=10, r=10, **kw):
    return dict(
        paper_bgcolor="#ffffff",  
        plot_bgcolor="#ffffff",    
        font=dict(
            family="DM Sans, sans-serif",
            color="#1a2e1a",  
            size=13
        ),
        margin=dict(t=t, b=b, l=l, r=r),
        **kw
    )

def conf_color(c):
    return G_MID if c >= 75 else (AMBER if c >= 50 else CORAL)

# ── Data & model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train_model():
    data = {
        'Year': [2019,2018,2017,2016,2015,2014,2013,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,
                 2021,2020,2016,2015,2014,2013,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,
                 2021,2020,2019,2018,2017,2016,2015,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000],
        'State': ['Punjab'] * 55,
        'Crop_Type': ['Wheat']*19 + ['Rice']*18 + ['Bajra']*18,
        'Rainfall': [578.6,598.3,493.0,426.7,546.9,384.9,619.7,218.9,472.1,384.9,529.2,438.0,418.3,565.9,375.2,
                     459.5,314.5,662.8,391.9,556.9,602.6,426.7,546.9,384.9,619.7,218.9,472.1,384.9,529.2,438.0,
                     418.3,565.9,375.2,459.5,314.5,662.8,391.9,556.9,602.6,578.6,598.3,493.0,426.7,546.9,
                     472.1,384.9,529.2,438.0,418.3,565.9,375.2,459.5,314.5,662.8,391.9],
        'Soil_Type': ['Loamy']*19 + ['alluvial']*18 + ['Loamy']*18,
        'Irrigation_Area': [3515.2,3499.3,3467.7,3474.6,3474.7,3474.7,3488.1,3466.9,3474.8,3474.8,3437.9,3406.9,
                            3404.8,3410.5,3381.7,3311.6,3353.5,3333.6,3284.3,3229.5,3118.8,2961.4,2838.3,2838.3,
                            2837.6,2814.2,2721.8,2721.8,2592.2,2602.4,2639.9,2632.3,2599.6,2515.7,2471.0,2590.3,
                            2584.7,3.9,2.0,1.9,2.8,3.1,1.9,1.2,4.9,4.9,3.5,4.2,5.2,5.6,7.2,6.1,7.6,5.4,4.6],
        'Crop_Yield': [5188,5077,5046,4583,4304,5017,4724,4693,4307,4462,4507,4210,4179,4221,4207,4200,4532,4563,
                       4696,4443,4034,3974,3838,3952,3998,3828,4010,4022,4019,3868,3858,3943,3694,3510,3545,3506,
                       3347,40,635,583,597,580,0,0,1495,1055,950,977,1045,978,993,810,929,893,703]
    }
    df = pd.DataFrame(data)
    raw = df.copy()
    encoders = {}
    for col in ['State', 'Crop_Type', 'Soil_Type']:
        le = LabelEncoder().fit(df[col])
        df[col] = le.transform(df[col])
        encoders[col] = le
    scaler = StandardScaler().fit(df[['Rainfall', 'Irrigation_Area']])
    df[['Rainfall', 'Irrigation_Area']] = scaler.transform(df[['Rainfall', 'Irrigation_Area']])
    X = df[['Year', 'State', 'Crop_Type', 'Rainfall', 'Soil_Type', 'Irrigation_Area']]
    mdl = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=42).fit(X, df['Crop_Yield'])
    return mdl, encoders, scaler, raw

model, encoders, scaler, raw_data = load_and_train_model()
FEAT_NAMES = ['Year', 'State', 'Crop Type', 'Rainfall', 'Soil Type', 'Irrigation Area']

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-header-icon">🌾</div>
  <div>
    <h1>AgroAI — Crop Yield Predictor</h1>
    <p>RandomForest regression · Punjab dataset · Innovative AI Challenge 2024</p>
  </div>
  <div class="model-badge">Model ready</div>
</div>
""", unsafe_allow_html=True)

# ── Form + Result ──────────────────────────────────────────────────────────────
c_form, c_res = st.columns([1, 1.5], gap="large")

with c_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Parameters</div>', unsafe_allow_html=True)
    state = st.selectbox("Region", encoders['State'].classes_)
    crop  = st.selectbox("Crop type", encoders['Crop_Type'].classes_)
    soil  = st.selectbox("Soil type", encoders['Soil_Type'].classes_)
    st.markdown("<hr style='border:none;border-top:1px solid #e8e5de;margin:14px 0'>", unsafe_allow_html=True)
    rain = st.slider("Rainfall (mm/yr)", 0, 1000, 500)
    irr  = st.slider("Irrigation area (ha)", 0, 4000, 2000)
    year = st.number_input("Forecast year", 2024, 2030, 2025)
    run  = st.button("Calculate yield →")
    st.markdown('</div>', unsafe_allow_html=True)

with c_res:
    if run:
        s_e  = encoders['State'].transform([state])[0]
        c_e  = encoders['Crop_Type'].transform([crop])[0]
        so_e = encoders['Soil_Type'].transform([soil])[0]
        sc   = scaler.transform([[rain, irr]])
        Xin  = [[year, s_e, c_e, sc[0][0], so_e, sc[0][1]]]

        result = int(model.predict(Xin)[0])

        tree_preds = np.array([t.predict(Xin)[0] for t in model.estimators_])
        std  = float(tree_preds.std())
        mean = float(tree_preds.mean())
        cov  = std / abs(mean) if mean != 0 else 1.0
        conf = round(max(0.0, min(100.0, (1 - cov) * 100)), 1)
        low, high = int(result - std), int(result + std)

        avg  = int(raw_data[raw_data['Crop_Type'] == crop]['Crop_Yield'].mean())
        diff = result - avg
        sign = "+" if diff >= 0 else ""
        dc   = G_MID if diff >= 0 else CORAL
        cc   = conf_color(conf)

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted yield · {year}</div>
            <div>
                <span class="result-number">{result:,}</span>
                <span class="result-unit">kg/ha</span>
            </div>
            <div class="result-delta">
                <span style="color:{dc};font-weight:500">{sign}{diff:,} kg/ha</span>
                &nbsp;vs. {crop} average ({avg:,} kg/ha)
            </div>
            <div class="metrics-row">
                <div class="metric-mini"><div class="ml">Crop</div><div class="mv">{crop}</div></div>
                <div class="metric-mini"><div class="ml">Rainfall</div><div class="mv">{rain} mm</div></div>
                <div class="metric-mini"><div class="ml">Irrigation</div><div class="mv">{irr} ha</div></div>
            </div>
            <div class="conf-wrap">
                <div class="conf-header">
                    <span class="conf-label">Model confidence</span>
                    <span class="conf-pct" style="color:{cc}">{conf}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf}%;background:{cc}"></div>
                </div>
                <div class="conf-range">
                    Range: {low:,} – {high:,} kg/ha &nbsp;·&nbsp; ±{int(std):,} kg/ha (1σ across 200 trees)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for k, v in dict(result=result, conf=conf, std=std, low=low, high=high,
                         crop=crop, rain=rain, irr=irr, year=year,
                         tree_preds=tree_preds, Xin=Xin, predicted=True).items():
            st.session_state[k] = v
    else:
        if not st.session_state.get('predicted'):
            st.markdown("""
            <div class="result-box">
                <div class="empty-state">
                    <div class="ei">🌱</div>
                    <div>Set parameters and click<br><strong>Calculate yield</strong></div>
                </div>
            </div>""", unsafe_allow_html=True)

# ── Analytics section ──────────────────────────────────────────────────────────
if st.session_state.get('predicted'):
    R  = st.session_state
    result     = R['result']
    conf       = R['conf']
    std        = R['std']
    low        = R['low']
    high       = R['high']
    crop       = R['crop']
    rain       = R['rain']
    irr        = R['irr']
    year       = R['year']
    tree_preds = R['tree_preds']
    cc         = conf_color(conf)

    st.markdown('<div class="section-title">Analytics & Visualizations</div>', unsafe_allow_html=True)

    # ── Row 1: Crop comparison  |  Tree distribution ───────────────────────────
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        avg_df = raw_data.groupby('Crop_Type')['Crop_Yield'].mean().reset_index()
        names  = list(avg_df['Crop_Type']) + ['Forecast']
        vals   = list(avg_df['Crop_Yield'].round(0).astype(int)) + [result]
        colors = [G_LIGHT if c != crop else G_MID for c in avg_df['Crop_Type']] + [G_DARK]
        errs   = [None] * len(avg_df) + [int(std)]

        fig = go.Figure(go.Bar(
            x=names, y=vals, marker_color=colors,
            text=[f"{v:,}" for v in vals], textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=11, color='#4a4a40'),
            error_y=dict(type='data', array=errs, visible=True, color=G_DARK, thickness=1.5),
        ))
        fig.update_layout(**base_layout(height=300, title="Yield by crop type"),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showticklabels=False),
            xaxis=dict(showgrid=False, title=None, tickfont=dict(size=12)),
            bargap=0.35)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=tree_preds, nbinsx=28,
            marker_color=G_LIGHT, marker_line=dict(color=G_MID, width=0.5),
        ))
        fig.add_vline(x=result, line_color=G_DARK, line_width=2,
                      annotation_text=f"Mean {result:,}",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color=G_DARK))
        fig.add_vrect(x0=low, x1=high, fillcolor=G_MID, opacity=0.1, line_width=0,
                      annotation_text="±1σ", annotation_position="top left",
                      annotation_font=dict(size=10, color=G_MID))
        fig.update_layout(**base_layout(height=300, title="Uncertainty — distribution across 200 trees"),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title=None),
            xaxis=dict(showgrid=False, title="Predicted kg/ha", tickfont=dict(size=11)),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Historical trend  |  Rainfall vs Yield ─────────────────────────
    col3, col4 = st.columns(2, gap="medium")

    with col3:
        hist = raw_data[raw_data['Crop_Type'] == crop].sort_values('Year')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist['Year'], y=hist['Crop_Yield'],
            mode='lines+markers',
            line=dict(color=G_MID, width=2),
            marker=dict(size=6, color=G_MID, line=dict(color='white', width=1.5)),
            name='Historical', fill='tozeroy', fillcolor='rgba(46,125,50,0.07)',
        ))
        fig.add_trace(go.Scatter(
            x=[year], y=[result], mode='markers',
            marker=dict(size=14, color=G_DARK, symbol='star',
                        line=dict(color='white', width=2)),
            name=f'Forecast {year}',
            error_y=dict(type='data', array=[int(std)], visible=True,
                         color=G_DARK, thickness=1.5, width=6),
        ))
        fig.update_layout(**base_layout(height=300, title=f"{crop} — historical yield & forecast"),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="kg/ha", tickfont=dict(size=11)),
            xaxis=dict(showgrid=False, title=None, tickfont=dict(size=11)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        cmap = {'Wheat': G_MID, 'Rice': AMBER, 'Bajra': INDIGO}
        fig = go.Figure()
        for c in raw_data['Crop_Type'].unique():
            sub = raw_data[raw_data['Crop_Type'] == c]
            fig.add_trace(go.Scatter(
                x=sub['Rainfall'], y=sub['Crop_Yield'], mode='markers', name=c,
                marker=dict(size=8, color=cmap.get(c, '#999'),
                            opacity=0.7, line=dict(color='white', width=1)),
            ))
        fig.add_trace(go.Scatter(
            x=[rain], y=[result], mode='markers', name='Your input',
            marker=dict(size=16, color=G_DARK, symbol='star',
                        line=dict(color='white', width=2)),
        ))
        fig.update_layout(**base_layout(height=300, title="Rainfall vs yield by crop"),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="kg/ha", tickfont=dict(size=11)),
            xaxis=dict(showgrid=False, title="Rainfall (mm/yr)", tickfont=dict(size=11)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Feature importance  |  Confidence gauge ────────────────────────
    col5, col6 = st.columns(2, gap="medium")

    with col5:
        imp_df = pd.DataFrame({'Feature': FEAT_NAMES, 'Importance': model.feature_importances_})
        imp_df = imp_df.sort_values('Importance')
        bc = [G_MID if f in ['Irrigation Area', 'Year'] else G_LIGHT for f in imp_df['Feature']]

        fig = go.Figure(go.Bar(
            x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
            marker_color=bc, marker_line=dict(color=G_MID, width=0.5),
            text=[f"{v:.1%}" for v in imp_df['Importance']],
            textposition='outside',
            textfont=dict(family='DM Mono, monospace', size=11, color='#4a4a40'),
        ))
        fig.update_layout(**base_layout(height=300, title="Feature importance"),
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, showticklabels=False, title=None),
            yaxis=dict(showgrid=False, title=None, tickfont=dict(size=12)),
            bargap=0.3)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        steps = [
            dict(range=[0,  50], color="#fee2e2"),
            dict(range=[50, 75], color="#fef3c7"),
            dict(range=[75,100], color="#dcfce7"),
        ]
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=conf,
            number=dict(suffix="%", font=dict(size=40, family="DM Mono, monospace", color=cc)),
            delta=dict(reference=75, suffix="%",
                       increasing=dict(color=G_MID), decreasing=dict(color=CORAL)),
            title=dict(text="Model confidence", font=dict(size=13, color="#9a9a8e")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=0, tickcolor="white",
                          tickfont=dict(size=11), nticks=6),
                bar=dict(color=cc, thickness=0.25),
                bgcolor="white", steps=steps,
                threshold=dict(line=dict(color="#1a2e1a", width=2), thickness=0.8, value=75),
            ),
        ))
        # ↓ margin passed via t/b/l/r — no conflict
        fig.update_layout(**base_layout(height=300, t=40, b=10, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: Heatmap ────────────────────────────────────────────────────────
    crop_hist = raw_data[raw_data['Crop_Type'] == crop].copy()
    crop_hist['Rain_bucket'] = pd.cut(
        crop_hist['Rainfall'], bins=4, labels=['Low', 'Medium', 'High', 'Very High']
    )
    pivot = crop_hist.pivot_table(
        index='Rain_bucket', columns='Year', values='Crop_Yield', aggfunc='mean'
    ).fillna(0).astype(int)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        colorscale=[[0, G_PALE], [0.5, G_LIGHT], [1, G_DARK]],
        text=pivot.values,
        texttemplate="%{text:,}",
        textfont=dict(size=10, family="DM Mono, monospace"),
        hovertemplate="Year: %{x}<br>Rainfall: %{y}<br>Yield: %{z:,} kg/ha<extra></extra>",
        showscale=True,
        colorbar=dict(title="kg/ha", tickfont=dict(size=10)),
    ))
    # ↓ margin passed via t/b/l/r — no conflict
    fig.update_layout(**base_layout(height=270, t=40, b=10, l=80, r=10,
                                    title=f"{crop} — yield heatmap (year × rainfall level)"),
        xaxis=dict(showgrid=False, title=None, tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, title=None, tickfont=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 5: Irrigation vs yield  |  Box plot ────────────────────────────────
    col7, col8 = st.columns(2, gap="medium")

    with col7:
        irr_df = raw_data[raw_data['Crop_Type'] == crop].sort_values('Irrigation_Area')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=irr_df['Irrigation_Area'], y=irr_df['Crop_Yield'],
            mode='markers+lines',
            marker=dict(size=7, color=G_MID, line=dict(color='white', width=1)),
            line=dict(color=G_LIGHT, width=1.5, dash='dot'),
            name='Data',
        ))
        fig.add_trace(go.Scatter(
            x=[irr], y=[result], mode='markers', name='Your input',
            marker=dict(size=14, color=G_DARK, symbol='star',
                        line=dict(color='white', width=2)),
        ))
        fig.update_layout(**base_layout(height=280, title=f"{crop} — irrigation area vs yield"),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="kg/ha", tickfont=dict(size=11)),
            xaxis=dict(showgrid=False, title="Irrigation area (ha)", tickfont=dict(size=11)),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        bcolors = {'Wheat': G_MID, 'Rice': AMBER, 'Bajra': INDIGO}
        fig = go.Figure()
        for c in raw_data['Crop_Type'].unique():
            sub = raw_data[raw_data['Crop_Type'] == c]
            fig.add_trace(go.Box(
                y=sub['Crop_Yield'], name=c,
                marker_color=bcolors.get(c, G_LIGHT),
                boxmean='sd',
                line=dict(width=1.5),
            ))
        fig.update_layout(**base_layout(height=280, title="Yield distribution by crop (box plot)"),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, title="kg/ha", tickfont=dict(size=11)),
            xaxis=dict(showgrid=False, title=None, tickfont=dict(size=12)),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    ADA-2403M: Бисимбаева Аружан, Дуйсен Жанерке, Казагашева Валерия, Жумагулова Карина, Шаймуран Алишер
    &nbsp;·&nbsp; Astana IT University &nbsp;·&nbsp; RandomForestRegressor ensemble
</div>
""", unsafe_allow_html=True)
