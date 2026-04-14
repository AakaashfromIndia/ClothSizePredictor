import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Perfit", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,500;0,600;1,400;1,600&family=Jost:wght@300;400;500&display=swap');

*, html, body, [class*="css"] { font-family: 'Jost', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #FFF0F8 0%, #F5E6FF 30%, #EBE8FF 60%, #E4F0FF 100%);
    min-height: 100vh;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]  { display: none !important; }

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #C044A0 !important;
    border-color: #C044A0 !important;
}

.perfit-brand {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 7px;
    text-transform: uppercase;
    color: #A030A0;
    margin-bottom: 2px;
}

.perfit-title {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 3rem;
    font-weight: 600;
    line-height: 1.15;
    background: linear-gradient(120deg, #E0206A 0%, #8B35C5 50%, #2E7FD4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.perfit-tagline {
    font-size: 0.88rem;
    font-weight: 300;
    color: #9060A8;
    margin-top: 8px;
    letter-spacing: 0.5px;
    font-style: italic;
}

.slider-panel {
    margin-top: 20px;
}

.pred-card {
    background: linear-gradient(135deg, #FF2080 0%, #A020D0 50%, #2060E0 100%);
    border-radius: 26px;
    padding: 28px 0 24px 0;
    width: 100%;
    text-align: center;
    box-shadow: 0 12px 48px rgba(180,30,160,0.3), 0 3px 10px rgba(0,0,0,0.08);
    position: relative;
    overflow: hidden;
}

.pred-card::after {
    content: "";
    position: absolute;
    top: -30%; left: -10%;
    width: 150%; height: 160%;
    background: radial-gradient(ellipse at 65% 30%, rgba(255,255,255,0.2) 0%, transparent 55%);
    pointer-events: none;
}

.pred-size {
    font-family: 'Cormorant Garamond', serif;
    font-size: 5.5rem;
    font-weight: 600;
    color: #ffffff;
    line-height: 1;
    letter-spacing: 4px;
    text-shadow: 0 2px 24px rgba(0,0,0,0.15);
}

.pred-label {
    font-size: 0.66rem;
    color: rgba(255,255,255,0.78);
    font-weight: 400;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    margin-top: 10px;
}

.size-meaning {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #6020A0;
    text-align: center;
    margin: 12px 0 10px 0;
}

.size-regions {
    display: flex;
    gap: 10px;
    margin-top: 4px;
}

.region-pill {
    flex: 1;
    background: rgba(255,255,255,0.7);
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
    border: 1.5px solid rgba(200,140,220,0.35);
    backdrop-filter: blur(8px);
}

.region-flag {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #A060B0;
    display: block;
    margin-bottom: 3px;
    font-family: 'Jost', sans-serif;
}

.region-value {
    font-family: 'Jost', sans-serif;
    font-size: 1.05rem;
    font-weight: 500;
    color: #5010A0;
    display: block;
    letter-spacing: 0.5px;
}

.section-heading {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.2rem;
    font-weight: 600;
    color: #7030B0;
    margin: 28px 0 12px 0;
}

.soft-rule {
    border: none;
    border-top: 1px solid rgba(200,150,220,0.35);
    margin: 24px 0;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.65) !important;
    border: 1.5px solid rgba(210,155,230,0.4) !important;
    border-radius: 16px !important;
    box-shadow: 0 3px 14px rgba(160,80,200,0.1) !important;
    backdrop-filter: blur(8px);
}

[data-testid="metric-container"] label {
    font-size: 0.68rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
    color: #A060B0 !important;
    font-weight: 400 !important;
    font-family: 'Jost', sans-serif !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.9rem !important;
    color: #5010A0 !important;
    font-weight: 600 !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

SIZE_ORDER = ["XXS", "XS", "S", "M", "L", "XL", "XXL", "XXXL"]

SIZE_DATA = {
    "XXS":  ("Extra Extra Small", "4-6",   "32-34", "0"),
    "XS":   ("Extra Small",       "6-8",   "34-36", "0-2"),
    "S":    ("Small",             "8-10",  "36-38", "4-6"),
    "M":    ("Medium",            "10-12", "38-40", "8-10"),
    "L":    ("Large",             "12-14", "40-42", "12-14"),
    "XL":   ("Extra Large",       "14-16", "42-44", "16-18"),
    "XXL":  ("Extra Extra Large", "16-18", "44-46", "20-22"),
    "XXXL": ("3XL",               "18-20", "46-48", "24-26"),
}

SIZE_COLORS = {
    "XXS": "#FF2D78", "XS": "#C930E8", "S": "#6A40F0",
    "M":   "#0EA5E9", "L":  "#10C080", "XL": "#F59E0B",
    "XXL": "#F05000", "XXXL": "#E01050",
}

PLOT_PAPER = "rgba(0,0,0,0)"
PLOT_BG    = "rgba(255,255,255,0.55)"
GRID       = "rgba(210,170,230,0.28)"
FONT_COL   = "#6A4080"

LEGEND_STYLE = dict(
    bgcolor="rgba(255,255,255,0.75)",
    bordercolor="rgba(200,155,225,0.4)",
    borderwidth=1,
    font=dict(size=11, family="Jost"),
)

BASE = dict(
    paper_bgcolor=PLOT_PAPER,
    plot_bgcolor=PLOT_BG,
    font=dict(color=FONT_COL, family="Jost", size=12),
    margin=dict(l=36, r=16, t=46, b=32),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(200,155,225,0.25)"),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor="rgba(200,155,225,0.25)"),
    title_font=dict(size=13, color="#6A30A8", family="Cormorant Garamond"),
)


def hex_to_rgba(hex_color, alpha=0.1):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


@st.cache_data
def load_and_train():
    data = pd.read_csv(r"ClothesSizePrediction.csv")
    present = [s for s in SIZE_ORDER if s in data["size"].unique()]
    encoder = LabelEncoder()
    encoder.fit(present)
    data["size_encoded"] = encoder.transform(data["size"])
    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(data[["weight", "age", "height"]], data["size_encoded"])
    return data, clf, encoder, present


df, model, le, present_sizes = load_and_train()

col_hero, col_pred = st.columns([1.55, 1], gap="large")

with col_hero:
    st.markdown("""
        <div>
            <div class='perfit-brand'>Perfit</div>
            <div class='perfit-title'>Find your fit<br>before you try it.</div>
            <div class='perfit-tagline'>Move the sliders to instantly discover your size</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='slider-panel'>", unsafe_allow_html=True)
    weight = st.slider("Weight (kg)",
                       min_value=int(df["weight"].min()),
                       max_value=int(df["weight"].max()),
                       value=int(df["weight"].median()))
    age = st.slider("Age",
                    min_value=int(df["age"].min()),
                    max_value=int(df["age"].max()),
                    value=int(df["age"].median()))
    height = st.slider("Height (cm)",
                       min_value=float(df["height"].min()),
                       max_value=float(df["height"].max()),
                       value=float(df["height"].median()),
                       step=0.5)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Weight", f"{weight} kg")
    mc2.metric("Height", f"{height:.0f} cm")
    mc3.metric("Age",    f"{age} yrs")

pred_enc  = model.predict([[weight, age, height]])[0]
pred_size = le.inverse_transform([pred_enc])[0]
meaning, uk, eu, us = SIZE_DATA.get(pred_size, (pred_size, "–", "–", "–"))

with col_pred:
    st.markdown(f"""
        <div style='padding-top: 4px;'>
            <div class='pred-card'>
                <div class='pred-size'>{pred_size}</div>
                <div class='pred-label'>recommended size</div>
            </div>
            <div class='size-meaning'>{meaning}</div>
            <div class='size-regions'>
                <div class='region-pill'>
                    <span class='region-flag'>UK</span>
                    <span class='region-value'>{uk}</span>
                </div>
                <div class='region-pill'>
                    <span class='region-flag'>EU</span>
                    <span class='region-value'>{eu}</span>
                </div>
                <div class='region-pill'>
                    <span class='region-flag'>US</span>
                    <span class='region-value'>{us}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-heading'>How your size shifts</div>", unsafe_allow_html=True)

lc1, lc2, lc3 = st.columns(3)

w_range = np.linspace(df["weight"].min(), df["weight"].max(), 60)
h_range = np.linspace(df["height"].min(), df["height"].max(), 60)
a_range = np.linspace(df["age"].min(),    df["age"].max(),    60)


@st.cache_data
def predict_sensitivity_w(age_val, height_val):
    return le.inverse_transform(model.predict([[w, age_val, height_val] for w in w_range]))

@st.cache_data
def predict_sensitivity_h(weight_val, age_val):
    return le.inverse_transform(model.predict([[weight_val, age_val, h] for h in h_range]))

@st.cache_data
def predict_sensitivity_a(weight_val, height_val):
    return le.inverse_transform(model.predict([[weight_val, a, height_val] for a in a_range]))


p_w = predict_sensitivity_w(age, height)
p_h = predict_sensitivity_h(weight, age)
p_a = predict_sensitivity_a(weight, height)


def sensitivity_chart(x_vals, y_preds, x_mark, title, line_color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_preds,
        mode="lines",
        line=dict(color=line_color, width=2.5, shape="hv"),
        fill="tozeroy",
        fillcolor=hex_to_rgba(line_color, 0.1),
        showlegend=False,
    ))
    fig.add_vline(x=x_mark, line_dash="dot", line_color="#A030C0", line_width=1.8,
                  annotation_text="  you", annotation_font_color="#8020A0",
                  annotation_font_size=11)
    fig.update_layout(
        **BASE, title=title, height=220,
        yaxis_categoryorder="array",
        yaxis_categoryarray=present_sizes,
        legend=LEGEND_STYLE,
    )
    return fig


with lc1:
    st.plotly_chart(
        sensitivity_chart(w_range, p_w, weight, "Size as weight changes", "#FF2D78"),
        use_container_width=True,
    )
with lc2:
    st.plotly_chart(
        sensitivity_chart(h_range, p_h, height, "Size as height changes", "#C930E8"),
        use_container_width=True,
    )
with lc3:
    st.plotly_chart(
        sensitivity_chart(a_range, p_a, age, "Size as age changes", "#6A40F0"),
        use_container_width=True,
    )

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)
st.markdown("<div class='section-heading'>Explore the data</div>", unsafe_allow_html=True)

ea, eb = st.columns([1.4, 1], gap="large")

with ea:
    scatter = px.scatter(
        df, x="height", y="weight", color="size",
        color_discrete_map=SIZE_COLORS,
        category_orders={"size": present_sizes},
        title="Where do you land?",
        labels={"height": "Height (cm)", "weight": "Weight (kg)", "size": "Size"},
        opacity=0.82,
    )
    scatter.add_trace(go.Scatter(
        x=[height], y=[weight], mode="markers",
        marker=dict(color="#FF1060", size=16, symbol="star",
                    line=dict(color="white", width=2)),
        name="You", showlegend=True,
    ))
    scatter.update_layout(**BASE, height=340, legend=LEGEND_STYLE)
    st.plotly_chart(scatter, use_container_width=True)

with eb:
    size_counts = df["size"].value_counts().reindex(present_sizes).dropna()
    donut = go.Figure(go.Pie(
        labels=size_counts.index,
        values=size_counts.values,
        hole=0.58,
        marker_colors=[SIZE_COLORS.get(s, "#ccc") for s in size_counts.index],
        textinfo="label+percent",
        textfont=dict(size=11, family="Jost"),
        pull=[0.07 if s == pred_size else 0 for s in size_counts.index],
    ))
    donut.update_layout(
        **BASE, title="How common is your size?",
        height=340, showlegend=False, legend=LEGEND_STYLE,
        annotations=[dict(text=pred_size, x=0.5, y=0.5, showarrow=False,
                          font=dict(size=22, color="#8020C0",
                                    family="Cormorant Garamond"))],
    )
    st.plotly_chart(donut, use_container_width=True)

ec, ed = st.columns(2, gap="large")

with ec:
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    bmi_fig = px.strip(
        df, x="size", y="bmi", color="size",
        color_discrete_map=SIZE_COLORS,
        category_orders={"size": present_sizes},
        title="BMI range per size",
        labels={"bmi": "BMI", "size": "Size"},
        stripmode="overlay",
    )
    user_bmi = weight / ((height / 100) ** 2)
    bmi_fig.add_hline(y=user_bmi, line_dash="dot", line_color="#FF2D78", line_width=2,
                      annotation_text=f"  your BMI {user_bmi:.1f}",
                      annotation_font_color="#CC1060", annotation_font_size=11)
    bmi_fig.update_traces(marker_size=7, marker_opacity=0.65)
    bmi_fig.update_layout(**BASE, showlegend=False, height=300, legend=LEGEND_STYLE)
    st.plotly_chart(bmi_fig, use_container_width=True)

with ed:
    age_avg = df.groupby("size")["age"].mean().reindex(present_sizes).dropna().reset_index()
    bar_fig = px.bar(
        age_avg, x="size", y="age", color="size",
        color_discrete_map=SIZE_COLORS,
        title="Average age per size",
        labels={"age": "Average Age", "size": "Size"},
        text_auto=".0f",
    )
    bar_fig.add_hline(y=age, line_dash="dot", line_color="#6A40F0", line_width=2,
                      annotation_text=f"  your age {age}",
                      annotation_font_color="#5030D0", annotation_font_size=11)
    bar_fig.update_traces(marker_line_width=0, textposition="outside",
                          textfont=dict(size=10, color="#7A4080"))
    bar_fig.update_layout(**BASE, showlegend=False, height=300, legend=LEGEND_STYLE)
    st.plotly_chart(bar_fig, use_container_width=True)

ee, ef = st.columns(2, gap="large")

with ee:
    violin = go.Figure()
    for size in present_sizes:
        sub = df[df["size"] == size]
        if len(sub) < 2:
            continue
        violin.add_trace(go.Violin(
            x=sub["size"], y=sub["height"],
            name=size,
            fillcolor=SIZE_COLORS.get(size, "#ccc"),
            line_color=SIZE_COLORS.get(size, "#ccc"),
            opacity=0.70, box_visible=True,
            meanline_visible=True, showlegend=False,
        ))
    violin.add_hline(y=height, line_dash="dot", line_color="#C930E8", line_width=2,
                     annotation_text=f"  your height {height:.0f}cm",
                     annotation_font_color="#A020C0", annotation_font_size=11)
    violin.update_layout(**BASE, title="Height distribution per size",
                         height=300, legend=LEGEND_STYLE)
    st.plotly_chart(violin, use_container_width=True)

with ef:
    weight_avg = df.groupby("size")["weight"].mean().reindex(present_sizes).dropna().reset_index()
    h_bar = px.bar(
        weight_avg, x="size", y="weight", color="size",
        color_discrete_map=SIZE_COLORS,
        title="Average weight per size",
        labels={"weight": "Average Weight (kg)", "size": "Size"},
        text_auto=".0f",
    )
    h_bar.add_hline(y=weight, line_dash="dot", line_color="#FF2D78", line_width=2,
                    annotation_text=f"  your weight {weight}kg",
                    annotation_font_color="#CC1060", annotation_font_size=11)
    h_bar.update_traces(marker_line_width=0, textposition="outside",
                        textfont=dict(size=10, color="#7A4080"))
    h_bar.update_layout(**BASE, showlegend=False, height=300, legend=LEGEND_STYLE)
    st.plotly_chart(h_bar, use_container_width=True)

st.markdown("""
    <div style='text-align:center; padding: 30px 0 14px 0;
                font-family: Cormorant Garamond, serif; font-style:italic;
                font-size: 0.88rem; color: #B080C0; letter-spacing: 0.5px;'>
        Perfit - find your fit before you try it
    </div>
""", unsafe_allow_html=True)
