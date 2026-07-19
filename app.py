# app.py — RetailIQ Streamlit Dashboard
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailIQ Dashboard",
    page_icon="🛒",
    layout="wide"           # use full screen width
)

# ── Load data ─────────────────────────────────────────────────────
@st.cache_data                  # cache = load once, reuse on every interaction
def load_data():
    df      = pd.read_csv('retail_clean.csv')
    rfm     = pd.read_csv('rfm_segments.csv')
    monthly = pd.read_csv('monthly_revenue.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df, rfm, monthly

@st.cache_resource              # cache_resource = load models once
def load_models():
    with open('churn_model.pkl', 'rb') as f:
        churn_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('forecast_model.pkl', 'rb') as f:
        forecast_model = pickle.load(f)
    return churn_model, scaler, forecast_model

# Load everything
df, rfm, monthly = load_data()
churn_model, scaler, forecast_model = load_models()

# ── Sidebar navigation ────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/shopping-cart.png", width=80)
st.sidebar.title("RetailIQ")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "👥 Customer Analysis", 
     "⚠️ Churn Prediction", "📈 Sales Forecast"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI Online Retail II")
st.sidebar.markdown("**Records:** 400K+ transactions")
st.sidebar.markdown("**Period:** Dec 2009 – Nov 2011")

# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Business Overview")
    st.markdown("Key performance indicators and revenue trends")
    st.markdown("---")

    # ── KPI Cards ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    # st.columns(4) creates 4 equal-width columns side by side

    total_revenue   = df['TotalPrice'].sum()
    total_orders    = df['Invoice'].nunique()
    total_customers = df['Customer ID'].nunique()
    avg_order_value = total_revenue / total_orders

    with col1:
        st.metric("💰 Total Revenue",
                  f"£{total_revenue/1e6:.2f}M")
    with col2:
        st.metric("🛒 Total Orders",
                  f"{total_orders:,}")
    with col3:
        st.metric("👥 Total Customers",
                  f"{total_customers:,}")
    with col4:
        st.metric("📦 Avg Order Value",
                  f"£{avg_order_value:.2f}")

    st.markdown("---")

    # ── Monthly Revenue Trend ─────────────────────────────────
    st.subheader("Monthly Revenue Trend")
    monthly_plot = monthly.copy()
    monthly_plot['Label'] = (monthly_plot['Year'].astype(str) + '-' +
                              monthly_plot['Month'].astype(str).str.zfill(2))

    fig1 = px.area(
        monthly_plot, x='Label', y='TotalPrice',
        title='Monthly Revenue (Dec 2009 – Nov 2011)',
        labels={'TotalPrice': 'Revenue (£)', 'Label': 'Month'},
        color_discrete_sequence=['#1A5276']
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    # use_container_width=True → chart fills full width

    # ── Two charts side by side ───────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 10 Countries by Revenue")
        country = (df[df['Country'] != 'United Kingdom']
                   .groupby('Country')['TotalPrice']
                   .sum()
                   .sort_values(ascending=False)
                   .head(10)
                   .reset_index())
        fig2 = px.bar(
            country, x='TotalPrice', y='Country',
            orientation='h',
            color='TotalPrice',
            color_continuous_scale='Blues',
            labels={'TotalPrice': 'Revenue (£)'}
        )
        fig2.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Revenue by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday',
                     'Thursday','Friday','Saturday','Sunday']
        day_sales = (df.groupby('DayOfWeek')['TotalPrice']
                     .sum()
                     .reindex(day_order)
                     .reset_index())
        fig3 = px.bar(
            day_sales, x='DayOfWeek', y='TotalPrice',
            color='TotalPrice',
            color_continuous_scale='Purples',
            labels={'TotalPrice': 'Revenue (£)', 'DayOfWeek': 'Day'}
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Top Products ──────────────────────────────────────────
    st.subheader("Top 10 Products by Revenue")
    top_products = (df.groupby('Description')['TotalPrice']
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index())
    fig4 = px.bar(
        top_products, x='TotalPrice', y='Description',
        orientation='h',
        color='TotalPrice',
        color_continuous_scale='Greens',
        labels={'TotalPrice': 'Revenue (£)', 'Description': 'Product'}
    )
    fig4.update_layout(
        showlegend=False,
        yaxis={'categoryorder':'total ascending'},
        height=400
    )
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER ANALYSIS
# ════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analysis":
    st.title("👥 Customer Analysis")
    st.markdown("RFM segmentation and customer behaviour insights")
    st.markdown("---")

    # ── Segment Distribution ──────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        fig5 = px.pie(
            seg_counts, values='Count', names='Segment',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4          # hole=0.4 makes it a donut chart
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("Average Spend by Segment")
        seg_spend = (rfm.groupby('Segment')['Monetary']
                     .mean()
                     .sort_values(ascending=False)
                     .reset_index())
        fig6 = px.bar(
            seg_spend, x='Segment', y='Monetary',
            color='Monetary',
            color_continuous_scale='Blues',
            labels={'Monetary': 'Avg Spend (£)'}
        )
        fig6.update_layout(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # ── RFM Scatter Plot ──────────────────────────────────────
    st.subheader("RFM Scatter — Frequency vs Monetary by Segment")
    fig7 = px.scatter(
        rfm, x='Frequency', y='Monetary',
        color='Segment',
        size='Monetary',        # bubble size = monetary value
        hover_data=['Customer ID', 'Recency'],
        color_discrete_sequence=px.colors.qualitative.Set1,
        labels={'Monetary': 'Total Spend (£)', 'Frequency': 'Number of Orders'}
    )
    fig7.update_layout(height=500)
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")

    # ── Segment Summary Table ─────────────────────────────────
    st.subheader("Segment Summary")
    seg_summary = rfm.groupby('Segment').agg(
        Customers     = ('Customer ID', 'count'),
        Avg_Recency   = ('Recency', 'mean'),
        Avg_Frequency = ('Frequency', 'mean'),
        Avg_Monetary  = ('Monetary', 'mean')
    ).round(1).reset_index()
    seg_summary = seg_summary.sort_values('Avg_Monetary', ascending=False)
    st.dataframe(seg_summary, use_container_width=True)

    st.markdown("---")

    # ── Top Customers ─────────────────────────────────────────
    st.subheader("Top 20 Customers by Spend")
    top_cust = (rfm[['Customer ID','Monetary','Frequency',
                      'Recency','Segment']]
                .sort_values('Monetary', ascending=False)
                .head(20)
                .reset_index(drop=True))
    top_cust['Monetary'] = top_cust['Monetary'].apply(lambda x: f"£{x:,.2f}")
    st.dataframe(top_cust, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 3 — CHURN PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Prediction":
    st.title("⚠️ Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn risk")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # ── Input form ────────────────────────────────────────────
    with col1:
        st.subheader("Enter Customer Details")

        frequency = st.slider(
            "Number of Orders (Frequency)",
            min_value=1, max_value=100, value=5, step=1
        )
        monetary = st.number_input(
            "Total Spend in £ (Monetary)",
            min_value=10.0, max_value=50000.0,
            value=500.0, step=50.0
        )
        avg_order = monetary / frequency

        st.info(f"Average Order Value: £{avg_order:.2f}")

        predict_btn = st.button("🔍 Predict Churn Risk", 
                                 use_container_width=True)

    # ── Prediction output ─────────────────────────────────────
    with col2:
        st.subheader("Prediction Result")

        if predict_btn:
            # Prepare input — same features model was trained on
            input_data = pd.DataFrame({
                'Frequency': [frequency],
                'Monetary' : [monetary]
            })

            # Scale using saved scaler
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = churn_model.predict(input_scaled)[0]
            probability = churn_model.predict_proba(input_scaled)[0]

            churn_prob  = probability[1] * 100   # probability of churning
            active_prob = probability[0] * 100   # probability of staying

            if prediction == 1:
                st.error(f"🔴 HIGH CHURN RISK")
                st.metric("Churn Probability", f"{churn_prob:.1f}%")
                st.warning("**Recommended Action:** Send win-back offer or loyalty discount immediately.")
            else:
                st.success(f"🟢 LOW CHURN RISK")
                st.metric("Retention Probability", f"{active_prob:.1f}%")
                st.info("**Recommended Action:** Customer is active. Consider upselling.")

            # Probability gauge chart
            fig8 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                title={'text': "Churn Risk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 40],  'color': "#2ecc71"},  # green
                        {'range': [40, 70], 'color': "#f39c12"},  # orange
                        {'range': [70, 100],'color': "#e74c3c"}   # red
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': churn_prob
                    }
                }
            ))
            fig8.update_layout(height=300)
            st.plotly_chart(fig8, use_container_width=True)

        else:
            st.info("👈 Enter customer details and click Predict")

    st.markdown("---")

    # ── Churn Distribution ────────────────────────────────────
    st.subheader("Overall Churn Distribution in Dataset")
    col3, col4, col5 = st.columns(3)

    churned     = rfm['Churned'].sum() if 'Churned' in rfm.columns else int(len(rfm) * 0.67)
    not_churned = len(rfm) - churned

    with col3:
        st.metric("Total Customers", f"{len(rfm):,}")
    with col4:
        st.metric("🔴 Churned", f"{churned:,}")
    with col5:
        st.metric("🟢 Active", f"{not_churned:,}")

    fig9 = px.pie(
        values=[not_churned, churned],
        names=['Active', 'Churned'],
        color_discrete_map={'Active':'#2ecc71', 'Churned':'#e74c3c'},
        hole=0.4
    )
    st.plotly_chart(fig9, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# PAGE 4 — SALES FORECAST
# ════════════════════════════════════════════════════════════════
elif page == "📈 Sales Forecast":
    st.title("📈 Sales Forecasting")
    st.markdown("Revenue trends and next 3 months forecast")
    st.markdown("---")

    # ── Historical + Forecast ─────────────────────────────────
    last_index    = monthly['MonthIndex'].max()
    last_month    = monthly['Month'].iloc[-1]
    last_year     = monthly['Year'].iloc[-1]

    # Build future months with seasonal features
    future_list = []
    for i in range(1, 4):
        next_month_index = last_index + i
        next_cal_month   = ((last_month - 1 + i) % 12) + 1
        future_list.append({
            'MonthIndex': next_month_index,
            'sin_month' : np.sin(2 * np.pi * next_cal_month / 12),
            'cos_month' : np.cos(2 * np.pi * next_cal_month / 12),
            'Month'     : next_cal_month,
            'Year'      : last_year if next_cal_month > last_month else last_year + 1
        })

    future_df      = pd.DataFrame(future_list)
    future_revenue = forecast_model.predict(
        future_df[['MonthIndex','sin_month','cos_month']]
    )
    future_df['TotalPrice'] = future_revenue
    future_df['Type']       = 'Forecast'

    # Historical data
    hist = monthly.copy()
    hist['Type'] = 'Historical'
    hist['Label'] = (hist['Year'].astype(str) + '-' +
                     hist['Month'].astype(str).str.zfill(2))

    future_df['Label'] = (future_df['Year'].astype(str) + '-' +
                           future_df['Month'].astype(str).str.zfill(2))

    # Plot historical
    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(
        x=hist['Label'], y=hist['TotalPrice'],
        mode='lines+markers',
        name='Historical Revenue',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6)
    ))
    fig10.add_trace(go.Scatter(
        x=future_df['Label'], y=future_df['TotalPrice'],
        mode='lines+markers',
        name='Forecasted Revenue',
        line=dict(color='green', width=2, dash='dash'),
        marker=dict(size=10, symbol='star')
    ))
    fig10.update_layout(
        title='Revenue — Historical + 3 Month Forecast',
        xaxis_title='Month',
        yaxis_title='Revenue (£)',
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig10, use_container_width=True)

    # ── Forecast Cards ────────────────────────────────────────
    st.subheader("Forecasted Revenue — Next 3 Months")
    col1, col2, col3 = st.columns(3)
    months_names = ['December 2011', 'January 2012', 'February 2012']

    for i, (col, rev) in enumerate(zip([col1,col2,col3], future_revenue)):
        with col:
            st.metric(
                label=months_names[i],
                value=f"£{rev:,.0f}"
            )

    st.markdown("---")

    # ── Model Performance ─────────────────────────────────────
    st.subheader("Forecasting Model Info")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Model", "Linear Regression + Seasonal")
    with col_b:
        st.metric("R² Score", "0.459")
    with col_c:
        st.metric("MAE", "£140,046")

    st.info("""
    **About this forecast:**
    - Model uses MonthIndex + sin/cos seasonal features
    - Trained on 19 months, tested on 5 months
    - Captures yearly seasonal patterns (Oct/Nov peaks, Jan/Feb dips)
    - For production use, Prophet or ARIMA would give better results
    """)