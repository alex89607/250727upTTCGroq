#!/usr/bin/env python3
"""
TTC Benchmarking Dashboard - Streamlit app for visualizing results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import duckdb
import yaml
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="TTC Benchmarking Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric > label {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    """Load configuration from YAML file"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Configuration file not found. Please run the benchmarks first.")
        return {}

@st.cache_data
def load_data():
    """Load benchmark results from parquet file"""
    results_path = Path("results/results.parquet")
    if not results_path.exists():
        st.error("Results file not found. Please run the benchmarks first.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(results_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return pd.DataFrame()

@st.cache_data
def load_summary_data():
    """Load summary statistics from dbt model"""
    db_path = Path("results/ttc_benchmarking.duckdb")
    if not db_path.exists():
        return pd.DataFrame()
    
    try:
        conn = duckdb.connect(str(db_path))
        summary_df = conn.execute("SELECT * FROM bench_summary").fetchdf()
        conn.close()
        return summary_df
    except Exception as e:
        st.warning(f"Could not load summary data: {e}")
        return pd.DataFrame()

def create_latency_vs_quality_plot(df):
    """Create scatter plot of latency vs quality trade-offs"""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    fig = go.Figure()
    
    # Create scatter plot manually to avoid plotly express issues
    for policy in df['policy_name'].unique():
        policy_data = df[df['policy_name'] == policy]
        
        fig.add_trace(go.Scatter(
            x=policy_data['first_token_latency'],
            y=policy_data.get('rouge_l', policy_data.get('f1', [0] * len(policy_data))),
            mode='markers',
            name=policy,
            marker=dict(
                size=policy_data['throughput'] / 20,  # Scale size
                opacity=0.7
            ),
            hovertemplate=(
                f"<b>{policy}</b><br>" +
                "Latency: %{x:.3f}s<br>" +
                "Quality: %{y:.3f}<br>" +
                "Throughput: %{customdata:.1f} tok/s<br>" +
                "<extra></extra>"
            ),
            customdata=policy_data['throughput']
        ))
    
    fig.update_layout(
        title="Latency vs Quality Trade-offs",
        xaxis_title="First Token Latency (s)",
        yaxis_title="Quality Score",
        height=500,
        showlegend=True
    )
    
    return fig

def create_throughput_distribution(df):
    """Create box plot of throughput distribution by policy"""
    if df.empty:
        return go.Figure().add_annotation(text="No data available", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    fig = go.Figure()
    
    # Create box plot manually to avoid plotly express issues
    for policy in df['policy_name'].unique():
        policy_data = df[df['policy_name'] == policy]
        
        fig.add_trace(go.Box(
            y=policy_data['throughput'],
            name=policy,
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig.update_layout(
        title="Throughput Distribution by Policy",
        xaxis_title="Policy",
        yaxis_title="Throughput (tokens/s)",
        height=500,
        showlegend=False
    )
    
    return fig

def create_performance_radar(summary_df, selected_policies):
    """Create radar chart comparing selected policies"""
    if summary_df.empty or not selected_policies:
        return go.Figure()
    
    # Normalize metrics to 0-1 scale for radar chart
    metrics = ['mean_first_token_latency', 'mean_throughput', 'mean_f1', 'mean_rouge_l']
    metric_labels = ['Latency (inv)', 'Throughput', 'F1 Score', 'ROUGE-L']
    
    fig = go.Figure()
    
    for policy in selected_policies:
        policy_data = summary_df[summary_df['policy_name'] == policy]
        if policy_data.empty:
            continue
        
        # Normalize values (invert latency so higher is better)
        values = []
        for i, metric in enumerate(metrics):
            if metric in policy_data.columns:
                val = policy_data[metric].iloc[0]
                if metric == 'mean_first_token_latency':
                    # Invert latency (lower is better -> higher normalized value)
                    val = 1 / (1 + val) if val > 0 else 1
                elif metric in ['mean_throughput', 'mean_f1', 'mean_rouge_l']:
                    # Normalize to 0-1 range
                    val = min(val, 1.0)
                values.append(val)
            else:
                values.append(0)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels,
            fill='toself',
            name=policy,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Policy Performance Comparison",
        height=500
    )
    
    return fig

def create_efficiency_ranking(summary_df):
    """Create bar chart of efficiency rankings"""
    if summary_df.empty:
        return go.Figure()
    
    # Sort by efficiency score
    top_policies = summary_df.nlargest(10, 'efficiency_score')
    
    fig = px.bar(
        top_policies,
        x='efficiency_score',
        y='policy_name',
        color='benchmark',
        orientation='h',
        title="Top 10 Most Efficient Policies",
        labels={
            'efficiency_score': 'Efficiency Score',
            'policy_name': 'Policy'
        }
    )
    
    fig.update_layout(height=500)
    
    return fig

def main():
    """Main dashboard function"""
    st.title("🚀 TTC Benchmarking Dashboard")
    st.markdown("Real-time analysis of Test-Time Compute policy performance")
    
    # Load data
    config = load_config()
    df = load_data()
    summary_df = load_summary_data()
    
    if df.empty:
        st.warning("No benchmark results found. Please run some benchmarks first.")
        st.code("python runner.py --policy speculative_decoding")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Policy filter
    available_policies = df['policy_name'].unique()
    selected_policies = st.sidebar.multiselect(
        "Select Policies",
        available_policies,
        default=available_policies[:5] if len(available_policies) > 5 else available_policies
    )
    
    # Benchmark filter
    available_benchmarks = df['benchmark'].unique()
    selected_benchmarks = st.sidebar.multiselect(
        "Select Benchmarks",
        available_benchmarks,
        default=available_benchmarks
    )
    
    # Date range filter
    if 'timestamp' in df.columns:
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
            min_value=df['timestamp'].min().date(),
            max_value=df['timestamp'].max().date()
        )
    
    # Filter data
    filtered_df = df[
        (df['policy_name'].isin(selected_policies)) &
        (df['benchmark'].isin(selected_benchmarks))
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # Key metrics
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_latency = filtered_df['first_token_latency'].mean()
        st.metric(
            "Avg First Token Latency",
            f"{avg_latency:.3f}s",
            delta=f"{(avg_latency - df['first_token_latency'].mean()):.3f}s"
        )
    
    with col2:
        avg_throughput = filtered_df['throughput'].mean()
        st.metric(
            "Avg Throughput",
            f"{avg_throughput:.1f} tok/s",
            delta=f"{(avg_throughput - df['throughput'].mean()):.1f}"
        )
    
    with col3:
        avg_f1 = filtered_df['f1'].mean()
        st.metric(
            "Avg F1 Score",
            f"{avg_f1:.3f}",
            delta=f"{(avg_f1 - df['f1'].mean()):.3f}"
        )
    
    with col4:
        total_samples = len(filtered_df)
        st.metric(
            "Total Samples",
            f"{total_samples:,}",
            delta=f"{total_samples - len(df):,}"
        )
    
    # Main visualizations
    st.header("📈 Performance Analysis")
    
    # Latency vs Quality scatter plot
    st.subheader("Latency vs Quality Trade-offs")
    latency_quality_fig = create_latency_vs_quality_plot(filtered_df)
    st.plotly_chart(latency_quality_fig, use_container_width=True)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Throughput Distribution")
        throughput_fig = create_throughput_distribution(filtered_df)
        st.plotly_chart(throughput_fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Radar")
        radar_fig = create_performance_radar(summary_df, selected_policies)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    # Efficiency ranking
    if not summary_df.empty:
        st.subheader("Efficiency Rankings")
        efficiency_fig = create_efficiency_ranking(summary_df)
        st.plotly_chart(efficiency_fig, use_container_width=True)
    
    # Detailed results table
    st.header("📋 Detailed Results")
    
    # Summary statistics table
    if not summary_df.empty:
        st.subheader("Summary Statistics")
        display_cols = [
            'policy_name', 'benchmark', 'sample_count',
            'mean_first_token_latency', 'mean_throughput', 'mean_f1',
            'efficiency_score', 'performance_profile'
        ]
        available_cols = [col for col in display_cols if col in summary_df.columns]
        st.dataframe(
            summary_df[available_cols].round(4),
            use_container_width=True
        )
    
    # Raw results table (sample)
    st.subheader("Sample Results")
    sample_size = st.slider("Number of samples to display", 10, 100, 50)
    
    display_cols = [
        'sample_id', 'policy_name', 'benchmark', 'first_token_latency',
        'throughput', 'f1', 'exact_match', 'rouge_l'
    ]
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    
    st.dataframe(
        filtered_df[available_cols].head(sample_size).round(4),
        use_container_width=True
    )
    
    # Export functionality
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Filtered Results"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ttc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not summary_df.empty and st.button("Download Summary"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary CSV",
                data=csv,
                file_name=f"ttc_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
