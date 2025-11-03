"""
Bottleneck visualization functions adapted from longest_station.ipynb for API use.
Supports both matplotlib (static images) and plotly (interactive JSON) outputs.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import io
import base64

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for API
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Stage colors matching notebook
STAGE_COLORS = {
    'Вїзд': '#FF0000',
    'Відбір_проб': '#FF7F00',
    'Зважування': '#FFFF00',
    'Лабораторія': '#00FF00',
    'Розвантаження': "#0021C4",
    'Виїзд': '#8F00FF'
}

# Configure matplotlib for Cyrillic fonts
if MATPLOTLIB_AVAILABLE:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


def prepare_bottleneck_data(df_durations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare bottleneck dataframes for visualizations.
    Returns: daily, weekly, monthly bottlenecks, overall stats
    """
    # Add date column if not present
    if 'date' not in df_durations.columns and 'start_ts' in df_durations.columns:
        df_durations = df_durations.copy()
        # Ensure start_ts is datetime before using .dt accessor
        if not pd.api.types.is_datetime64_any_dtype(df_durations['start_ts']):
            df_durations['start_ts'] = pd.to_datetime(df_durations['start_ts'], errors="coerce")
        df_durations = df_durations.dropna(subset=['start_ts'])  # Remove rows where conversion failed
        df_durations['date'] = df_durations['start_ts'].dt.date
        df_durations['week'] = df_durations['start_ts'].dt.to_period('W')
        df_durations['month'] = df_durations['start_ts'].dt.to_period('M')
    
    # Daily bottlenecks - ensure date is a proper date type
    daily = df_durations.groupby(['date', 'stage'])['duration_min'].median().reset_index()
    # Convert date to datetime if it's a date object (not datetime)
    if daily['date'].dtype == 'object' or not pd.api.types.is_datetime64_any_dtype(daily['date']):
        daily['date'] = pd.to_datetime(daily['date'])
    daily['is_bottleneck'] = daily.groupby('date')['duration_min'].transform(lambda x: x == x.max())
    
    # Weekly bottlenecks
    weekly = df_durations.groupby(['week', 'stage'])['duration_min'].median().reset_index()
    weekly['is_bottleneck'] = weekly.groupby('week')['duration_min'].transform(lambda x: x == x.max())
    
    # Monthly bottlenecks
    monthly = df_durations.groupby(['month', 'stage'])['duration_min'].median().reset_index()
    monthly['is_bottleneck'] = monthly.groupby('month')['duration_min'].transform(lambda x: x == x.max())
    
    # Overall statistics
    overall = df_durations.groupby('stage')['duration_min'].agg(['median', 'mean', 'std']).reset_index()
    overall['throughput_per_hour'] = 60 / overall['median']  # vehicles per hour
    
    return {
        'daily': daily,
        'weekly': weekly,
        'monthly': monthly,
        'overall': overall,
        'raw': df_durations
    }


def plot_daily_bottleneck_timeline(data: Dict[str, pd.DataFrame], output_format: str = 'base64') -> str | go.Figure:
    """
    1. Timeline visualization showing which stage was bottleneck each day.
    
    Returns base64-encoded PNG string (if format='base64') or plotly Figure (if format='plotly')
    """
    daily = data['daily']
    bottlenecks_only = daily[daily['is_bottleneck'] == True].copy()
    # Ensure date is datetime before using it
    if not pd.api.types.is_datetime64_any_dtype(bottlenecks_only['date']):
        bottlenecks_only['date'] = pd.to_datetime(bottlenecks_only['date'], errors='coerce')
    bottlenecks_only = bottlenecks_only.dropna(subset=['date'])  # Remove rows where conversion failed
    
    if output_format == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure()
        for stage in bottlenecks_only['stage'].unique():
            stage_data = bottlenecks_only[bottlenecks_only['stage'] == stage]
            fig.add_trace(go.Scatter(
                x=stage_data['date'],
                y=stage_data['duration_min'],
                mode='markers',
                name=stage,
                marker=dict(
                    color=STAGE_COLORS.get(stage, '#000000'),
                    size=10,
                    line=dict(width=1, color='black')
                )
            ))
        fig.update_layout(
            title='Динаміка вузьких місць по днях',
            xaxis_title='Дата',
            yaxis_title='Час (хвилини)',
            hovermode='closest',
            width=1200,
            height=400
        )
        return fig
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(16, 6))
        for stage in bottlenecks_only['stage'].unique():
            stage_data = bottlenecks_only[bottlenecks_only['stage'] == stage]
            ax.scatter(stage_data['date'], stage_data['duration_min'],
                      label=stage, color=STAGE_COLORS.get(stage, '#000000'),
                      s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Дата', fontsize=12, fontweight='bold')
        ax.set_ylabel('Час (хвилини)', fontsize=12, fontweight='bold')
        ax.set_title('Динаміка вузьких місць по днях', fontsize=14, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_format == 'base64':
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_base64
        return fig
    else:
        raise ImportError("matplotlib or plotly required for visualizations")




def plot_bottleneck_frequency(data: Dict[str, pd.DataFrame], output_format: str = 'base64') -> str | go.Figure:
    """
    3. Frequency of bottlenecks: how often each stage is the bottleneck (daily, weekly, monthly).
    """
    daily = data['daily']
    weekly = data['weekly']
    monthly = data['monthly']
    
    if output_format == 'plotly' and PLOTLY_AVAILABLE:
        fig = make_subplots(rows=1, cols=3, subplot_titles=('По днях', 'По тижнях', 'По місяцях'))
        
        for idx, (df, period_col) in enumerate([(daily, 'date'), (weekly, 'week'), (monthly, 'month')], 1):
            bottlenecks_only = df[df['is_bottleneck'] == True]
            frequency = bottlenecks_only['stage'].value_counts()
            total = len(bottlenecks_only[period_col].unique())
            percentages = (frequency / total * 100).sort_values(ascending=True)
            
            colors = [STAGE_COLORS.get(stage, '#000000') for stage in percentages.index]
            
            fig.add_trace(go.Bar(
                y=percentages.index,
                x=percentages.values,
                orientation='h',
                name=f'Period {idx}',
                marker_color=colors,
                text=[f'{v:.1f}%' for v in percentages.values],
                textposition='outside'
            ), row=1, col=idx)
        
        fig.update_layout(
            title='Варіативність вузьких місць',
            height=400,
            showlegend=False
        )
        fig.update_xaxes(title_text='Відсоток періодів (%)', row=1, col=1)
        fig.update_xaxes(title_text='Відсоток періодів (%)', row=1, col=2)
        fig.update_xaxes(title_text='Відсоток періодів (%)', row=1, col=3)
        return fig
    elif MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        periods = [
            (daily, 'По днях', axes[0], 'date'),
            (weekly, 'По тижнях', axes[1], 'week'),
            (monthly, 'По місяцях', axes[2], 'month')
        ]
        
        for df, title, ax, period_col in periods:
            bottlenecks_only = df[df['is_bottleneck'] == True]
            frequency = bottlenecks_only['stage'].value_counts()
            total = len(bottlenecks_only[period_col].unique())
            percentages = (frequency / total * 100).sort_values(ascending=True)
            colors = [STAGE_COLORS.get(stage, '#000000') for stage in percentages.index]
            
            percentages.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Відсоток періодів (%)', fontsize=10)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            for i, v in enumerate(percentages):
                ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if output_format == 'base64':
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_base64
        return fig
    else:
        raise ImportError("matplotlib or plotly required for visualizations")


def create_summary_table(data: Dict[str, pd.DataFrame], output_format: str = 'base64') -> str | go.Figure:
    """
    6. Summary statistics table.
    """
    overall = data['overall'].sort_values('median', ascending=False)
    
    if output_format == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Етап', 'Медіана (хв)', 'Середнє (хв)', 'Ст.відх. (хв)', 'Пропускна здатність (м/год)'],
                fill_color='#4ECDC4',
                align='center',
                font=dict(color='white', size=12, family='Arial Black')
            ),
            cells=dict(
                values=[
                    overall['stage'],
                    [f"{v:.2f}" for v in overall['median']],
                    [f"{v:.2f}" for v in overall['mean']],
                    [f"{v:.2f}" for v in overall['std']],
                    [f"{v:.1f}" for v in overall['throughput_per_hour']]
                ],
                fill_color=[['#F0F0F0' if i % 2 == 0 else 'white' for i in range(len(overall))]],
                align='center',
                font=dict(size=10)
            )
        )])
        fig.update_layout(title='Загальна статистика по етапах', height=300)
        return fig
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [['Етап', 'Медіана (хв)', 'Середнє (хв)', 'Ст.відх. (хв)', 'Пропускна\nздатність (м/год)']]
        for _, row in overall.iterrows():
            table_data.append([
                row['stage'],
                f"{row['median']:.2f}",
                f"{row['mean']:.2f}",
                f"{row['std']:.2f}",
                f"{row['throughput_per_hour']:.1f}"
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(5):
            cell = table[(0, i)]
            cell.set_facecolor('#4ECDC4')
            cell.set_text_props(weight='bold', color='white')
        
        for i in range(1, len(table_data)):
            for j in range(5):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#F0F0F0')
        
        ax.set_title('Загальна статистика по етапах', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_format == 'base64':
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_base64
        return fig
    else:
        raise ImportError("matplotlib or plotly required for visualizations")





# --- UPDATED: generator to the new minimal set --------------------------------
def generate_all_visualizations(
    df_durations: pd.DataFrame,
    df_stations: pd.DataFrame = None,
    output_format: str = "base64",
    flow_freq: str = "15T"
) -> Dict[str, str | go.Figure]:
    """
    Generate bottleneck visualizations:
      - timeline: хто bottleneck щодня
      - frequency: варіативність bottleneckів
      - summary: підсумкова таблиця
    """
    # durations => bottleneck views
    data = prepare_bottleneck_data(df_durations)

    return {
        "timeline": plot_daily_bottleneck_timeline(data, output_format),
        "frequency": plot_bottleneck_frequency(data, output_format),
        "summary": create_summary_table(data, output_format),
    }

