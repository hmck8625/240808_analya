import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI 
from dotenv import load_dotenv
import os
import requests
from datetime import timedelta
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import math

# .env ファイルから環境変数を読み込む
load_dotenv()

st.title('デジタル広告データ分析アプリ')

#APIキーを環境変数から取得
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#API_KEY = os.getenv("SHEETS_API_KEY")

OPENAI_API_KEY = st.secrets.AzureApiKey.OPENAI_API_KEY
API_KEY = st.secrets.AzureApiKey.SHEETS_API_KEY

# スプレッドシートIDの入力
SPREADSHEET_ID = st.text_input("Google SpreadsheetのIDを入力してください", value="1BD-AEaNEWpPyzb5CySUc_XlNqWNIzu_1tC8C0g68Dpw")

# シート名の入力
SHEET_NAME = st.text_input("シート名を入力してください", value="シート1")

# OpenAIクライアントの初期化
client = OpenAI(api_key=OPENAI_API_KEY)


if API_KEY and SPREADSHEET_ID and SHEET_NAME:
    try:
        # Google Sheetsからデータを取得
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{SPREADSHEET_ID}/values/{SHEET_NAME}?key={API_KEY}"
        response = requests.get(url)
        data = response.json()

        # DataFrameに変換
        df = pd.DataFrame(data['values'][1:], columns=data['values'][0])

        # データ型の変換
        df['day'] = pd.to_datetime(df['day'])
        df['media'] = df['media'].astype(str)

        # 'media'列以外の数値列を変換
        numeric_columns = df.columns.drop(['day', 'media'])
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace('¥', ''), errors='coerce')

        # 週の情報を追加
        df['week'] = df['day'].dt.to_period('W').astype(str)

        # 日別データの集計
        df_daily = df.groupby(['media', 'day']).agg({
            'impression': 'sum',
            'click': 'sum',
            'cost': 'sum',
            'cv': 'sum'
        }).reset_index()
        df_daily['cpa'] = df_daily['cost'] / df_daily['cv'].replace(0, 1)  # 0で割るのを防ぐ

        # 週別データの集計
        df_weekly = df.groupby(['media', 'week']).agg({
            'impression': 'sum',
            'click': 'sum',
            'cost': 'sum',
            'cv': 'sum'
        }).reset_index()
        df_weekly['cpa'] = df_weekly['cost'] / df_weekly['cv'].replace(0, 1)  # 0で割るのを防ぐ

        # 分析期間の選択
        start_date = df['day'].min()
        end_date = df['day'].max()
        date_range = st.date_input(
            "分析期間を選択してください",
            [start_date, end_date],
            min_value=start_date,
            max_value=end_date
        )

        if len(date_range) == 2:
            df_daily_filtered = df_daily[(df_daily['day'] >= pd.Timestamp(date_range[0])) & 
                                         (df_daily['day'] <= pd.Timestamp(date_range[1]))]
            df_weekly_filtered = df_weekly[df_weekly['week'].isin(df['week'][(df['day'] >= pd.Timestamp(date_range[0])) & 
                                                                        (df['day'] <= pd.Timestamp(date_range[1]))])]

            # 分析タイプの選択
            analysis_type = st.radio(
                "分析タイプを選択してください",
                ('日別', '週別')
            )

            if analysis_type == '日別':
                df_filtered = df_daily_filtered
                x_axis = 'day'
            else:
                df_filtered = df_weekly_filtered
                x_axis = 'week'

            # グラフ作成関数
            def create_stacked_bar(df, x, y, title):
                fig = go.Figure()
                for medium in df['media'].unique():
                    df_medium = df[df['media'] == medium]
                    fig.add_trace(go.Bar(x=df_medium[x], y=df_medium[y], name=medium))
                fig.update_layout(barmode='stack', title=title, xaxis_title=x, yaxis_title=y)
                return fig

            def create_line_chart(df, x, y, title):
                fig = go.Figure()
                for medium in df['media'].unique():
                    df_medium = df[df['media'] == medium]
                    fig.add_trace(go.Scatter(x=df_medium[x], y=df_medium[y], mode='lines+markers', name=medium))
                fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
                return fig

            # グラフ表示
            st.subheader(f"{analysis_type}分析結果")

            # Cost推移
            fig_cost = create_stacked_bar(df_filtered, x_axis, 'cost', f'媒体別の{analysis_type}Cost推移')
            st.plotly_chart(fig_cost)

            # Install推移
            fig_install = create_stacked_bar(df_filtered, x_axis, 'cv', f'媒体別の{analysis_type}Install推移')
            st.plotly_chart(fig_install)

            # CPA推移
            fig_cpa = create_line_chart(df_filtered, x_axis, 'cpa', f'媒体別の{analysis_type}CPA推移')
            st.plotly_chart(fig_cpa)

            st.subheader("日付比較分析")
            analysis_date = st.date_input("分析する日付を選択してください", min_value=start_date, max_value=end_date)

            if st.button("分析実行"):
                
                # 新しい分析関数
                def get_date_data(df, date):
                    data = df[df['day'] == pd.to_datetime(date)].copy()
                    # 足りない指標を計算
                    data['cpm'] = (data['cost'] / data['impression']) * 1000
                    data['ctr'] = (data['click'] / data['impression']) * 100

                    data['cpc'] = data['cost'] / data['click']
                    data['cvr'] = (data['cv'] / data['click']) * 100
                    data['cpa'] = data['cost'] / data['cv']
                    
                    return data[['day', 'media', 'impression', 'click', 'cpm', 'ctr', 'cpc', 'cost', 'cv', 'cvr', 'cpa']]

                def calculate_total_metrics(data):
                    total = data.groupby('day').agg({'cv': 'sum', 'cost': 'sum'}).reset_index()
                    total['cpa'] = total['cost'] / total['cv']
                    return total

                def calculate_media_metrics(data):
                    media = data.groupby('media').agg({'cv': 'sum', 'cost': 'sum'}).reset_index()
                    media['cpa'] = media['cost'] / media['cv']
                    return media

                def analyze_overall(total1, total2):
                    if total1.empty or total2.empty:
                        st.warning("警告: 指定された日付のデータが見つかりません。")
                        return None

                    if 'cv' not in total1.columns or 'cv' not in total2.columns or 'cpa' not in total1.columns or 'cpa' not in total2.columns:
                        st.warning("警告: 必要なカラム（cv または cpa）が見つかりません。")
                        return None

                    cv_diff = total2['cv'].iloc[0] - total1['cv'].iloc[0]
                    cv_ratio = ((total2['cv'].iloc[0] / total1['cv'].iloc[0]) - 1) * 100 if total1['cv'].iloc[0] != 0 else float('inf')

                    cpa_diff = total2['cpa'].iloc[0] - total1['cpa'].iloc[0]
                    cpa_ratio = ((total2['cpa'].iloc[0] / total1['cpa'].iloc[0]) - 1) * 100 if total1['cpa'].iloc[0] != 0 else float('inf')

                    return {
                        'cv_diff': cv_diff,
                        'cv_ratio': cv_ratio,
                        'cpa_diff': cpa_diff,
                        'cpa_ratio': cpa_ratio
                    }

                def analyze_media_contribution(media1, media2, overall_results):
                    merged = pd.merge(media1, media2, on='media', suffixes=('_1', '_2'))
                    merged['delta_cv'] = merged['cv_2'] - merged['cv_1']
                    merged['delta_cpa'] = merged['cpa_2'] - merged['cpa_1']

                    total_cost = merged['cost_1'].sum()
                    merged['cost_volume'] = merged['cost_1'] / total_cost

                    merged['cv_contribution'] = (merged['delta_cv'] / overall_results['cv_diff']) * 100
                    merged['cpa_contribution'] = (merged['delta_cpa'] / overall_results['cpa_diff']) * merged['cost_volume'] * 100

                    return merged
                
                def format_overall_results(results):
                    text_output = (f"全体分析結果:\n"
                                  f"CV変化: {results['cv_diff']} ({results['cv_ratio']:.2f}%)\n"
                                  f"CPA変化: {results['cpa_diff']:.2f} ({results['cpa_ratio']:.2f}%)\n")
                    
                    # 表形式のデータを作成
                    table_data = pd.DataFrame({
                        '指標': ['CV変化', 'CPA変化'],
                        '絶対値': [results['cv_diff'], results['cpa_diff']],
                        '変化率(%)': [results['cv_ratio'], results['cpa_ratio']]
                    })
                    
                    return text_output, table_data

                def format_media_results(media_results):
                    text_output = "\nメディア別貢献度分析結果:\n"
                    for _, row in media_results.iterrows():
                        text_output += (f"メディア: {row['media']}\n"
                                        f"CV貢献度: {row['cv_contribution']:.2f}%\n"
                                        f"CPA貢献度: {row['cpa_contribution']:.2f}%\n\n")
                    
                    # 表形式のデータを作成
                    table_data = media_results[['media', 'delta_cv', 'cv_contribution', 'delta_cpa', 'cpa_contribution']].copy()
                    table_data.columns = ['メディア','CV差分','CV貢献度(%)', 'CPA差分','CPA貢献度(%)']
                                        
                    return text_output, table_data

                def add_total_row(data):
                    # 数値列のみを合計
                    total = data.select_dtypes(include=[np.number]).sum()
                    
                    # 'media' 列に 'Total' を設定
                    total['media'] = 'Total'
                    
                    # 計算が必要な指標を正しく計算
                    total['ctr'] = (total['click'] / total['impression']) * 100 if total['impression'] != 0 else 0
                    total['cpm'] = (total['cost'] / total['impression']) * 1000 if total['cost'] != 0 else 0

                    total['cpc'] = total['cost'] / total['click'] if total['click'] != 0 else 0
                    total['cvr'] = (total['cv'] / total['click']) * 100 if total['click'] != 0 else 0
                    total['cpa'] = total['cost'] / total['cv'] if total['cv'] != 0 else 0
                    
                    # 元のデータと合計行を結合し、インデックスをリセット
                    return pd.concat([data, total.to_frame().T]).reset_index(drop=True)
                
                def get_date_range_data(df, start_date, end_date):
                    mask = (df['day'] >= pd.to_datetime(start_date)) & (df['day'] <= pd.to_datetime(end_date))
                    data = df[mask].copy()
                    data['ctr'] = (data['click'] / data['impression']) * 100
                    data['cpm'] = (data['cost'] / data['impression']) * 1000
                    data['cpc'] = data['cost'] / data['click']
                    data['cvr'] = (data['cv'] / data['click']) * 100
                    data['cpa'] = data['cost'] / data['cv']
                    return data.groupby('media').agg({
                        'impression': 'sum',
                        'click': 'sum',
                        'cost': 'sum',
                        'cv': 'sum'
                    }).reset_index()

                def calculate_metrics(data):
                    data['ctr'] = (data['click'] / data['impression']) * 100
                    data['cpm'] = (data['cost'] / data['impression']) * 1000
                    data['cpc'] = data['cost'] / data['click']
                    data['cvr'] = (data['cv'] / data['click']) * 100
                    data['cpa'] = data['cost'] / data['cv']
                    return data


                def analyze_comparison(data1, data2):
                    # データをマージし、計算を行う
                    merged = pd.merge(data1, data2, on='media', suffixes=('_1', '_2'))
                    merged['delta_cv'] = merged['cv_2'] - merged['cv_1']
                    merged['delta_cpa'] = merged['cpa_2'] - merged['cpa_1']
                    total_cost = merged['cost_1'].sum()
                    merged['cost_volume'] = merged['cost_1'] / total_cost
                    
                    overall_cv_diff = merged['delta_cv'].sum()
                    merged['cv_contribution'] = (merged['delta_cv'] / overall_cv_diff) * 100

                    overall_cpa_diff = merged['cpa_2'] - merged['cpa_1']
                    merged['cpa_contribution'] = merged['cpa_2'] / merged['cpa_1'] * 100

                    # Total行の計算と追加（修正版）
                    total_row = pd.DataFrame({
                        'media': ['Total'],
                        'impression_1': [merged['impression_1'].sum()],
                        'impression_2': [merged['impression_2'].sum()],
                        'click_1': [merged['click_1'].sum()],
                        'click_2': [merged['click_2'].sum()],
                        'cv_1': [merged['cv_1'].sum()],
                        'cv_2': [merged['cv_2'].sum()],
                        'cost_1': [merged['cost_1'].sum()],
                        'cost_2': [merged['cost_2'].sum()],
                        'cost_volume': [0],
                        'cv_contribution': [np.nan],
                    })

                    # 合計値に基づく指標の計算
                    total_row['cpa_1'] = total_row['cost_1'] / total_row['cv_1']
                    total_row['cpa_2'] = total_row['cost_2'] / total_row['cv_2']
                    total_row['ctr_1'] = total_row['click_1'] / total_row['impression_1'] * 100
                    total_row['ctr_2'] = total_row['click_2'] / total_row['impression_2'] * 100
                    total_row['cpm_1'] = total_row['cost_1'] / total_row['impression_1'] * 1000
                    total_row['cpm_2'] = total_row['cost_2'] / total_row['impression_2'] * 1000
                    total_row['cpc_1'] = total_row['cost_1'] / total_row['click_1']
                    total_row['cpc_2'] = total_row['cost_2'] / total_row['click_2']
                    total_row['cvr_1'] = total_row['cv_1'] / total_row['click_1'] * 100
                    total_row['cvr_2'] = total_row['cv_2'] / total_row['click_2'] * 100

                    # delta値の計算
                    total_row['delta_cv'] = total_row['cv_2'] - total_row['cv_1']
                    total_row['delta_cpa'] = total_row['cpa_1'] - total_row['cpa_2']
                    total_row['cpa_contoribution'] = total_row['cpa_2'] / total_row['cpa_1'] * 100

                    #total_row['delta_ctr'] = total_row['ctr_1'] - total_row['ctr_2']
                    #total_row['delta_cpc'] = total_row['cpc_1'] - total_row['cpc_2']
                    #total_row['delta_cvr'] = total_row['cvr_1'] - total_row['cvr_2']

                    # マージ結果にTotal行を追加
                    merged = pd.concat([merged, total_row], ignore_index=True)

                    return merged, overall_cv_diff, overall_cpa_diff
                       
                ### cpa変化の指標分析関数

                def analyze_cpa_change(data, media='Total'):

                    """
                    CPAの変化要因を分析し、Streamlitでグラフを描画する関数
                    
                    Parameters:
                    data (pd.DataFrame): 各指標のデータを含むDataFrame
                    media (str): 分析対象のメディア（デフォルトは'Total'）
                    
                    Returns:
                    dict: 分析結果を含む辞書
                    """
                    # データのコピーを作成
                    df = data[data['media'] == media].copy()
                    
                    # 変化率の計算
                    for metric in ['cpm', 'ctr', 'cpc', 'cvr', 'cpa']:
                        df[f'{metric}_change'] = ((df[f'{metric}_2'] - df[f'{metric}_1']))
                    
                    # 改善幅を取得 改善率＝１０0%を超えればよい
                    ## cost系は低い方が良いため、前➗後
                    cpa_change = df['cpa_1'].values[0] / df['cpa_2'].values[0]
                    cpc_change = df['cpc_1'].values[0] / df['cpc_2'].values[0]
                    cpm_change = df['cpm_1'].values[0] / df['cpm_2'].values[0]

                    ## 率系は高い方が良いため、後➗前
                    cvr_change = df['cvr_2'].values[0] / df['cvr_1'].values[0]
                    ctr_change = df['ctr_2'].values[0] / df['ctr_1'].values[0]

                    # 結果をまとめる
                    results = {
                        'raw_data': df,
                        'cpa_impacts': {
                            'CPA': cpa_change,
                            'CVR': cvr_change,
                            'CPC': cpc_change
                        },
                        'cpc_impacts': {
                            'CPC': cpc_change,
                            'CPM': cpm_change,
                            'CTR': ctr_change
                        }
                    }
                    
                    # Streamlitでグラフ描画
                    st.subheader(f"{media}のCPA変化要因分析")
                    
                    def create_chart(data, title):
                        fig = go.Figure()
                        # 基準線（1.0）を追加
                        fig.add_shape(type="line",
                                      x0=-0.5, y0=1, x1=2.5, y1=1,
                                      line=dict(color="red", width=2, dash="dash"))
                        
                        # バーを追加
                        fig.add_trace(go.Bar(
                            x=data['Metric'],
                            y=data['Value'],
                            text=[f"{(v-1)*100:.2f}%" for v in data['Value']],  # パーセンテージ形式で値を表示
                            textposition='outside',  # テキストをバーの外側に配置
                            marker_color=['red' if v < 1 else 'green' for v in data['Value']]  # 1未満は赤、1以上は緑
                        ))
                        
                        # レイアウトを設定
                        fig.update_layout(
                            title=title,
                            yaxis=dict(
                                title="Change Rate",
                                range=[min(0.5, min(data['Value']) - 0.1), max(1.5, max(data['Value']) + 0.1)],
                                tickformat=".0%",  # y軸のラベルをパーセンテージ形式に
                                tickvals=[0.5, 0.75, 1, 1.25, 1.5],  # 表示するティックの値
                                ticktext=["-50%", "-25%", "0%", "+25%", "+50%"]  # ティックのラベル
                            ),
                            height=400,
                            showlegend=False
                        )
                        
                        return fig

                    # 1. CPAの変化率のグラフとCVR, CPCの変化率のグラフ
                    cpa_chart_data = pd.DataFrame({
                        'Metric': ['CPA', 'CVR', 'CPC'],
                        'Value': [cpa_change, cvr_change, cpc_change]
                    })
                    
                    st.plotly_chart(create_chart(cpa_chart_data, "CPAに対する CVR, CPCの改善率"))

                    # 2. CPCの変化率のグラフとCTR, CPMの変化率のグラフ
                    cpc_chart_data = pd.DataFrame({
                        'Metric': ['CPC', 'CTR', 'CPM'],
                        'Value': [cpc_change, ctr_change, cpm_change]
                    })
                    st.plotly_chart(create_chart(cpc_chart_data, "CPCに対する CTR, CPMの改善率"))

                    ### 生データの表示
                    st.write("媒体比較データ")
                    raw_data = df
                    
                    # 必要な列のみを選択
                    columns_to_display = ['media', 'impression', 'click', 'cost', 'cv', 'ctr', 'cpc', 'cvr', 'cpa', 'cpm']
                    
                    # 期間1と期間2のデータを別々の行に分ける
                    period1_data = raw_data[[f'{col}_1' for col in columns_to_display if f'{col}_1' in raw_data.columns]]
                    period2_data = raw_data[[f'{col}_2' for col in columns_to_display if f'{col}_2' in raw_data.columns]]
                    
                    # 列名から _1 と _2 を削除
                    period1_data.columns = [col.replace('_1', '') for col in period1_data.columns]
                    period2_data.columns = [col.replace('_2', '') for col in period2_data.columns]
                    
                    # 期間の列を追加
                    period1_data.insert(0, 'period', '期間1(過去)')
                    period2_data.insert(0, 'period', '期間2(直近)')
                    
                    # 2つのデータフレームを結合
                    display_data = pd.concat([period1_data, period2_data])
                    
                    st.dataframe(display_data)
                    
                    return results
                

                # 3つの分析パターンを実行
                patterns = [
                    ("前日比較", analysis_date - timedelta(days=1), analysis_date),
                    ("1週間比較", analysis_date - timedelta(days=13), analysis_date - timedelta(days=7), analysis_date - timedelta(days=6), analysis_date),
                    ("2週間比較", analysis_date - timedelta(days=25), analysis_date - timedelta(days=13), analysis_date - timedelta(days=12), analysis_date)
                ]

                for pattern_name, *dates in patterns:
                        with st.expander(f"{pattern_name}"):
                            # ここに各パターンの分析コードを配置
                            st.subheader(f"{pattern_name}の詳細")
                            
                            if len(dates) == 2:
                                data1 = get_date_range_data(df, dates[0], dates[0])
                                data2 = get_date_range_data(df, dates[1], dates[1])
                            else:
                                data1 = get_date_range_data(df, dates[0], dates[1])
                                data2 = get_date_range_data(df, dates[2], dates[3])

                            data1 = calculate_metrics(data1)
                            data2 = calculate_metrics(data2)

                            # 合計行を追加
                            data1_with_total = add_total_row(data1)
                            data2_with_total = add_total_row(data2)


                            st.write(f"期間1: {dates[0]} から {dates[1] if len(dates) > 2 else dates[0]}")
                            st.dataframe(data1_with_total.style.format({
                                'impression': '{:,.0f}',
                                'click': '{:,.0f}',
                                'ctr': '{:.2f}%',
                                'cpc': '¥{:.2f}',
                                'cost': '¥{:,.0f}',
                                'cv': '{:,.0f}',
                                'cvr': '{:.2f}%',
                                'cpa': '¥{:,.2f}'
                            }).apply(lambda x: ['background-color: gray' if x.name == len(x) else '' for i in x], axis=1))


                            st.write(f"期間2: {dates[2] if len(dates) > 2 else dates[1]} から {dates[3] if len(dates) > 2 else dates[1]}")
                            st.dataframe(data2_with_total.style.format({
                                'impression': '{:,.0f}',
                                'click': '{:,.0f}',
                                'ctr': '{:.2f}%',
                                'cpc': '¥{:.2f}',
                                'cost': '¥{:,.0f}',
                                'cv': '{:,.0f}',
                                'cvr': '{:.2f}%',
                                'cpa': '¥{:,.2f}'
                            }).apply(lambda x: ['background-color: gray' if x.name == len(x) else '' for i in x], axis=1))

                            media_results, overall_cv_diff, overall_cpa_diff = analyze_comparison(data1, data2)

                            overall_results = {
                                'cv_diff': overall_cv_diff,
                                'cv_ratio': ((data2['cv'].sum() / data1['cv'].sum()) - 1) * 100 if data1['cv'].sum() != 0 else float('inf'),
                                'cpa_diff': (data2['cost'].sum() / data2['cv'].sum()) - (data1['cost'].sum() / data1['cv'].sum()) if data1['cv'].sum() != 0 and data2['cv'].sum() != 0 else float('inf'),
                                'cpa_ratio': (((data2['cost'].sum() / data2['cv'].sum()) / (data1['cost'].sum() / data1['cv'].sum())) - 1) * 100 if data1['cv'].sum() != 0 and data2['cv'].sum() != 0 else float('inf')
                            }

                            ##調整########
                            
                            # media_resultsから必要な列を選択
                            selected_columns = ['media', 'delta_cv', 'cv_contribution', 'delta_cpa', 'cpa_contribution']
                            cv_cpa_table = media_results[selected_columns].copy()

                            # 数値のフォーマットを調整
                            cv_cpa_table['delta_cv'] = cv_cpa_table['delta_cv'].round(0)
                            cv_cpa_table['cv_contribution'] = cv_cpa_table['cv_contribution'].round(0).astype(str) + '%'
                            cv_cpa_table['delta_cpa'] = cv_cpa_table['delta_cpa'].round(0)
                            cv_cpa_table['cpa_contribution'] = cv_cpa_table['cpa_contribution'].round(0).astype(str) + '%'
                            
                            # 列名を日本語に変更
                            cv_cpa_table.columns = ['メディア', 'CV変化量', 'CV貢献度', 'CPA変化量', 'CPA変化率']

                            ############
                            st.subheader("CV・CPA指標の変化と貢献度")

                            def plot_cpa_cv_scatter(media_results):  ### cpa-cvの散布図を作成する関数
                                """
                                CPAとCV数の散布図を描画し、各メディアの期間1から期間2への変化を矢印で示す関数

                                Parameters:
                                media_results (pd.DataFrame): media, cpa_1, cpa_2, cv_1, cv_2 カラムを含むDataFrame

                                Returns:
                                plotly.graph_objects.Figure: 描画されたグラフオブジェクト
                                """
                                
                                # プロットの初期化
                                fig = go.Figure()

                                # 各メディアに対して散布図とアノテーションを追加
                                for _, row in media_results.iterrows():
                                    media = row['media']
                                    
                                    # 期間1と期間2のデータポイントを追加
                                    fig.add_trace(go.Scatter(
                                        x=[row['cv_1'], row['cv_2']],
                                        y=[row['cpa_1'], row['cpa_2']],
                                        mode='markers',
                                        name=media,
                                        text=[f"{media} (1)", f"{media} (2)"],
                                        textposition="top center",
                                        showlegend=True,
                                        hoverinfo='text',
                                        hovertext=[f"{media} (1)<br>CV: {row['cv_1']}<br>CPA: {row['cpa_1']:,.0f}", 
                                                   f"{media} (2)<br>CV: {row['cv_2']}<br>CPA: {row['cpa_2']:,.0f}"],
                                        ))
                                    
                                    # 矢印を追加
                                    fig.add_annotation(
                                        x=row['cv_2'],
                                        y=row['cpa_2'],
                                        ax=row['cv_1'],
                                        ay=row['cpa_1'],
                                        xref="x", yref="y",
                                        axref="x", ayref="y",
                                        showarrow=True,
                                        arrowhead=2,
                                        arrowsize=1,
                                        arrowwidth=2,
                                        arrowcolor="#636363"
                                    )

                                # レイアウトの設定
                                fig.update_layout(
                                    title="CPA vs CV 散布図（cpaの変化要因分析用）",
                                    xaxis_title="CV",
                                    yaxis_title="CPA",
                                    showlegend=True,
                                    hovermode="closest"
                                )

                                # 軸の範囲を自動調整（すべてのポイントが表示されるように）
                                fig.update_xaxes(autorange=True)
                                fig.update_yaxes(autorange=True, tickformat=",")  # Y軸のティックフォーマットを変更

                                return fig
                                
                            st.plotly_chart(plot_cpa_cv_scatter(media_results))

                            st.dataframe(cv_cpa_table)

                            overall_text, overall_table = format_overall_results(overall_results)
                            media_text, media_table = format_media_results(media_results)


                            prompt = (f"以下に広告の配信結果の{pattern_name}を示します。このデータを分析し、以下の指示に従って簡潔に記述してください。\n\n"
                                        f"#プロモーション全体の推移\n{overall_text}\n"
                                        f"#メディア別の差分\n{media_text}\n"
                                        f"#分析FMT"
                                        "全体変化:全体のCV数とCPAの変化について簡潔に述べてください。\n"
                                        "メディア別変化箇所：CV貢献度とCPA貢献から、全体の変化の要因となっているメディアを簡潔に教えてください\n"
                                        "#注意事項\n"
                                        "・値が0やinf, nanになっている項目については言及しないでください。\n"
                                        "・分析は事実の記述に留め、推測や提案は含めないでください。\n"
                                        f"・{pattern_name}のデータであることを前提に分析してください。\n"
                                        "最も重要です。上記の分析は簡潔に１００文字程度でまとめてください。")


                            response = client.chat.completions.create(
                                #model="gpt-3.5-turbo",
                                model="gpt-4-turbo",

                                messages=[
                                    {"role": "system", "content": "あなたはデジタル広告の専門家です。データを分析し、実用的な示唆を提供してください。"},
                                    {"role": "user", "content": prompt}
                                ]
                            )

                            st.subheader(f"全体推移コメント(AI調整中)")
                            st.write(response.choices[0].message.content)

                            ############ CPA変化要因分析
                            results = analyze_cpa_change(media_results, 'Total')


    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("正しい情報を入力してください。")

else:
    st.info("Google Sheets API Key、SpreadsheetのID、シート名を入力してください。")