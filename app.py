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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# .env ファイルから環境変数を読み込む
load_dotenv()

st.title('データ分析 はまたろう')


#APIキーを環境変数から取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("SHEETS_API_KEY")

"""

OPENAI_API_KEY = st.secrets.AzureApiKey.OPENAI_API_KEY
API_KEY = st.secrets.AzureApiKey.SHEETS_API_KEY
"""


# スプレッドシートIDの入力
SPREADSHEET_ID = st.text_input("Google SpreadsheetのIDを入力してください", value="1BD-AEaNEWpPyzb5CySUc_XlNqWNIzu_1tC8C0g68Dpw")

# シート名の入力
SHEET_NAME = st.selectbox("シート名を入力してください", ("三井_オーダー領域", "三井_セレクト領域", "リンリン", "レグルス"))

# モデル選択のプルダウンを追加
gpt_model = st.selectbox(
    "使用するGPTモデルを選択してください",
    ("gpt-4o-mini", "none")
)

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
        # DataFrameの準備部分で月の情報を追加
        df['month'] = df['day'].dt.to_period('M').astype(str)

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

        # 月別データの集計を追加
        df_monthly = df.groupby(['media', 'month']).agg({
            'impression': 'sum',
            'click': 'sum',
            'cost': 'sum',
            'cv': 'sum'
        }).reset_index()
        df_monthly['cpa'] = df_monthly['cost'] / df_monthly['cv'].replace(0, 1)

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
                ('日別', '週別', '月別')
            )

            if analysis_type == '日別':
                df_filtered = df_daily_filtered
                x_axis = 'day'
            elif analysis_type == '週別':
                df_filtered = df_weekly_filtered
                x_axis = 'week'
            else:  # 月別の場合
                df_filtered = df_monthly[df_monthly['month'].isin(
                    df['month'][(df['day'] >= pd.Timestamp(date_range[0])) & 
                            (df['day'] <= pd.Timestamp(date_range[1]))]
                )]
                x_axis = 'month'


            def create_analysis_prompt(pattern_name, overall_text, all_results, media_text):
                prompt = f"""
                    #広告パフォーマンス分析レポート（{pattern_name}）
                    本分析では、パフォーマンスの変化とその要因を3段階で特定し、各階層での重要な発見を報告。

                    #インプットデータ
                    ##1. 全体パフォーマンス指標
                    {overall_text}

                    ##2. 領域別貢献度分析
                    {all_results}

                    ##3. KPI要因分解データ
                    {media_text}

                    #分析フレームワーク
                    ##第1階層：全体数値評価
                    目的：全体のパフォーマンス変化が事業に与える影響を定量化
                    分析項目：
                    - CV指標：CV数の絶対値と変化率
                    - CPA指標：CPAの絶対値と変化率
                    →良し悪しの判断と、それを裏付けるCV/CPA推移を記載

                    ##第2階層：貢献度分析
                    目的：パフォーマンス変化の主要因領域を特定
                    分析項目：
                    - プラス貢献領域：CV貢献度上位１〜３つの特定と変化幅
                    - マイナス貢献領域：CV貢献度下位１〜３つの特定と変化幅
                    →各領域のCV/CPA変化を絶対値と比率で記載

                    ##第3階層：KPI要因分解
                    目的：特定領域の変化要因を分解
                    分析項目：
                    - CV変動要因：CVR、CTR、CPCの変化率
                    - CPA変動要因：CPM、CTR、CPC、CVRの変化率
                    →主要因とその変化率を記載

                    #出力要件
                    ■アウトプット形式
                    1. 全体数値評価
                    - 全体評価：[良/悪の判断]
                    - 根拠指標：
                    CV：[絶対値]の変化（[変化率]%）
                    CPA：[絶対値]の変化（[変化率]%）

                    2. 貢献度分析
                    - プラス貢献：[領域名]（CV貢献度[数値]%）
                    CV：[絶対値]増（[変化率]%）
                    CPA：[絶対値]変化（[変化率]%）
                    - マイナス貢献：[領域名]（CV貢献度[数値]%）
                    CV：[絶対値]減（[変化率]%）
                    CPA：[絶対値]変化（[変化率]%）

                    3. 要因分解（対象：[領域名]）
                    - CV変動：[主要因指標]が[変化率]%変化
                    - CPA変動：[主要因指標]が[変化率]%変化
                    └ 要因分解：[CPM変化率]% / [CTR変化率]%

                    ■制約条件
                    - 無効値（0/inf/nan）は除外
                    - 事実ベースの記述のみ
                    - 体言止め形式を維持
                    - {pattern_name}基準での分析
                    - 全体で500文字以内
                    """
                return prompt


            # グラフ作成関数
            def create_stacked_bar(df, x, y, title):
                fig = go.Figure()
                for medium in df['media'].unique():
                    df_medium = df[df['media'] == medium]
                    fig.add_trace(go.Bar(x=df_medium[x], y=df_medium[y], name=medium))
                fig.update_layout(barmode='stack', title=title, xaxis_title=x, yaxis_title=y)
                fig.update_yaxes(tickformat=',d')  # Y軸を整数形式で表示
                return fig

            def create_line_chart(df, x, y, title):
                fig = go.Figure()
                
                # 各メディアのデータをプロット
                for medium in df['media'].unique():
                    df_medium = df[df['media'] == medium]
                    fig.add_trace(go.Scatter(
                        x=df_medium[x], 
                        y=df_medium[y], 
                        mode='lines+markers', 
                        name=medium,
                        hovertemplate=f"{medium}<br>{x}: %{{x}}<br>{y}: ¥%{{y:,.0f}}<extra></extra>"
                    ))
                
                # Total CPAの計算と追加
                df_total = df.groupby(x).agg({
                    'cost': 'sum',
                    'cv': 'sum'
                }).reset_index()
                df_total['total_cpa'] = df_total['cost'] / df_total['cv'].replace(0, 1)
                
                # Total CPAをプロット（赤紫色で表示）
                fig.add_trace(go.Scatter(
                    x=df_total[x],
                    y=df_total['total_cpa'],
                    mode='lines+markers',
                    name='Total CPA',
                    line=dict(
                        color='rgb(148, 0, 211)',  # 紫色
                        width=3,
                        dash='dot'
                    ),
                    marker=dict(
                        size=10,
                        symbol='diamond'
                    ),
                    hovertemplate="Total CPA<br>" +
                                f"{x}: %{{x}}<br>" +
                                "CPA: ¥%{y:,.0f}<br>" +
                                "CV: %{customdata[0]:,.0f}<br>" +
                                "Cost: ¥%{customdata[1]:,.0f}<extra></extra>",
                    customdata=np.column_stack((df_total['cv'], df_total['cost']))
                ))
                
                return fig
                                    
            def create_percentage_stacked_bar(df, x, y, title):
                """
                100%積み上げグラフを作成する関数
                
                Parameters:
                df (pd.DataFrame): データフレーム
                x (str): x軸の列名
                y (str): y軸の列名
                title (str): グラフのタイトル
                
                Returns:
                go.Figure: Plotlyのグラフオブジェクト
                """
                # 日付ごとの合計を計算
                totals = df.groupby(x)[y].sum().reset_index()
                
                fig = go.Figure()
                
                # 各メディアのデータを追加
                for medium in df['media'].unique():
                    df_medium = df[df['media'] == medium]
                    
                    # パーセンテージの計算
                    percentages = []
                    values = []  # 実数値を保存
                    for date in df_medium[x]:
                        total = totals[totals[x] == date][y].iloc[0]
                        value = df_medium[df_medium[x] == date][y].iloc[0]
                        percentage = (value / total) * 100 if total != 0 else 0
                        percentages.append(percentage)
                        values.append(value)
                    
                    # グラフにトレースを追加（name パラメータのみで凡例は自動的に表示される）
                    fig.add_trace(go.Bar(
                        x=df_medium[x],
                        y=percentages,
                        name=medium,  # これだけで凡例は自動的に表示される
                        hovertemplate=(
                            f"{medium}: %{{y:.0f}}%" +
                            f", %{{text}}" +
                            "<extra></extra>"
                        ),
                        text=[f"{v:,.0f}" for v in values]
                    ))
                
                # 基本的なレイアウトの設定
                fig.update_layout(
                    barmode='relative',  # 積み上げモード
                    title=title,
                    xaxis_title=x,
                    yaxis_title='Percentage (%)',
                    yaxis=dict(
                        tickformat='.0f',  # パーセンテージを整数で表示
                        range=[0, 100],  # y軸の範囲を0-100%に固定
                        dtick=20  # 20%ごとに目盛りを表示
                    ),
                    hovermode='x unified'  # ホバー時の表示モード
                )
                
                return fig



            # グラフ表示
            st.subheader(f"{analysis_type}分析結果")

            # ===========Cost推移グラフ===========
            st.subheader(f"{analysis_type}Cost推移")

            # タブの作成
            cv_tab1, cv_tab2 = st.tabs(["実数", "構成比(%)"])

            # 実数のタブ
            with cv_tab1:
                fig_cv = create_stacked_bar(df_filtered, x_axis, 'cost', '')
                fig_cv.update_layout(
                    height=400,
                    margin=dict(t=10),
                    yaxis_title="Cost"
                )
                st.plotly_chart(fig_cv, use_container_width=True)

            # 構成比のタブ
            with cv_tab2:
                fig_cv_percentage = create_percentage_stacked_bar(df_filtered, x_axis, 'cost', '')
                fig_cv_percentage.update_layout(
                    height=400,
                    margin=dict(t=10),
                    yaxis_title="構成比(%)"
                )
                st.plotly_chart(fig_cv_percentage, use_container_width=True)


            # ===========CV推移グラフ===========
            st.subheader(f"{analysis_type}CV推移")

            # タブの作成
            cv_tab1, cv_tab2 = st.tabs(["実数", "構成比(%)"])

            # 実数のタブ
            with cv_tab1:
                fig_cv = create_stacked_bar(df_filtered, x_axis, 'cv', '')
                fig_cv.update_layout(
                    height=400,
                    margin=dict(t=10),
                    yaxis_title="CV数"
                )
                st.plotly_chart(fig_cv, use_container_width=True)

            # 構成比のタブ
            with cv_tab2:
                fig_cv_percentage = create_percentage_stacked_bar(df_filtered, x_axis, 'cv', '')
                fig_cv_percentage.update_layout(
                    height=400,
                    margin=dict(t=10),
                    yaxis_title="構成比(%)"
                )
                st.plotly_chart(fig_cv_percentage, use_container_width=True)

            # CPA推移
            
            st.subheader(f"{analysis_type}CPA推移")
            fig_cpa = create_line_chart(df_filtered, x_axis, 'cpa', f'媒体別の{analysis_type}CPA推移')
            st.plotly_chart(fig_cpa)


            # 今日の日付を取得し、1日前の日付を計算
            from datetime import datetime, timedelta
            today = datetime.now().date()
            default_date = today - timedelta(days=1)

            st.subheader("日付比較分析")
            analysis_date = st.date_input("分析する日付を選択してください", min_value=start_date, max_value=end_date, value=default_date)

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
                                        f"CV差分: {row['delta_cv']:.0f}%\n"
                                        f"CV貢献度: {row['cv_contribution']:.0f}%\n"
                                        f"CPA差分: {row['delta_cpa']:.0f}%\n"
                                        f"CPA変化割合: {row['cpa_contribution']:.0f}%\n")
                    
                    # 表形式のデータを作成
                    table_data = media_results[['media', 'delta_cv', 'cv_contribution', 'delta_cpa', 'cpa_contribution']].copy()
                    table_data.columns = ['メディア','CV差分','CV貢献度(%)', 'CPA差分','CPA変化割合(%)']
                                        
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
                    merged['cv_contribution'] = ((merged['delta_cv'] / overall_cv_diff) * 100).round()

                    overall_cpa_diff = (merged['cpa_2'] - merged['cpa_1']).round()
                    merged['cpa_contribution'] = ((merged['cpa_2'] / merged['cpa_1']) * 100).round()

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


                def format_dataframe(df):
                    # インデックスと列名の重複をチェック
                    if df.index.duplicated().any():
                        print("重複しているインデックスがあります:")
                        duplicated_indices = df.index[df.index.duplicated()].unique()
                        for idx in duplicated_indices:
                            locs = df.index.get_loc(idx)
                            print(f'重複しているインデックス: {idx} - 行番号: {locs}')
                            if isinstance(locs, slice):
                                print(f'行名: {df.index[locs]}')
                            else:
                                print(f'行名: {df.index[locs].tolist()}')
                        # 重複しているインデックスを修正
                        df = df.reset_index(drop=True)
                    
                    if df.columns.duplicated().any():
                        print("重複している列名があります:")
                        duplicated_columns = df.columns[df.columns.duplicated()].unique()
                        for col in duplicated_columns:
                            duplicated_locs = [i for i, x in enumerate(df.columns) if x == col]
                            print(f'重複している列名: {col} - 列番号: {duplicated_locs}')
                        # 重複している列名を修正
                        df.columns = [f'{col}_{i}' if df.columns.tolist().count(col) > 1 else col 
                                    for i, col in enumerate(df.columns)]

                    format_dict = {
                        'impression': '{:,.0f}',
                        'click': '{:,.0f}',
                        'cpm': '¥{:.0f}',
                        'ctr': '{:.1f}%',
                        'cpc': '¥{:.0f}',
                        'cost': '¥{:,.0f}',
                        'cv': '{:,.0f}',
                        'cvr': '{:.2f}%',
                        'cpa': '¥{:,.0f}'
                    }
                    
                    def background_color_heatmap(col):
                        cmap = plt.get_cmap('RdYlGn_r')
                        
                        # 数値型に変換を試みる
                        numeric_col = pd.to_numeric(col, errors='coerce')
                        
                        # NaN値とinf値を除外してランクを計算
                        valid_data = numeric_col.dropna()
                        ranks = valid_data.rank(pct=True)
                        
                        colors = []
                        font_colors = []

                        for i, x in enumerate(numeric_col):
                            if pd.isna(x) or np.isinf(x):
                                colors.append('#808080')
                                font_colors.append('white')  # デフォルトで文字色は白
                            else:
                                color = mcolors.rgb2hex(cmap(ranks.get(i, 0)))
                                colors.append(color)
                                
                                # 背景色が薄い場合は文字色を黒に設定
                                r, g, b = mcolors.hex2color(color)
                                luminance = 0.299*r + 0.587*g + 0.114*b
                                font_colors.append('black' if luminance > 0.5 else 'white')

                        return [f'background-color: {color}; color: {font_color}' for color, font_color in zip(colors, font_colors)]

                    return (df.style
                            .format(format_dict)
                            .apply(background_color_heatmap, subset=['cost', 'cv', 'cpa'])
                    )

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
                    st.write(f"{media}の媒体比較データ")
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

                    # DataFrameの列を並べ替え
                    desired_columns = ['period', 'impression', 'click', 'cpm', 'ctr','cpc', 'cost', 'cv', 'cvr', 'cpa']
                    display_data = display_data.reindex(columns=desired_columns)

                    st.dataframe(format_dataframe(display_data))
                    
                    return results
                
                # 3つの分析パターンの結果を格納するリスト
                all_pattern_results = []

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

                            # DataFrameの列を並べ替え
                            desired_columns = ['media', 'impression', 'click', 'cpm', 'ctr','cpc', 'cost', 'cv', 'cvr', 'cpa']
                            data1_with_total = data1_with_total.reindex(columns=desired_columns)
                            data2_with_total = data2_with_total.reindex(columns=desired_columns)

                            st.write(f"期間1: {dates[0]} から {dates[1] if len(dates) > 2 else dates[0]}")
                            st.dataframe(format_dataframe(data1_with_total).apply(lambda x: ['background-color: gray' if x.name == len(x) else '' for i in x],axis=1))


                            st.write(f"期間2: {dates[2] if len(dates) > 2 else dates[1]} から {dates[3] if len(dates) > 2 else dates[1]}")
                            st.dataframe(format_dataframe(data2_with_total).apply(lambda x: ['background-color: gray' if x.name == len(x) else '' for i in x],axis=1))

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
                            cv_cpa_table['cv_contribution'] = (cv_cpa_table['cv_contribution']).round(0)
                            cv_cpa_table['delta_cpa'] = cv_cpa_table['delta_cpa'].round(0)
                            cv_cpa_table['cpa_contribution'] = (cv_cpa_table['cpa_contribution']).round(0)

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

                            st.write('cv, cpaの媒体毎変化量')
                            st.info("CV貢献度:全体のCV変化数に対する、対象領域のCV変化幅がどの程度影響を与えているかを示しています")

                            st.dataframe(cv_cpa_table.style.format({
                                    'CV変化量': '{:.0f}',
                                    'CV貢献度': '{:.1f}%',
                                    'CPA変化量': '{:.0f}',
                                    'CPA変化率': '{:.1f}%'
                                })
                            )

                            # 棒グラフの作成
                            def create_metric_bar_charts(data):
                                metrics = ['CV貢献度', 'CV変化量','CPA変化量', 'CPA変化率']
                                tabs = st.tabs(metrics)
                                
                                for metric, tab in zip(metrics, tabs):
                                    with tab:
                                        fig = go.Figure()
                                        
                                        # 値に基づいて色を設定
                                        colors = ['green' if val >= 0 else 'red' for val in data[metric]]
                                        
                                        # メディアごとの値を棒グラフで表示
                                        fig.add_trace(go.Bar(
                                            x=data['メディア'],
                                            y=data[metric],
                                            text=data[metric].apply(lambda x: f'{x:,.1f}' + ('%' if '率' in metric or '貢献度' in metric else '')),
                                            textposition='auto',
                                            marker_color=colors,  # 色を適用
                                        ))
                                        
                                        # レイアウトの設定
                                        fig.update_layout(
                                            title=f'メディア別 {metric}',
                                            xaxis_title='メディア',
                                            yaxis_title=metric,
                                            height=400,
                                            showlegend=False,
                                            # y軸のグリッド線を表示
                                            yaxis=dict(
                                                showgrid=True,
                                                gridwidth=1,
                                                gridcolor='LightGray'
                                            )
                                        )
                                        
                                        # ゼロラインを強調表示
                                        fig.add_hline(y=0, line_width=1, line_color="black")
                                        
                                        st.plotly_chart(fig, use_container_width=True)

                            # 棒グラフの表示
                            create_metric_bar_charts(cv_cpa_table)


                            overall_text, overall_table = format_overall_results(overall_results)
                            media_text, media_table = format_media_results(media_results)



                            ############ CPA変化要因分析
                            def analyze_top_contributors(media_results):
                                
                                # 'CV貢献度'と'CPA変化率'列を数値型に変換し、infとnanを除外
                                media_results['CV貢献度'] = pd.to_numeric(media_results['CV貢献度'], errors='coerce')
                                media_results['CPA変化率'] = pd.to_numeric(media_results['CPA変化率'], errors='coerce')                                
                                # infとnanを除外
                                media_results_filtered = media_results[~media_results['CV貢献度'].isin([np.inf, -np.inf, np.nan]) & 
                                                                    ~media_results['CPA変化率'].isin([np.inf, -np.inf, np.nan])]

                                
                                # CV貢献度の上位5メディアを抽出
                                top_cv = media_results_filtered.nlargest(5, 'CV貢献度')['メディア'].tolist()
                                
                                # CPA変化率の上位5メディアを抽出
                                top_cpa = media_results_filtered.nlargest(5, 'CPA変化率')['メディア'].tolist()
                                
                                # 重複を除去して統合
                                top_media = list(dict.fromkeys(top_cv + top_cpa))
                                
                                return top_media
                            
                            top_contributors_medias = analyze_top_contributors(cv_cpa_table)

                            # タブを作成
                            tabs = st.tabs(top_contributors_medias)

                            # 全メディアの結果を格納する文字列
                            all_results = "#media別 数値改善幅（-は悪化）\n"
                                
                            # 各タブの内容を設定
                            for i, media in enumerate(top_contributors_medias):
                                with tabs[i]:
                                    results = analyze_cpa_change(media_results, media)
            
                                    # 各数値を (results数値 - 1) × 100 に変換する関数
                                    def transform(value):
                                        return (value - 1) * 100

                                    all_results += f"{media}\n"
                                    all_results += f"CPAの改善幅:{transform(results['cpa_impacts']['CPA']):.1f}% に対して"
                                    all_results += f" CVR:{transform(results['cpa_impacts']['CVR']):.1f}%"
                                    all_results += f" CPC:{transform(results['cpa_impacts']['CPC']):.1f}%\n"
                                    all_results += f"CPCの改善幅:{transform(results['cpc_impacts']['CPC']):.1f}% に対して"
                                    all_results += f" CPM:{transform(results['cpc_impacts']['CPM']):.1f}%"
                                    all_results += f" CTR:{transform(results['cpc_impacts']['CTR']):.1f}%\n\n"

                            # 全結果を表示
                            print("---overall_text")
                            print(overall_text)
                            print("---media_text")
                            print(media_text)
                            print("---all_results")
                            print(all_results)


                            prompt = create_analysis_prompt(pattern_name, overall_text, media_text, all_results)
                            print(prompt)

                            
                            response = client.chat.completions.create(
                                model=gpt_model,  # ここでプルダウンで選択されたモデルを使用
                                messages=[
                                    {"role": "system", "content": "あなたはデジタル広告の専門家です。データを分析し、実用的な示唆を提供してください。"},
                                    {"role": "user", "content": prompt}
                                ]
                            )

                            st.subheader(f"全体推移コメント")
                            st.write(response.choices[0].message.content)

                            # 各パターンの分析結果を保存
                            all_pattern_results.append({
                                "pattern_name": pattern_name,
                                "overall_text": overall_text,
                                "media_text": media_text,
                                "ai_analysis": response.choices[0].message.content
                            })

               # 3つのパターンの結果を統合してAIに分析させる
                st.subheader("総合分析結果")

                combined_prompt = "以下の3つの期間における広告パフォーマンスの分析結果を基に、最近の広告数値の推移を総合的に分析し、簡潔にまとめてください。\n\n"

                for result in all_pattern_results:
                    combined_prompt += f"# {result['pattern_name']}\n"
                    combined_prompt += f"全体推移: {result['overall_text']}\n"
                    combined_prompt += f"メディア別推移: {result['media_text']}\n"
                    combined_prompt += f"AI分析: {result['ai_analysis']}\n\n"

                combined_prompt += """
                上記の情報を元に、以下の点について300-400字程度で総合的に分析してください：
                1. 全体的なトレンド（CV数とCPAの推移）
                2. 主要なメディアの貢献度の変化
                3. 短期的（前日比）と中期的（1週間、2週間）な変化の違い
                4. 注目すべき特徴的な動き

                分析は事実に基づいて行い、推測や提案は避けてください。"""

                combined_response = client.chat.completions.create(
                    model=gpt_model,
                    messages=[
                        {"role": "system", "content": "あなたはデジタル広告の専門家です。複数の期間のデータを総合的に分析し、簡潔かつ洞察力のある分析を提供してください。"},
                        {"role": "user", "content": combined_prompt}
                    ]
                )

                st.write(combined_response.choices[0].message.content)


    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("正しい情報を入力してください。")

else:
    st.info("Google Sheets API Key、SpreadsheetのID、シート名を入力してください。")