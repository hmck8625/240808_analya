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

# .env ファイルから環境変数を読み込む
load_dotenv()

# OpenAI APIキーを環境変数から取得
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAIクライアントの初期化
client = OpenAI(api_key=openai_api_key)

st.title('デジタル広告データ分析アプリ')

# Google Sheets API キーの入力
API_KEY = st.text_input("Google Sheets API Keyを入力してください", type="password", value="AIzaSyBP9qP9XZh1Nm2jsi_MvcWKmTaVNM6F-7A")

# スプレッドシートIDの入力
SPREADSHEET_ID = st.text_input("Google SpreadsheetのIDを入力してください", value="1BD-AEaNEWpPyzb5CySUc_XlNqWNIzu_1tC8C0g68Dpw")

# シート名の入力
SHEET_NAME = st.text_input("シート名を入力してください", value="シート1")

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
                    data['ctr'] = (data['click'] / data['impression']) * 100
                    data['cpc'] = data['cost'] / data['click']
                    data['cvr'] = (data['cv'] / data['click']) * 100
                    data['cpa'] = data['cost'] / data['cv']
                    return data[['day', 'media', 'impression', 'click', 'ctr', 'cpc', 'cost', 'cv', 'cvr', 'cpa']]

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
                    table_data = media_results[['media', 'cv_contribution', 'cpa_contribution']].copy()
                    table_data.columns = ['メディア', 'CV貢献度(%)', 'CPA貢献度(%)']
                    
                    # Total行を追加
                    total_row = pd.DataFrame({
                        'メディア': ['Total'],
                        'CV貢献度(%)': [table_data['CV貢献度(%)'].sum()],
                        'CPA貢献度(%)': [table_data['CPA貢献度(%)'].sum()]
                    })
                    table_data = pd.concat([table_data, total_row], ignore_index=True)
                    
                    return text_output, table_data

                def add_total_row(data):
                    # 数値列のみを合計
                    total = data.select_dtypes(include=[np.number]).sum()
                    
                    # 'media' 列に 'Total' を設定
                    total['media'] = 'Total'
                    
                    # 計算が必要な指標を正しく計算
                    total['ctr'] = (total['click'] / total['impression']) * 100 if total['impression'] != 0 else 0
                    total['cpc'] = total['cost'] / total['click'] if total['click'] != 0 else 0
                    total['cvr'] = (total['cv'] / total['click']) * 100 if total['click'] != 0 else 0
                    total['cpa'] = total['cost'] / total['cv'] if total['cv'] != 0 else 0
                    
                    # 元のデータと合計行を結合し、インデックスをリセット
                    return pd.concat([data, total.to_frame().T]).reset_index(drop=True)
                
                def get_date_range_data(df, start_date, end_date):
                    mask = (df['day'] >= pd.to_datetime(start_date)) & (df['day'] <= pd.to_datetime(end_date))
                    data = df[mask].copy()
                    data['ctr'] = (data['click'] / data['impression']) * 100
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
                    data['cpc'] = data['cost'] / data['click']
                    data['cvr'] = (data['cv'] / data['click']) * 100
                    data['cpa'] = data['cost'] / data['cv']
                    return data

                def analyze_comparison(data1, data2):
                    merged = pd.merge(data1, data2, on='media', suffixes=('_1', '_2'))
                    merged['delta_cv'] = merged['cv_2'] - merged['cv_1']
                    merged['delta_cpa'] = merged['cpa_2'] - merged['cpa_1']
                    total_cost = merged['cost_1'].sum()
                    merged['cost_volume'] = merged['cost_1'] / total_cost
                    overall_cv_diff = merged['delta_cv'].sum()
                    overall_cpa_diff = (merged['cpa_2'] * merged['cost_volume']).sum() - (merged['cpa_1'] * merged['cost_volume']).sum()
                    merged['cv_contribution'] = (merged['delta_cv'] / overall_cv_diff) * 100
                    merged['cpa_contribution'] = (merged['delta_cpa'] / overall_cpa_diff) * merged['cost_volume'] * 100
                    return merged, overall_cv_diff, overall_cpa_diff

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
                          }).apply(lambda x: ['background-color: yellow' if x.name == len(x) - 1 else '' for i in x], axis=1))


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
                          }).apply(lambda x: ['background-color: yellow' if x.name == len(x) - 1 else '' for i in x], axis=1))

                          media_results, overall_cv_diff, overall_cpa_diff = analyze_comparison(data1, data2)

                          overall_results = {
                              'cv_diff': overall_cv_diff,
                              'cv_ratio': ((data2['cv'].sum() / data1['cv'].sum()) - 1) * 100 if data1['cv'].sum() != 0 else float('inf'),
                              'cpa_diff': (data2['cost'].sum() / data2['cv'].sum()) - (data1['cost'].sum() / data1['cv'].sum()) if data1['cv'].sum() != 0 and data2['cv'].sum() != 0 else float('inf'),
                              'cpa_ratio': (((data2['cost'].sum() / data2['cv'].sum()) / (data1['cost'].sum() / data1['cv'].sum())) - 1) * 100 if data1['cv'].sum() != 0 and data2['cv'].sum() != 0 else float('inf')
                          }

                          overall_text, overall_table = format_overall_results(overall_results)
                          st.dataframe(overall_table.style.format({
                              '絶対値': '{:.2f}',
                              '変化率(%)': '{:.2f}%'
                          }))

                          media_text, media_table = format_media_results(media_results)
                          st.dataframe(media_table.style.format({
                              'CV貢献度(%)': '{:.2f}%',
                              'CPA貢献度(%)': '{:.2f}%'
                          }))

                          prompt = (f"以下に広告の配信結果の{pattern_name}を示します。このデータを分析し、以下の指示に従って簡潔に記述してください。\n\n"
                                    f"#プロモーション全体の推移\n{overall_text}\n"
                                    f"#メディア別の差分\n{media_text}\n"
                                    f"#分析FMT"
                                    "全体変化:全体のCV数とCPAの変化について簡潔に述べてください。\n"
                                    "メディア別変化箇所：CV貢献度とCPA貢献から、全体の変化の要因となっているメディアを簡潔に教えてください\n"
                                    "#注意事項\n"
                                    "・値が0やinf, nanになっている項目については言及しないでください。\n"
                                    "・分析は事実の記述に留め、推測や提案は含めないでください。\n"
                                    f"・{pattern_name}のデータであることを前提に分析してください。")

                          response = client.chat.completions.create(
                              #model="gpt-3.5-turbo",
                              model="gpt-4-turbo",

                              messages=[
                                  {"role": "system", "content": "あなたはデジタル広告の専門家です。データを分析し、実用的な示唆を提供してください。"},
                                  {"role": "user", "content": prompt}
                              ]
                          )

                          st.subheader(f"AIコメント")
                          st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("正しい情報を入力してください。")

else:
    st.info("Google Sheets API Key、SpreadsheetのID、シート名を入力してください。")