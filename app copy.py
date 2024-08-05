import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI 
from dotenv import load_dotenv
import os
import requests
from datetime import timedelta

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

            # 新しい分析機能
            st.subheader("日付比較分析")
            date1 = st.date_input("比較する1つ目の日付を選択してください", min_value=start_date, max_value=end_date)
            date2 = st.date_input("比較する2つ目の日付を選択してください", min_value=start_date, max_value=end_date)

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
                    total = data.sum(numeric_only=True)
                    total['media'] = 'Total'
                    total['ctr'] = (total['click'] / total['impression']) * 100
                    total['cpc'] = total['cost'] / total['click']
                    total['cvr'] = (total['cv'] / total['click']) * 100
                    total['cpa'] = total['cost'] / total['cv']
                    return pd.concat([data, total.to_frame().T])
                

                # 分析実行
                data1 = get_date_data(df, date1)
                data2 = get_date_data(df, date2)

                # 合計行を追加
                data1_with_total = add_total_row(data1)
                data2_with_total = add_total_row(data2)
                
                st.subheader(f"{date1}のデータ")
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

                st.subheader(f"{date2}のデータ")
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

                if data1.empty or data2.empty:
                    st.error(f"エラー: {date1}または{date2}のデータが見つかりません。")
                else:
                    total1 = calculate_total_metrics(data1)
                    total2 = calculate_total_metrics(data2)

                    overall_results = analyze_overall(total1, total2)
                    if overall_results is not None:
                        overall_analysis = format_overall_results(overall_results)

                        media1 = calculate_media_metrics(data1)
                        media2 = calculate_media_metrics(data2)

                        media_results = analyze_media_contribution(media1, media2, overall_results)
                        media_analysis = format_media_results(media_results)

                        prompt = (f"以下に広告の配信結果の{date1}と{date2}の差分を示します。このデータを分析し、以下の指示に従って簡潔に記述してください。\n\n"
                                  f"#プロモーション全体の推移\n{overall_analysis}\n"
                                  f"#メディア別の差分\n{media_analysis}\n"
                                  "1. 全体のCV数とCPAの変化について簡潔に述べてください。\n"
                                  "2. 各メディアについて、以下の点を簡潔に記述してください：\n"
                                  "   a) CV数の変化とその全体に対する相対的な寄与度\n"
                                  "   b) CPAの変化とその全体に対する相対的な寄与度\n"
                                  "3. メディア間で相対的に大きな変化や顕著な違いがある場合、それを指摘してください。\n"
                                  "4. 値が0やinfになっている項目については言及しないでください。\n"
                                  "5. 各メディアの変化について、他のメディアと比較した相対的な位置づけを述べてください。\n"
                                  "6. 分析は事実の記述に留め、推測や提案は含めないでください。\n"
                                  "7. 前日対比のデータであることを前提に分析してください。\n")

                        st.subheader("分析結果")
                        # 全体結果の表示
                        overall_text, overall_table = format_overall_results(overall_results)
                        st.dataframe(overall_table.style.format({
                            '絶対値': '{:.2f}',
                            '変化率(%)': '{:.2f}%'
                        }))

                        # メディア別結果の表示
                        media_text, media_table = format_media_results(media_results)
                        st.dataframe(media_table.style.format({
                            'CV貢献度(%)': '{:.2f}%',
                            'CPA貢献度(%)': '{:.2f}%'
                        }))


                        # OpenAI APIを使用して分析結果を生成
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "あなたはデジタル広告の専門家です。データを分析し、実用的な示唆を提供してください。"},
                                {"role": "user", "content": prompt}
                            ]
                        )

                        st.subheader("AI分析結果")
                        st.write(response.choices[0].message.content)

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.error("正しい情報を入力してください。")

else:
    st.info("Google Sheets API Key、SpreadsheetのID、シート名を入力してください。")