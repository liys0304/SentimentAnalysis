import streamlit as st
from IPython.core.pylabtools import figsize
from requests.exceptions import ProxyError
import pandas as pd
from tensorflow.python.data.experimental.ops.testing import sleep
import time
import wb_spider
from wb_spider import spider, daily_topic
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#错误处理
if "error" not in st.session_state:
    st.session_state.error = ""

def delete_temp_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        return

def clear_error():
    st.session_state.error = ""

try:
    st.set_page_config(layout="wide")
    st.title("微博话题情感分析预测系统 :sunglasses:")
    with st.spinner("加载中，请稍等"):
        texts = daily_topic()
    font = r'C:\Windows\Fonts\Deng.ttf'
    word_cloud_topic = WordCloud(font_path=font,
                                 background_color='white',
                                 width=1200,
                                 height=800
                                 ).generate(texts)
    st.header("实时热搜话题：", divider="gray")
    fig, ax = plt.subplots(dpi=500)
    ax.imshow(word_cloud_topic, interpolation='bilinear')
    ax.axis("off")
    fig.savefig("./src/wordcloud.png", dpi=500)
    st.image("./src/wordcloud.png", width=700)
    with st.sidebar:
        st.write("请不要开代理运行网页！")
        
    # 输入框，on_change 触发清除报错
    with st.form(key='input_style'):
        user_input = st.text_input("输入一个感兴趣的话题：")
        submit_button = st.form_submit_button("确定")
        max_count = st.select_slider('选择想要爬取的数据数量,不建议超过200（慢）', range(50, 501))

    if user_input and submit_button:
        progress_bar = st.progress(0)
        file_path = f"./topic_tmp/{user_input}.csv"
        delete_temp_file(file_path)
        spider(user_input, max_count, progress_bar)
        st.session_state.error = ""

        data = pd.read_csv(f'topic_tmp/{user_input}.csv')
        data['publish_time'] = pd.to_datetime(data['publish_time'])

        five_days_ago = pd.Timestamp.now() - pd.Timedelta(days=5)
        data_days = data[data['publish_time'] >= five_days_ago]
        data_days['day_group'] = data_days['publish_time'].dt.date

        seventy_two_hours_ago = pd.Timestamp.now() - pd.Timedelta(hours=72)
        data_hours = data[data['publish_time'] >= seventy_two_hours_ago]
        data_hours['hour_group'] = data_hours['publish_time'].dt.floor('H')

        predict_by_dic_hours = data_hours.groupby('hour_group')['predict_by_dic'].value_counts(
             normalize=False).unstack().fillna(0)
        predict_by_bert_hours = data_hours.groupby('hour_group')['predict_by_bert'].value_counts(
             normalize=False).unstack().fillna(0)
        predict_by_dic_days = data_days.groupby('day_group')['predict_by_dic'].value_counts(
             normalize=False).unstack().fillna(0)
        predict_by_bert_days = data_days.groupby('day_group')['predict_by_bert'].value_counts(
             normalize=False).unstack().fillna(0)

        max_count = len(data)
        if max_count > 100:
            sampled_data = data.sample(n=min(50, max_count), random_state=42)
        else:
            sampled_data = data.sample(n=min(20, max_count), random_state=42)
        st.write(f"#{user_input}#的分析结果如下")
        hourly, daily, total, comment= st.tabs(["数据折线图", "数据柱状图","总览", "评论预览"])
        with hourly:
            dic, bert = st.columns(2)
            with dic:
                st.write("基于字典的分析结果：")
                st.line_chart(predict_by_dic_hours)
            with bert:
                st.write("基于BERT的分析结果：")
                st.line_chart(predict_by_bert_hours)
        with daily:
            dic, bert = st.columns(2)
            with dic:
                st.write("基于字典的分析结果：")
                st.bar_chart(predict_by_dic_hours)
            with bert:
                st.write("基于BERT的分析结果：")
                st.bar_chart(predict_by_bert_hours)
        with total:
            dic, bert = st.columns(2)
            colors = ["snow", "lightsalmon", "palegreen", "lightskyblue", "grey", ]
            dic_counts = data['predict_by_dic'].value_counts()
            bert_counts = data['predict_by_bert'].value_counts()
            with dic:
                st.write("基于字典的分析结果：")
                fig1, ax1 = plt.subplots()
                ax1.pie(dic_counts, labels=dic_counts.index, autopct='%1.1f%%', startangle=90,
                        colors=colors)
                ax1.axis('equal')
                st.pyplot(fig1)
            with bert:
                st.write("基于BERT的分析结果：")
                fig2, ax2 = plt.subplots()
                ax2.pie(bert_counts, labels=bert_counts.index, autopct='%1.1f%%', startangle=90,
                        colors=colors)
                ax2.axis('equal')
                st.pyplot(fig2)
        with comment:
            st.dataframe(sampled_data)

except PermissionError:
    st.session_state.error = "请先不要打开爬取的excel文件！"
except ProxyError:
    st.session_state.error = "请关闭代理后再试！"
except ConnectionError as e:
    st.session_state.error = "爬取过多，网络访问出错，请稍后再试！"

# 显示报错信息
if st.session_state.error:
    st.error(st.session_state.error)
