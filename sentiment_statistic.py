import pandas as pd
from wb_spider import spider
import matplotlib.pyplot as plt


def sentiment_analysis(topic):
    user_input = topic
    spider(user_input)

    # 假设你有一个包含时间戳的数据集
    data = pd.read_csv(f'topic_tmp/{user_input}.csv')
    # 假设数据集有一个 'timestamp' 列，包含日期时间
    data['publish_time'] = pd.to_datetime(data['publish_time'])  # 转换为日期时间格式

    # 按天分组
    data_by_day = data.groupby(data['publish_time'].dt.date)

    # 获取当前日期
    current_date = pd.Timestamp.now()

    # 计算过去五天的日期范围
    five_days_ago = current_date - pd.Timedelta(days=5)

    # 过滤出过去五天的数据
    last_five_days = data[data['publish_time'] >= five_days_ago]

    categories = ['pos', 'neg', 'neu']
    # 按天统计每类情感的数量
    daily_sentiment_count = last_five_days.groupby(
        [last_five_days['publish_time'].dt.date, 'predicted_class']).size().unstack(fill_value=0)
    daily_total_comments = daily_sentiment_count.sum(axis=1)
    # 如果某些天缺少某个类别，添加缺失类别的列，并填充为0
    for category in categories:
        if category not in daily_sentiment_count.columns:
            daily_sentiment_count[category] = 0

    # 计算每一天情感类别的比例
    daily_sentiment_ratio = daily_sentiment_count.div(daily_sentiment_count.sum(axis=1), axis=0) * 100
    daily_sentiment_ratio['total_comments'] = daily_total_comments

    positive_change = daily_sentiment_ratio['pos'].diff()
    negative_change = daily_sentiment_ratio['neg'].diff()
    neutral_change = daily_sentiment_ratio['neu'].diff()
    comments_change = daily_sentiment_ratio['total_comments'].diff()
    # 预测未来情感比例 (假设变化趋势保持不变)
    future_positive = daily_sentiment_ratio['pos'][-1] + positive_change[-1]
    future_negative = daily_sentiment_ratio['neg'][-1] + negative_change[-1]
    future_neutral = daily_sentiment_ratio['neu'][-1] + neutral_change[-1]
    future_comment = daily_sentiment_ratio['total_comments'][-1] + comments_change[-1]

    # 绘制折线图
    plt.figure(figsize=(12, 6))
    for category in daily_sentiment_ratio.columns[:-1]:  # 排除总评论数
        plt.plot(daily_sentiment_ratio.index, daily_sentiment_ratio[category], marker='o', label=category)

    plt.title("variation in past five days")
    plt.xlabel("date")
    plt.ylabel("ratio")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/trend_image.png')
    plt.close()

    # 绘制柱状图
    daily_sentiment_count.plot(kind='bar', figsize=(12, 6))
    plt.title("statistics data in past five days")
    plt.xlabel("date")
    plt.ylabel("total comments")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/sentiment_image.png')
    plt.close()

    # 预测数据
    analyze = ""
    if future_negative < 0 and future_positive > 0:
        analyze += "舆论趋于积极方向，"
    elif future_negative > 0 and future_positive < 0:
        analyze += "舆论趋于消极方向，"
    elif future_neutral > 0 and future_positive > 0:
        analyze += "舆论情况正在好转，"
    elif future_negative > 0 and future_negative > 0:
        analyze += "舆论正在持续发酵，"
    else:
        analyze += "舆论趋于平缓，"
    if future_comment > 0:
        analyze += "热度正在上升。"
    elif future_comment < 0:
        analyze += "热度正在下降。"
    else:
        analyze += "热度处于平缓阶段"
    return analyze