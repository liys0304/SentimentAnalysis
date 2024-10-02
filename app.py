from flask import Flask, request, jsonify, render_template

from sentiment_statistic import sentiment_analysis

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # 返回 HTML 文件

@app.route('/analyze')
def analyze():
    topic = request.args.get('topic')

    # 这里应调用您的分析代码并生成图像
    conclusion = sentiment_analysis(topic)
    # 示例返回随机图片（您需要替换为真实的生成逻辑）
    trend_image = "http://127.0.0.1:5000/static/trend_image.png"  # 替换为实际图像URL
    sentiment_image = "http://127.0.0.1:5000/static/sentiment_image.png"  # 替换为实际图像URL

    # 结果说明
    explanation = f"关于 '{topic}' 的舆情分析结果如下。\n{conclusion}"

    return jsonify({
        'trend_image': trend_image,
        'sentiment_image': sentiment_image,
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(debug=True)
