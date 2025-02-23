import pandas as pd
import os

def get_trade_signals(result_file):
    """
    从结果 CSV 文件中提取最强和最弱合约及交易信号。
    
    参数:
    - result_file: CSV 文件路径（如 '../results/result_group_1_20250220_123456.csv'）
    
    返回:
    - dict: 包含买单和卖单的合约代码及附加信息
    """
    # 读取 CSV 文件
    df = pd.read_csv(result_file)
    
    # 提取最强和最弱合约
    strongest_row = df[df['is_strongest'] == True].iloc[0]
    weakest_row = df[df['is_weakest'] == True].iloc[0]
    strongest = strongest_row['symbol']
    weakest = weakest_row['symbol']
    market_direction = df['market_direction'].iloc[0]
    timestamp = df['timestamp'].iloc[0]
    
    # 根据市场方向生成交易信号
    if market_direction == 'up':
        trade_signals = {
            'buy': strongest,    # 上涨行情做多最强合约
            'sell': weakest,     # 上涨行情做空最弱合约
            'timestamp': timestamp,
            'group_id': df['group_id'].iloc[0]
        }
    elif market_direction == 'down':
        trade_signals = {
            'buy': strongest,    # 下跌行情做多最强（跌得慢）
            'sell': weakest,     # 下跌行情做空最弱（跌得快）
            'timestamp': timestamp,
            'group_id': df['group_id'].iloc[0]
        }
    else:
        trade_signals = {
            'buy': None,         # 震荡行情不交易
            'sell': None,
            'timestamp': timestamp,
            'group_id': df['group_id'].iloc[0]
        }
    
    # 添加额外信息供查看
    if 'prediction' in df.columns:
        trade_signals['strongest_prediction'] = strongest_row['prediction']
        trade_signals['weakest_prediction'] = weakest_row['prediction']
    elif 'score' in df.columns:
        trade_signals['strongest_score'] = strongest_row['score']
        trade_signals['weakest_score'] = weakest_row['score']
    
    return trade_signals

# 示例使用
if __name__ == "__main__":
    # 假设结果文件路径
    result_file = "../results/result_group_1_20250220_123456.csv"
    
    # 获取交易信号
    signals = get_trade_signals(result_file)
    print(f"交易信号: Buy={signals['buy']}, Sell={signals['sell']}")
    print(f"时间: {signals['timestamp']}, 组号: {signals['group_id']}")
    if 'prediction' in signals:
        print(f"最强预测: {signals['strongest_prediction']}, 最弱预测: {signals['weakest_prediction']}")
    elif 'score' in signals:
        print(f"最强得分: {signals['strongest_score']}, 最弱得分: {signals['weakest_score']}")
    
    # 模拟交易
    if signals['buy']:
        print(f"执行买入: {signals['buy']}")
    if signals['sell']:
        print(f"执行卖出: {signals['sell']}")