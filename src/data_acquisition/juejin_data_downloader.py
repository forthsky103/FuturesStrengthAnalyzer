# src/data_acquisition/juejin_data_downloader.py

import pandas as pd
from gm.api import set_token, history
from typing import Optional
import os
from datetime import datetime

# 设置掘金量化平台的 API Token
API_TOKEN = 'f0609484d5bdc4f7bdf8f7d11d21fb6f8e7cace0'
set_token(API_TOKEN)

def generate_filename(symbol: str, data_type: str, start_time: str, end_time: str) -> str:
    """
    根据参数自动生成文件名

    Args:
        symbol: 合约代码，例如 'SHFE.rb2505'
        data_type: 数据类型，例如 '5m', 'tick'
        start_time: 开始时间
        end_time: 结束时间

    Returns:
        str: 生成的文件名，格式为 'data/symbol_datatype_starttime_endtime.csv'
    """
    # 提取合约代码中的具体代号（例如从'SHFE.rb2505'提取'rb2505'）
    contract = symbol.split('.')[-1]
    
    # 格式化时间字符串，移除特殊字符
    start = start_time.replace('-', '').replace(':', '').replace(' ', '')[:12]
    end = end_time.replace('-', '').replace(':', '').replace(' ', '')[:12]
    
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    
    # 生成文件名
    return f'../../data/{contract}_{data_type}_{start}_{end}.csv'

def download_bar_data(symbol: str, 
                     start_time: str, 
                     end_time: str,
                     frequency: str = '1m') -> None:
    """
    下载K线数据并进行处理

    Args:
        symbol: 合约代码，例如 'SHFE.rb2005'
        start_time: 开始时间，格式为 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
        end_time: 结束时间，格式为 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
        frequency: K线周期，例如 '1m', '5m', '1d' 等
    """
    try:
        # 生成文件名
        filename = generate_filename(symbol, frequency, start_time, end_time)
        
        data = history(
            symbol=symbol,
            frequency=frequency,
            start_time=start_time,
            end_time=end_time,
            df=True
        )

        if data.empty:
            print(f"未能获取到 {symbol} 在 {start_time} 到 {end_time} 期间的 {frequency} K线数据")
            return

        # 数据处理
        data = data.rename(columns={'bob': 'date'})  # 仅重命名时间戳列
        data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 需要删除的列（保留position和volume）
        columns_to_drop = ['eob', 'pre_close']  # 不要删除close列
        for col in columns_to_drop:
            if col in data.columns:
                data = data.drop(col, axis=1)
        
        cols = ['date'] + [col for col in data.columns if col != 'date']
        data = data[cols]

        # 保存处理后的数据
        data.to_csv(filename, index=False)
        print(f"成功下载并处理 {symbol} 的 {frequency} K线数据:")
        print(f"数据大小: {data.shape}")
        print(f"时间范围: {data['date'].iloc[0]} 到 {data['date'].iloc[-1]}")
        print(f"保存至: {filename}")
        print(f"列名顺序: {', '.join(data.columns)}")

    except Exception as e:
        print(f"下载或处理 {symbol} 的 {frequency} K线数据时发生错误: {str(e)}")
        raise

def download_tick_data(symbol: str, 
                      start_time: str, 
                      end_time: str) -> None:
    """
    下载Tick数据并进行处理

    Args:
        symbol: 合约代码，例如 'SHFE.rb2005'
        start_time: 开始时间，格式为 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
        end_time: 结束时间，格式为 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'
    """
    try:
        filename = generate_filename(symbol, 'tick', start_time, end_time)
        
        data = history(
        symbol=symbol,
            frequency='tick',
        start_time=start_time,
        end_time=end_time,
            df=True
        )

        if data.empty:
            print(f"未能获取到 {symbol} 在 {start_time} 到 {end_time} 期间的Tick数据")
            return

        # 数据处理
        # 1. 重命名列
        data = data.rename(columns={
            'created_at': 'date',
            'price': 'close'  # 添加price到close的重命名
        })
        
        # 2. 处理时间格式
        data['date'] = data['date'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # 3. 处理quotes字段
        if 'quotes' in data.columns:
            quotes_df = pd.DataFrame([q[0] if isinstance(q, list) and q else {} for q in data['quotes']])
            for col in quotes_df.columns:
                data[col] = quotes_df[col]
            data = data.drop('quotes', axis=1)

        # 4. 删除不需要的列
        columns_to_drop = ['iopv', 'trade_type', 'flag']
        for col in columns_to_drop:
            if col in data.columns:
                data = data.drop(col, axis=1)
        
        # 5. 重排列顺序，确保date在第一列
        cols = ['date'] + [col for col in data.columns if col != 'date']
        data = data[cols]

        # 保存处理后的数据
        data.to_csv(filename, index=False)
        print(f"成功下载并处理 {symbol} 的Tick数据:")
        print(f"数据大小: {data.shape}")
        print(f"时间范围: {data['date'].iloc[0]} 到 {data['date'].iloc[-1]}")
        print(f"保存至: {filename}")
        print(f"列名顺序: {', '.join(data.columns)}")

    except Exception as e:
        print(f"下载或处理 {symbol} 的Tick数据时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 下载5分钟K线数据
    download_bar_data(
        symbol='SHFE.rb2510',
        start_time='2024-10-01 09:00:00',
        end_time='2025-02-22 15:00:00',
        frequency='1d'
    )

    # 下载tick数据
    # download_tick_data(
    #     symbol='SHFE.rb2505',
    #     start_time='2025-02-11 09:00:00',
    #     end_time='2025-02-11 10:00:00'
    # )

