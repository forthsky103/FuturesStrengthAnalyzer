# src/scoring/analyses.py
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class AnalysisModule(ABC):
    @abstractmethod
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        pass

# 20 个创新 Analysis 类
class PriceTrendAccelerationAnalysis(AnalysisModule):
    """价格趋势加速度：最近20天价格变化率的趋势变化"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        chg1 = data1['close'].pct_change().tail(20)
        chg2 = data2['close'].pct_change().tail(20)
        accel1 = chg1.diff().mean() * 1000  # 二阶导数放大
        accel2 = chg2.diff().mean() * 1000
        return min(max(accel1 + 5, 0), 10), min(max(accel2 + 5, 0), 10)

class IntradayVolatilityAnalysis(AnalysisModule):
    """日内波动性：开盘-收盘与高低价差的相对强度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        intra1 = (data1['close'] - data1['open']).abs().mean() / (data1['high'] - data1['low']).mean()
        intra2 = (data2['close'] - data2['open']).abs().mean() / (data2['high'] - data2['low']).mean()
        max_intra = max(intra1, intra2, 1e-6)
        return 10 - min(intra1 / max_intra * 10, 10), 10 - min(intra2 / max_intra * 10, 10)

class PriceSkewnessAnalysis(AnalysisModule):
    """价格偏度：20天收盘价分布的偏态"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        skew1 = skew(data1['close'].tail(20))
        skew2 = skew(data2['close'].tail(20))
        return min(max(skew1 + 5, 0), 10), min(max(skew2 + 5, 0), 10)

class PriceKurtosisAnalysis(AnalysisModule):
    """价格峰度：20天收盘价分布的尖峰程度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        kurt1 = kurtosis(data1['close'].tail(20))
        kurt2 = kurtosis(data2['close'].tail(20))
        max_kurt = max(kurt1, kurt2, 1e-6)
        return 10 - min(kurt1 / max_kurt * 10, 10), 10 - min(kurt2 / max_kurt * 10, 10)

class PriceCompressionAnalysis(AnalysisModule):
    """价格压缩：20天内价格波动的收敛性"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        range1 = (data1['high'] - data1['low']).tail(20).std()
        range2 = (data2['high'] - data2['low']).tail(20).std()
        max_range = max(range1, range2, 1e-6)
        return 10 - min(range1 / max_range * 10, 10), 10 - min(range2 / max_range * 10, 10)

class IntradayReversalAnalysis(AnalysisModule):
    """日内反转强度：开盘-收盘方向与高低点位置的反转幅度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        rev1 = ((data1['close'] - data1['open']) / (data1['high'] - data1['low'])).abs().tail(10).mean() * 10
        rev2 = ((data2['close'] - data2['open']) / (data2['high'] - data2['low'])).abs().tail(10).mean() * 10
        return min(max(rev1, 0), 10), min(max(rev2, 0), 10)

class PriceMeanReversionAnalysis(AnalysisModule):
    """均值回归倾向：价格与20天均线的回归速度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        ma1 = data1['close'].rolling(window=20).mean()
        ma2 = data2['close'].rolling(window=20).mean()
        revert1 = abs(data1['close'].iloc[-1] - ma1.iloc[-1]) / abs(data1['close'].iloc[-2] - ma1.iloc[-2]) if abs(data1['close'].iloc[-2] - ma1.iloc[-2]) != 0 else 1
        revert2 = abs(data2['close'].iloc[-1] - ma2.iloc[-1]) / abs(data2['close'].iloc[-2] - ma2.iloc[-2]) if abs(data2['close'].iloc[-2] - ma2.iloc[-2]) != 0 else 1
        return min(10 / revert1, 10), min(10 / revert2, 10)

class TrendDirectionConsistencyAnalysis(AnalysisModule):
    """趋势方向一致性：20天内价格方向的连续性"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        dir1 = np.sign(data1['close'].diff()).tail(20)
        dir2 = np.sign(data2['close'].diff()).tail(20)
        cons1 = (dir1 == dir1.shift()).mean() * 10
        cons2 = (dir2 == dir2.shift()).mean() * 10
        return min(max(cons1, 0), 10), min(max(cons2, 0), 10)

class PriceBounceStrengthAnalysis(AnalysisModule):
    """价格反弹强度：从20天最低点的反弹幅度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        low1 = data1['low'].tail(20).min()
        low2 = data2['low'].tail(20).min()
        bounce1 = (data1['close'].iloc[-1] - low1) / low1 * 100
        bounce2 = (data2['close'].iloc[-1] - low2) / low2 * 100
        return min(max(bounce1, 0), 10), min(max(bounce2, 0), 10)

class PricePullbackStrengthAnalysis(AnalysisModule):
    """价格回撤强度：从20天最高点的回撤幅度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        high1 = data1['high'].tail(20).max()
        high2 = data2['high'].tail(20).max()
        pull1 = (high1 - data1['close'].iloc[-1]) / high1 * 100
        pull2 = (high2 - data2['close'].iloc[-1]) / high2 * 100
        max_pull = max(pull1, pull2, 1e-6)
        return 10 - min(pull1 / max_pull * 10, 10), 10 - min(pull2 / max_pull * 10, 10)

class IntradayPriceEfficiencyAnalysis(AnalysisModule):
    """日内价格效率：收盘价接近高点或低点的程度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        eff1 = min((data1['close'] - data1['low']) / (data1['high'] - data1['low'])).tail(10).mean() * 10
        eff2 = min((data2['close'] - data2['low']) / (data2['high'] - data2['low'])).tail(10).mean() * 10
        return min(max(eff1, 0), 10), min(max(eff2, 0), 10)

class PricePressureAnalysis(AnalysisModule):
    """价格压力：收盘价接近日内高点或低点的偏向"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        press1 = ((data1['close'] - data1['low']) - (data1['high'] - data1['close'])).tail(20).mean() / data1['close'].mean() * 1000
        press2 = ((data2['close'] - data2['low']) - (data2['high'] - data2['close'])).tail(20).mean() / data2['close'].mean() * 1000
        return min(max(press1 + 5, 0), 10), min(max(press2 + 5, 0), 10)

class VolumeWeightedVolatilityAnalysis(AnalysisModule):
    """成交量加权波动性：波动与成交量的关系"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        vol1 = ((data1['high'] - data1['low']) * data1['volume']).tail(20).mean() / data1['volume'].tail(20).mean()
        vol2 = ((data2['high'] - data2['low']) * data2['volume']).tail(20).mean() / data2['volume'].tail(20).mean()
        max_vol = max(vol1, vol2, 1e-6)
        return 10 - min(vol1 / max_vol * 10, 10), 10 - min(vol2 / max_vol * 10, 10)

class PriceMomentumDivergenceAnalysis(AnalysisModule):
    """价格动能背离：价格变化与趋势动能的差异"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        chg1 = data1['close'].pct_change().tail(20).mean()
        chg2 = data2['close'].pct_change().tail(20).mean()
        accel1 = data1['close'].diff().diff().tail(20).mean()
        accel2 = data2['close'].diff().diff().tail(20).mean()
        div1 = abs(chg1 - accel1 / data1['close'].mean() * 1000) * 10
        div2 = abs(chg2 - accel2 / data2['close'].mean() * 1000) * 10
        return 10 - min(div1, 10), 10 - min(div2, 10)

class PriceClusterAnalysis(AnalysisModule):
    """价格聚集度：价格围绕均值的分布集中度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        ma1 = data1['close'].rolling(window=20).mean()
        ma2 = data2['close'].rolling(window=20).mean()
        clust1 = (data1['close'] - ma1).abs().tail(20).mean() / data1['close'].mean() * 100
        clust2 = (data2['close'] - ma2).abs().tail(20).mean() / data2['close'].mean() * 100
        max_clust = max(clust1, clust2, 1e-6)
        return 10 - min(clust1 / max_clust * 10, 10), 10 - min(clust2 / max_clust * 10, 10)

class IntradayTrendStrengthAnalysis(AnalysisModule):
    """日内趋势强度：开盘-收盘变化的方向性占比"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        trend1 = (data1['close'] - data1['open']).tail(20).mean() / data1['close'].mean() * 1000
        trend2 = (data2['close'] - data2['open']).tail(20).mean() / data2['close'].mean() * 1000
        return min(max(trend1 + 5, 0), 10), min(max(trend2 + 5, 0), 10)

class PriceExpansionAnalysis(AnalysisModule):
    """价格扩展性：20天内价格突破高低点的频率"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        expand1 = ((data1['high'] > data1['high'].shift().rolling(window=20).max()) | 
                   (data1['low'] < data1['low'].shift().rolling(window=20).min())).tail(20).mean() * 10
        expand2 = ((data2['high'] > data2['high'].shift().rolling(window=20).max()) | 
                   (data2['low'] < data2['low'].shift().rolling(window=20).min())).tail(20).mean() * 10
        return min(max(expand1, 0), 10), min(max(expand2, 0), 10)

class MarketTensionAnalysis(AnalysisModule):
    """市场张力：价格与成交量的日内动态平衡"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        tension1 = ((data1['close'] - data1['open']).abs() / data1['volume']).tail(20).mean() * 1000
        tension2 = ((data2['close'] - data2['open']).abs() / data2['volume']).tail(20).mean() * 1000
        max_tension = max(tension1, tension2, 1e-6)
        return min(tension1 / max_tension * 10, 10), min(tension2 / max_tension * 10, 10)

class PricePathEfficiencyAnalysis(AnalysisModule):
    """价格路径效率：价格变化的直线性"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        path1 = abs(data1['close'].iloc[-1] - data1['close'].iloc[-20]) / (data1['close'].diff().abs().tail(20).sum())
        path2 = abs(data2['close'].iloc[-1] - data2['close'].iloc[-20]) / (data2['close'].diff().abs().tail(20).sum())
        return min(max(path1 * 10, 0), 10), min(max(path2 * 10, 0), 10)

class VolatilityCompressionAnalysis(AnalysisModule):
    """波动压缩：10天内波动幅度的收敛性"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        vol1 = (data1['high'] - data1['low']).tail(10).std() / (data1['high'] - data1['low']).tail(10).mean()
        vol2 = (data2['high'] - data2['low']).tail(10).std() / (data2['high'] - data2['low']).tail(10).mean()
        max_vol = max(vol1, vol2, 1e-6)
        return 10 - min(vol1 / max_vol * 10, 10), 10 - min(vol2 / max_vol * 10, 10)
    
# 新增 15 个创新 Analysis 类
class IntradaySwingTimingAnalysis(AnalysisModule):
    """日内波动时机：高低点出现的时间偏向"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def swing_timing(df):
            high_idx = (df['high'] - df['open']).abs() / (df['high'] - df['low'])  # 高点偏向开盘
            low_idx = (df['low'] - df['open']).abs() / (df['high'] - df['low'])   # 低点偏向开盘
            return (high_idx - low_idx).tail(20).mean() * 5 + 5  # 平衡性
        time1 = swing_timing(data1)
        time2 = swing_timing(data2)
        return min(max(time1, 0), 10), min(max(time2, 0), 10)

class PriceSpikeFrequencyAnalysis(AnalysisModule):
    """价格尖峰频率：日内波动超过均值2倍的次数"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def spike_freq(df):
            daily_range = (df['high'] - df['low'])
            mean_range = daily_range.mean()
            spikes = (daily_range > 2 * mean_range).tail(20).mean() * 10
            return spikes
        freq1 = spike_freq(data1)
        freq2 = spike_freq(data2)
        return min(max(freq1, 0), 10), min(max(freq2, 0), 10)

class IntradayMomentumShiftAnalysis(AnalysisModule):
    """日内动能转换：开盘-最高/最低到收盘的变化强度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def shift_strength(df):
            open_high = (df['high'] - df['open']) / df['open']
            high_close = (df['close'] - df['high']) / df['high']
            shift = (open_high - high_close).abs().tail(20).mean() * 10
            return shift
        shift1 = shift_strength(data1)
        shift2 = shift_strength(data2)
        return min(max(shift1, 0), 10), min(max(shift2, 0), 10)

class MarketEmotionVolatilityAnalysis(AnalysisModule):
    """市场情绪波动：成交量/持仓量比率的波动性"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        emo1 = (data1['volume'] / data1['position']).std() / (data1['volume'] / data1['position']).mean()
        emo2 = (data2['volume'] / data2['position']).std() / (data2['volume'] / data2['position']).mean()
        max_emo = max(emo1, emo2, 1e-6)
        return 10 - min(emo1 / max_emo * 10, 10), 10 - min(emo2 / max_emo * 10, 10)

class PriceLevelStickinessAnalysis(AnalysisModule):
    """价格水平粘性：价格重复出现的频率"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def stickiness(df):
            unique_counts = df['close'].tail(20).value_counts()
            return (unique_counts.max() / 20) * 10  # 最高重复占比
        stick1 = stickiness(data1)
        stick2 = stickiness(data2)
        return min(max(stick1, 0), 10), min(max(stick2, 0), 10)

class SupportBreakFrequencyAnalysis(AnalysisModule):
    """支撑突破频率：价格跌破20天最低点的次数"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def break_freq(df):
            low_20 = df['low'].rolling(window=20).min().shift()
            breaks = (df['low'] < low_20).tail(20).mean() * 10
            return breaks
        freq1 = break_freq(data1)
        freq2 = break_freq(data2)
        return 10 - min(freq1, 10), 10 - min(freq2, 10)  # 突破少得分高

class ResistanceBreakFrequencyAnalysis(AnalysisModule):
    """阻力突破频率：价格突破20天最高点的次数"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def break_freq(df):
            high_20 = df['high'].rolling(window=20).max().shift()
            breaks = (df['high'] > high_20).tail(20).mean() * 10
            return breaks
        freq1 = break_freq(data1)
        freq2 = break_freq(data2)
        return min(max(freq1, 0), 10), min(max(freq2, 0), 10)

class PriceLevelTransitionAnalysis(AnalysisModule):
    """价格水平过渡：价格跨越关键区间的速度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def transition_speed(df):
            range_20 = df['high'].tail(20).max() - df['low'].tail(20).min()
            speed = range_20 / df['close'].diff().abs().tail(20).mean()
            return 10 / speed * 10 if speed != 0 else 5
        speed1 = transition_speed(data1)
        speed2 = transition_speed(data2)
        return min(max(speed1, 0), 10), min(max(speed2, 0), 10)

class IntradayPriceRangeDistributionAnalysis(AnalysisModule):
    """日内价格范围分布：高低价差的偏态"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        range_skew1 = skew((data1['high'] - data1['low']).tail(20))
        range_skew2 = skew((data2['high'] - data2['low']).tail(20))
        return min(max(range_skew1 + 5, 0), 10), min(max(range_skew2 + 5, 0), 10)

class MarketParticipationBalanceAnalysis(AnalysisModule):
    """市场参与平衡：成交量/持仓量比率的稳定性"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        bal1 = (data1['volume'] / data1['position']).tail(20).std()
        bal2 = (data2['volume'] / data2['position']).tail(20).std()
        max_bal = max(bal1, bal2, 1e-6)
        return 10 - min(bal1 / max_bal * 10, 10), 10 - min(bal2 / max_bal * 10, 10)

class PriceStructureDensityAnalysis(AnalysisModule):
    """价格结构密度：价格区间内数据的集中度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def density(df):
            bins = np.histogram(df['close'].tail(20), bins=5)[0]
            return bins.max() / 20 * 10  # 最高密度占比
        dens1 = density(data1)
        dens2 = density(data2)
        return min(max(dens1, 0), 10), min(max(dens2, 0), 10)

class IntradayPriceSymmetryAnalysis(AnalysisModule):
    """日内价格对称性：开盘-收盘与高低点的对称程度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        sym1 = abs((data1['close'] - data1['open']) - (data1['high'] - data1['low'])).tail(20).mean() / data1['close'].mean() * 1000
        sym2 = abs((data2['close'] - data2['open']) - (data2['high'] - data2['low'])).tail(20).mean() / data2['close'].mean() * 1000
        max_sym = max(sym1, sym2, 1e-6)
        return 10 - min(sym1 / max_sym * 10, 10), 10 - min(sym2 / max_sym * 10, 10)

class MarketDepthImpactAnalysis(AnalysisModule):
    """市场深度影响：成交量冲击下的价格稳定性"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        impact1 = (data1['close'].diff().abs() / data1['volume']).tail(20).mean() * 1000
        impact2 = (data2['close'].diff().abs() / data2['volume']).tail(20).mean() * 1000
        max_impact = max(impact1, impact2, 1e-6)
        return 10 - min(impact1 / max_impact * 10, 10), 10 - min(impact2 / max_impact * 10, 10)

class PriceMomentumReboundAnalysis(AnalysisModule):
    """价格动能反弹：跌幅后反弹的强度"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def rebound(df):
            drops = df['close'].diff().tail(20) < 0
            rebounds = df['close'].diff().shift(-1).tail(20)[drops].mean()
            return rebounds / df['close'].mean() * 1000 + 5 if not np.isnan(rebounds) else 5
        reb1 = rebound(data1)
        reb2 = rebound(data2)
        return min(max(reb1, 0), 10), min(max(reb2, 0), 10)

class PriceVolatilityCycleAnalysis(AnalysisModule):
    """价格波动周期性：20天波动幅度的周期特征"""
    def analyze(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[float, float]:
        def cycle(df):
            ranges = (df['high'] - df['low']).tail(20)
            fft = np.abs(np.fft.fft(ranges - ranges.mean()))[1:10]  # 取低频分量
            return fft.max() / fft.mean() * 2  # 周期性强弱
        cyc1 = cycle(data1)
        cyc2 = cycle(data2)
        return min(max(cyc1, 0), 10), min(max(cyc2, 0), 10)