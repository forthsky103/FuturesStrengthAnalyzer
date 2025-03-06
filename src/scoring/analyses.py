# src/scoring/analyses.py
from abc import ABC, abstractmethod
from typing import List, Dict
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

class AnalysisModule(ABC):
    @abstractmethod
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        """分析多个合约数据，返回每个合约的得分"""
        pass

class PriceTrendAccelerationAnalysis(AnalysisModule):
    """价格趋势加速度：最近20天价格变化率的趋势变化"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            chg = data['close'].pct_change().tail(20)
            accel = chg.diff().mean() * 1000
            scores[f"contract{i+1}"] = min(max(accel + 5, 0), 10)
        return scores

class IntradayVolatilityAnalysis(AnalysisModule):
    """日内波动性：开盘-收盘与高低价差的相对强度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_intra = max([((data['close'] - data['open']).abs().mean() / (data['high'] - data['low']).mean()) for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            intra = (data['close'] - data['open']).abs().mean() / (data['high'] - data['low']).mean()
            scores[f"contract{i+1}"] = 10 - min(intra / max_intra * 10, 10)
        return scores

class PriceSkewnessAnalysis(AnalysisModule):
    """价格偏度：20天收盘价分布的偏态"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            skew_val = skew(data['close'].tail(20))
            scores[f"contract{i+1}"] = min(max(skew_val + 5, 0), 10)
        return scores

class PriceKurtosisAnalysis(AnalysisModule):
    """价格峰度：20天收盘价分布的尖峰程度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_kurt = max([kurtosis(data['close'].tail(20)) for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            kurt = kurtosis(data['close'].tail(20))
            scores[f"contract{i+1}"] = 10 - min(kurt / max_kurt * 10, 10)
        return scores

class PriceCompressionAnalysis(AnalysisModule):
    """价格压缩：20天内价格波动的收敛性"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_range = max([(data['high'] - data['low']).tail(20).std() for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            range_val = (data['high'] - data['low']).tail(20).std()
            scores[f"contract{i+1}"] = 10 - min(range_val / max_range * 10, 10)
        return scores

class IntradayReversalAnalysis(AnalysisModule):
    """日内反转强度：开盘-收盘方向与高低点位置的反转幅度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            rev = ((data['close'] - data['open']) / (data['high'] - data['low'])).abs().tail(10).mean() * 10
            scores[f"contract{i+1}"] = min(max(rev, 0), 10)
        return scores

class PriceMeanReversionAnalysis(AnalysisModule):
    """均值回归倾向：价格与20天均线的回归速度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            ma = data['close'].rolling(window=20).mean()
            revert = abs(data['close'].iloc[-1] - ma.iloc[-1]) / abs(data['close'].iloc[-2] - ma.iloc[-2]) if abs(data['close'].iloc[-2] - ma.iloc[-2]) != 0 else 1
            scores[f"contract{i+1}"] = min(10 / revert, 10)
        return scores

class TrendDirectionConsistencyAnalysis(AnalysisModule):
    """趋势方向一致性：20天内价格方向的连续性"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            dir_val = np.sign(data['close'].diff()).tail(20)
            cons = (dir_val == dir_val.shift()).mean() * 10
            scores[f"contract{i+1}"] = min(max(cons, 0), 10)
        return scores

class PriceBounceStrengthAnalysis(AnalysisModule):
    """价格反弹强度：从20天最低点的反弹幅度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            low = data['low'].tail(20).min()
            bounce = (data['close'].iloc[-1] - low) / low * 100
            scores[f"contract{i+1}"] = min(max(bounce, 0), 10)
        return scores

class PricePullbackStrengthAnalysis(AnalysisModule):
    """价格回撤强度：从20天最高点的回撤幅度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_pull = max([(data['high'].tail(20).max() - data['close'].iloc[-1]) / data['high'].tail(20).max() * 100 for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            high = data['high'].tail(20).max()
            pull = (high - data['close'].iloc[-1]) / high * 100
            scores[f"contract{i+1}"] = 10 - min(pull / max_pull * 10, 10)
        return scores

class IntradayPriceEfficiencyAnalysis(AnalysisModule):
    """日内价格效率：收盘价接近高点或低点的程度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            eff_series = (data['close'] - data['low']) / (data['high'] - data['low']).tail(10)
            eff = eff_series.mean() * 10
            scores[f"contract{i+1}"] = min(max(eff, 0), 10)
        return scores

class PricePressureAnalysis(AnalysisModule):
    """价格压力：收盘价接近日内高点或低点的偏向"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            press = ((data['close'] - data['low']) - (data['high'] - data['close'])).tail(20).mean() / data['close'].mean() * 1000
            scores[f"contract{i+1}"] = min(max(press + 5, 0), 10)
        return scores

class VolumeWeightedVolatilityAnalysis(AnalysisModule):
    """成交量加权波动性：波动与成交量的关系"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_vol = max([((data['high'] - data['low']) * data['volume']).tail(20).mean() / data['volume'].tail(20).mean() for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            vol = ((data['high'] - data['low']) * data['volume']).tail(20).mean() / data['volume'].tail(20).mean()
            scores[f"contract{i+1}"] = 10 - min(vol / max_vol * 10, 10)
        return scores

class PriceMomentumDivergenceAnalysis(AnalysisModule):
    """价格动能背离：价格变化与趋势动能的差异"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            chg = data['close'].pct_change().tail(20).mean()
            accel = data['close'].diff().diff().tail(20).mean()
            div = abs(chg - accel / data['close'].mean() * 1000) * 10
            scores[f"contract{i+1}"] = 10 - min(div, 10)
        return scores

class PriceClusterAnalysis(AnalysisModule):
    """价格聚集度：价格围绕均值的分布集中度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_clust = max([(data['close'] - data['close'].rolling(window=20).mean()).abs().tail(20).mean() / data['close'].mean() * 100 for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            ma = data['close'].rolling(window=20).mean()
            clust = (data['close'] - ma).abs().tail(20).mean() / data['close'].mean() * 100
            scores[f"contract{i+1}"] = 10 - min(clust / max_clust * 10, 10)
        return scores

class IntradayTrendStrengthAnalysis(AnalysisModule):
    """日内趋势强度：开盘-收盘变化的方向性占比"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            trend = (data['close'] - data['open']).tail(20).mean() / data['close'].mean() * 1000
            scores[f"contract{i+1}"] = min(max(trend + 5, 0), 10)
        return scores

class PriceExpansionAnalysis(AnalysisModule):
    """价格扩展性：20天内价格突破高低点的频率"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            expand = ((data['high'] > data['high'].shift().rolling(window=20).max()) | 
                      (data['low'] < data['low'].shift().rolling(window=20).min())).tail(20).mean() * 10
            scores[f"contract{i+1}"] = min(max(expand, 0), 10)
        return scores

class MarketTensionAnalysis(AnalysisModule):
    """市场张力：价格与成交量的日内动态平衡"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_tension = max([((data['close'] - data['open']).abs() / data['volume']).tail(20).mean() * 1000 for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            tension = ((data['close'] - data['open']).abs() / data['volume']).tail(20).mean() * 1000
            scores[f"contract{i+1}"] = min(tension / max_tension * 10, 10)
        return scores

class PricePathEfficiencyAnalysis(AnalysisModule):
    """价格路径效率：价格变化的直线性"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            path = abs(data['close'].iloc[-1] - data['close'].iloc[-20]) / (data['close'].diff().abs().tail(20).sum()) if data['close'].diff().abs().tail(20).sum() != 0 else 0
            scores[f"contract{i+1}"] = min(max(path * 10, 0), 10)
        return scores

class VolatilityCompressionAnalysis(AnalysisModule):
    """波动压缩：10天内波动幅度的收敛性"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_vol = max([(data['high'] - data['low']).tail(10).std() / (data['high'] - data['low']).tail(10).mean() for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            vol = (data['high'] - data['low']).tail(10).std() / (data['high'] - data['low']).tail(10).mean()
            scores[f"contract{i+1}"] = 10 - min(vol / max_vol * 10, 10)
        return scores

class IntradaySwingTimingAnalysis(AnalysisModule):
    """日内波动时机：高低点出现的时间偏向"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            high_idx = (data['high'] - data['open']).abs() / (data['high'] - data['low'])
            low_idx = (data['low'] - data['open']).abs() / (data['high'] - data['low'])
            time_val = (high_idx - low_idx).tail(20).mean() * 5 + 5
            scores[f"contract{i+1}"] = min(max(time_val, 0), 10)
        return scores

class PriceSpikeFrequencyAnalysis(AnalysisModule):
    """价格尖峰频率：日内波动超过均值2倍的次数"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            daily_range = (data['high'] - data['low'])
            mean_range = daily_range.mean()
            spikes = (daily_range > 2 * mean_range).tail(20).mean() * 10
            scores[f"contract{i+1}"] = min(max(spikes, 0), 10)
        return scores

class IntradayMomentumShiftAnalysis(AnalysisModule):
    """日内动能转换：开盘-最高/最低到收盘的变化强度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            open_high = (data['high'] - data['open']) / data['open']
            high_close = (data['close'] - data['high']) / data['high']
            shift = (open_high - high_close).abs().tail(20).mean() * 10
            scores[f"contract{i+1}"] = min(max(shift, 0), 10)
        return scores

class MarketEmotionVolatilityAnalysis(AnalysisModule):
    """市场情绪波动：成交量/持仓量比率的波动性"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_emo = max([(data['volume'] / data['position']).std() / (data['volume'] / data['position']).mean() for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            emo = (data['volume'] / data['position']).std() / (data['volume'] / data['position']).mean()
            scores[f"contract{i+1}"] = 10 - min(emo / max_emo * 10, 10)
        return scores

class PriceLevelStickinessAnalysis(AnalysisModule):
    """价格水平粘性：价格重复出现的频率"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            unique_counts = data['close'].tail(20).value_counts()
            stick = (unique_counts.max() / 20) * 10
            scores[f"contract{i+1}"] = min(max(stick, 0), 10)
        return scores

class SupportBreakFrequencyAnalysis(AnalysisModule):
    """支撑突破频率：价格跌破20天最低点的次数"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            low_20 = data['low'].rolling(window=20).min().shift()
            breaks = (data['low'] < low_20).tail(20).mean() * 10
            scores[f"contract{i+1}"] = 10 - min(breaks, 10)
        return scores

class ResistanceBreakFrequencyAnalysis(AnalysisModule):
    """阻力突破频率：价格突破20天最高点的次数"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            high_20 = data['high'].rolling(window=20).max().shift()
            breaks = (data['high'] > high_20).tail(20).mean() * 10
            scores[f"contract{i+1}"] = min(max(breaks, 0), 10)
        return scores

class PriceLevelTransitionAnalysis(AnalysisModule):
    """价格水平过渡：价格跨越关键区间的速度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            range_20 = data['high'].tail(20).max() - data['low'].tail(20).min()
            speed = range_20 / data['close'].diff().abs().tail(20).mean() if data['close'].diff().abs().tail(20).mean() != 0 else 1
            scores[f"contract{i+1}"] = min(max(10 / speed * 10, 0), 10)
        return scores

class IntradayPriceRangeDistributionAnalysis(AnalysisModule):
    """日内价格范围分布：高低价差的偏态"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            range_skew = skew((data['high'] - data['low']).tail(20))
            scores[f"contract{i+1}"] = min(max(range_skew + 5, 0), 10)
        return scores

class MarketParticipationBalanceAnalysis(AnalysisModule):
    """市场参与平衡：成交量/持仓量比率的稳定性"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_bal = max([(data['volume'] / data['position']).tail(20).std() for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            bal = (data['volume'] / data['position']).tail(20).std()
            scores[f"contract{i+1}"] = 10 - min(bal / max_bal * 10, 10)
        return scores

class PriceStructureDensityAnalysis(AnalysisModule):
    """价格结构密度：价格区间内数据的集中度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            bins = np.histogram(data['close'].tail(20), bins=5)[0]
            dens = bins.max() / 20 * 10
            scores[f"contract{i+1}"] = min(max(dens, 0), 10)
        return scores

class IntradayPriceSymmetryAnalysis(AnalysisModule):
    """日内价格对称性：开盘-收盘与高低点的对称程度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_sym = max([abs((data['close'] - data['open']) - (data['high'] - data['low'])).tail(20).mean() / data['close'].mean() * 1000 for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            sym = abs((data['close'] - data['open']) - (data['high'] - data['low'])).tail(20).mean() / data['close'].mean() * 1000
            scores[f"contract{i+1}"] = 10 - min(sym / max_sym * 10, 10)
        return scores

class MarketDepthImpactAnalysis(AnalysisModule):
    """市场深度影响：成交量冲击下的价格稳定性"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        max_impact = max([(data['close'].diff().abs() / data['volume']).tail(20).mean() * 1000 for data in datasets] + [1e-6])
        for i, data in enumerate(datasets):
            impact = (data['close'].diff().abs() / data['volume']).tail(20).mean() * 1000
            scores[f"contract{i+1}"] = 10 - min(impact / max_impact * 10, 10)
        return scores

class PriceMomentumReboundAnalysis(AnalysisModule):
    """价格动能反弹：跌幅后反弹的强度"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            drops = data['close'].diff().tail(20) < 0
            rebounds = data['close'].diff().shift(-1).tail(20)[drops].mean()
            reb = rebounds / data['close'].mean() * 1000 + 5 if not np.isnan(rebounds) else 5
            scores[f"contract{i+1}"] = min(max(reb, 0), 10)
        return scores

class PriceVolatilityCycleAnalysis(AnalysisModule):
    """价格波动周期性：20天波动幅度的周期特征"""
    def analyze(self, datasets: List[pd.DataFrame]) -> Dict[str, float]:
        scores = {}
        for i, data in enumerate(datasets):
            ranges = (data['high'] - data['low']).tail(20)
            fft = np.abs(np.fft.fft(ranges - ranges.mean()))[1:10]
            cyc = fft.max() / fft.mean() * 2 if fft.mean() != 0 else 5
            scores[f"contract{i+1}"] = min(max(cyc, 0), 10)
        return scores