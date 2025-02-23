# src/features/features.py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import skew, kurtosis

class Feature(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        pass

# 现有特征（保留）
class PriceFeature(Feature):
    def __init__(self, column):
        super().__init__(f"{column}")
        self.column = column

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data[self.column]

class VolumeFeature(Feature):
    def __init__(self, window: int = 20):
        super().__init__(f"volume_sma_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["volume"].rolling(self.window).mean()

class SpreadFeature(Feature):
    def __init__(self):
        super().__init__("spread")

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["high"] - data["low"]

class PositionFeature(Feature):
    def __init__(self):
        super().__init__("position")

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["position"]

class AmountFeature(Feature):
    def __init__(self):
        super().__init__("amount")

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data["amount"]

# 新增 35 个特征
class PriceAccelerationFeature(Feature):
    """价格加速度：价格变化的二阶导数"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_acceleration_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].diff().diff().rolling(self.window).mean()

class IntradayVolatilityRatioFeature(Feature):
    """日内波动比率：开收盘价差与高低价差的比率"""
    def __init__(self, window: int = 10):
        super().__init__(f"intraday_volatility_ratio_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return ((data['close'] - data['open']).abs() / (data['high'] - data['low'])).rolling(self.window).mean()

class PriceSkewnessFeature(Feature):
    """价格偏度：收盘价的统计偏态"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_skewness_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].rolling(self.window).apply(skew, raw=False)

class PriceKurtosisFeature(Feature):
    """价格峰度：收盘价的统计峰态"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_kurtosis_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].rolling(self.window).apply(kurtosis, raw=False)

class PriceEntropyFeature(Feature):
    """价格熵：价格分布的信息熵"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_entropy_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        def entropy(series):
            counts = np.histogram(series, bins=10, density=True)[0]
            return -np.sum(counts * np.log(counts + 1e-10)) if counts.sum() > 0 else 0
        return data['close'].rolling(self.window).apply(entropy, raw=False)

class VolumePressureFeature(Feature):
    """成交量压力：成交量与价格波动的比率"""
    def __init__(self, window: int = 20):
        super().__init__(f"volume_pressure_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return (data['volume'] / (data['high'] - data['low'])).rolling(self.window).mean()

class PositionVolatilityFeature(Feature):
    """持仓量波动性：持仓量的变化标准差"""
    def __init__(self, window: int = 20):
        super().__init__(f"position_volatility_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['position'].diff().rolling(self.window).std()

class PriceMomentumFeature(Feature):
    """价格动量：价格变化的累计和"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_momentum_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].diff().rolling(self.window).sum()

class IntradayPriceRangeFeature(Feature):
    """日内价格范围扩展：高低价差与开收盘价差的差值"""
    def __init__(self, window: int = 10):
        super().__init__(f"intraday_price_range_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return ((data['high'] - data['low']) - (data['close'] - data['open']).abs()).rolling(self.window).mean()

class PriceCycleAmplitudeFeature(Feature):
    """价格周期幅度：基于傅里叶变换的低频分量强度"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_cycle_amplitude_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        def fft_amplitude(series):
            fft = np.abs(np.fft.fft(series - series.mean()))[1:self.window//2]
            return fft.max()
        return data['close'].rolling(self.window).apply(fft_amplitude, raw=True)

class PriceShadowAsymmetryFeature(Feature):
    """价格影线不对称性：上下影线的长度差异"""
    def __init__(self, window: int = 10):
        super().__init__(f"price_shadow_asymmetry_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        upper = (data['high'] - data[['open', 'close']].max(axis=1))
        lower = (data[['open', 'close']].min(axis=1) - data['low'])
        return (upper - lower).rolling(self.window).mean()

class TurnoverEfficiencyFeature(Feature):
    """换手效率：成交量与持仓量的比率波动"""
    def __init__(self, window: int = 20):
        super().__init__(f"turnover_efficiency_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return (data['volume'] / data['position']).rolling(self.window).std()

class PriceVelocityFeature(Feature):
    """价格速度：单位时间价格变化率"""
    def __init__(self, window: int = 10):
        super().__init__(f"price_velocity_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return (data['close'].diff() / data['close'].shift()).rolling(self.window).mean()

class IntradayPivotStrengthFeature(Feature):
    """日内枢轴强度：价格围绕高低点中点的偏离"""
    def __init__(self, window: int = 10):
        super().__init__(f"intraday_pivot_strength_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        pivot = (data['high'] + data['low']) / 2
        return (data['close'] - pivot).abs().rolling(self.window).mean()

class PriceFractalDimensionFeature(Feature):
    """价格分形维度：价格路径的复杂性近似"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_fractal_dimension_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        def fractal_dim(series):
            diffs = series.diff().abs()
            return len(diffs[diffs > diffs.mean()]) / self.window
        return data['close'].rolling(self.window).apply(fractal_dim, raw=False)

class AmountVelocityFeature(Feature):
    """资金流速度：成交金额变化率"""
    def __init__(self, window: int = 20):
        super().__init__(f"amount_velocity_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return (data['amount'].diff() / data['amount'].shift()).rolling(self.window).mean()

class PriceElasticityFeature(Feature):
    """价格弹性：价格变化与成交量的相对弹性"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_elasticity_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return (data['close'].pct_change() / data['volume'].pct_change()).rolling(self.window).mean()

class VolatilityCycleFeature(Feature):
    """波动周期性：高低价差的周期强度"""
    def __init__(self, window: int = 20):
        super().__init__(f"volatility_cycle_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        def fft_cycle(series):
            fft = np.abs(np.fft.fft(series - series.mean()))[1:self.window//2]
            return fft.max() / fft.mean()
        return (data['high'] - data['low']).rolling(self.window).apply(fft_cycle, raw=True)

class PriceMeanDistanceFeature(Feature):
    """价格均值距离：价格与均线的平均偏离"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_mean_distance_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ma = data['close'].rolling(self.window).mean()
        return (data['close'] - ma).abs().rolling(self.window).mean()

class IntradayPriceSymmetryFeature(Feature):
    """日内价格对称性：开收盘与高低点的对称程度"""
    def __init__(self, window: int = 10):
        super().__init__(f"intraday_price_symmetry_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return abs((data['close'] - data['open']) - (data['high'] - data['low'])).rolling(self.window).mean()

class VolumeMomentumFeature(Feature):
    """成交量动量：成交量变化的累计和"""
    def __init__(self, window: int = 20):
        super().__init__(f"volume_momentum_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['volume'].diff().rolling(self.window).sum()

class PriceWaveletEnergyFeature(Feature):
    """价格小波能量：价格波动的小波变换能量"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_wavelet_energy_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        def wavelet_energy(series):
            coeffs = np.diff(series)  # 简单近似小波系数
            return np.sum(coeffs**2)
        return data['close'].rolling(self.window).apply(wavelet_energy, raw=True)

class PositionAccelerationFeature(Feature):
    """持仓量加速度：持仓量变化的二阶导数"""
    def __init__(self, window: int = 20):
        super().__init__(f"position_acceleration_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['position'].diff().diff().rolling(self.window).mean()

class PriceDensityFeature(Feature):
    """价格密度：价格分布的集中度"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_density_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        def density(series):
            bins = np.histogram(series, bins=10)[0]
            return bins.max() / self.window
        return data['close'].rolling(self.window).apply(density, raw=False)

class IntradayPriceVelocityFeature(Feature):
    """日内价格速度：开收盘变化的速度"""
    def __init__(self, window: int = 10):
        super().__init__(f"intraday_price_velocity_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return ((data['close'] - data['open']) / data['open']).rolling(self.window).mean()

class PriceRotationFrequencyFeature(Feature):
    """价格旋转频率：价格围绕均值的旋转次数"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_rotation_frequency_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ma = data['close'].rolling(self.window).mean()
        return ((data['close'] > ma) != (data['close'].shift() > ma.shift())).rolling(self.window).mean()

class VolumePriceCorrelationFeature(Feature):
    """成交量-价格相关性：成交量与价格的相关系数"""
    def __init__(self, window: int = 20):
        super().__init__(f"volume_price_correlation_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['volume'].rolling(self.window).corr(data['close'])

class PriceBreakoutStrengthFeature(Feature):
    """突破强度：价格突破均线的幅度"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_breakout_strength_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        ma = data['close'].rolling(self.window).mean()
        return (data['close'] - ma) / ma

class IntradayPriceCenterFeature(Feature):
    """日内价格中心：收盘价与高低点中点的距离"""
    def __init__(self, window: int = 10):
        super().__init__(f"intraday_price_center_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        center = (data['high'] + data['low']) / 2
        return (data['close'] - center).rolling(self.window).mean()

class PriceHarmonicAmplitudeFeature(Feature):
    """价格谐波幅度：价格波动的谐波分量强度"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_harmonic_amplitude_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        def harmonic(series):
            fft = np.abs(np.fft.fft(series - series.mean()))[1:self.window//2]
            return fft[0]  # 第一谐波分量
        return data['close'].rolling(self.window).apply(harmonic, raw=True)

class AmountPressureFeature(Feature):
    """资金流压力：成交金额与价格波动的比率"""
    def __init__(self, window: int = 20):
        super().__init__(f"amount_pressure_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return (data['amount'] / (data['high'] - data['low'])).rolling(self.window).mean()

class PositionMomentumFeature(Feature):
    """持仓量动量：持仓量变化的累计和"""
    def __init__(self, window: int = 20):
        super().__init__(f"position_momentum_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return data['position'].diff().rolling(self.window).sum()

class PriceSpikeFrequencyFeature(Feature):
    """价格尖峰频率：超出均值2倍波动的次数"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_spike_frequency_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        range_mean = (data['high'] - data['low']).rolling(self.window).mean()
        return ((data['high'] - data['low']) > 2 * range_mean).rolling(self.window).mean()

class IntradayPriceElasticityFeature(Feature):
    """日内价格弹性：开收盘变化与成交量的相对弹性"""
    def __init__(self, window: int = 10):
        super().__init__(f"intraday_price_elasticity_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return ((data['close'] - data['open']).pct_change() / data['volume'].pct_change()).rolling(self.window).mean()

class PriceTrendPersistenceFeature(Feature):
    """趋势持久性：价格方向一致性比例"""
    def __init__(self, window: int = 20):
        super().__init__(f"price_trend_persistence_{window}")
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.Series:
        direction = np.sign(data['close'].diff())
        return (direction == direction.shift()).rolling(self.window).mean()