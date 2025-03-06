# src/rules/evaluator.py
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import os  # 添加 os 导入
import pandas as pd
import numpy as np
import yaml
import logging
import ta  # 引入技术分析库
from ..utils.market_conditions import MarketCondition

class Rule(ABC):
    """规则基类，每个具体规则需实现 evaluate 方法"""
    @abstractmethod
    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        """评估规则，返回是否满足、置信度和解释
        Args:
            features (pd.DataFrame): 单个合约的原始数据，至少包含基础列（如 close, high, low）
        Returns:
            Tuple[bool, float, str]: (是否满足, 置信度, 解释)
        """
        pass

class BreakoutMARule(Rule):
    """规则：价格突破均线，阈值随波动率动态调整"""
    def __init__(self, window: int = 20, volatility_factor: float = 0.01):
        self.window = window
        self.volatility_factor = volatility_factor

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        recent = features.tail(self.window)
        ma = recent['close'].mean()
        latest_close = recent['close'].iloc[-1]
        volatility = features['market_volatility'].iloc[-1] if 'market_volatility' in features else 0.01
        threshold = ma * (1 + self.volatility_factor * volatility)
        is_met = latest_close > threshold
        confidence = min((latest_close - threshold) / threshold * 10, 1.0) if is_met else 0.0
        explanation = f"价格 {latest_close:.2f} {'>' if is_met else '<='} 均线+阈值 {threshold:.2f}"
        return is_met, confidence, explanation

class VolumeIncreaseRule(Rule):
    """规则：成交量增加"""
    def __init__(self, window: int = 20):
        self.window = window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        recent = features.tail(self.window)
        vol_change = (recent['volume'].iloc[-1] - recent['volume'].iloc[0]) / recent['volume'].iloc[0]
        is_met = vol_change > 0
        confidence = min(vol_change * 10, 1.0) if is_met else 0.0
        explanation = f"成交量变化率 {vol_change:.2%}"
        return is_met, confidence, explanation

class PositionTrendRule(Rule):
    """规则：持仓量趋势上升"""
    def __init__(self, window: int = 20):
        self.window = window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        recent = features.tail(self.window)
        pos_change = (recent['position'].iloc[-1] - recent['position'].iloc[0]) / recent['position'].iloc[0]
        is_met = pos_change > 0
        confidence = min(pos_change * 10, 1.0) if is_met else 0.0
        explanation = f"持仓量变化率 {pos_change:.2%}"
        return is_met, confidence, explanation

class VolatilityExpansionRule(Rule):
    """规则：波动率扩展（ATR增加且价格上涨）"""
    def __init__(self, window: int = 20):
        self.window = window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        recent = features.tail(self.window)
        atr = (recent['high'] - recent['low']).mean()
        prev_atr = features.iloc[-self.window-20:-self.window]['high'].mean() - features.iloc[-self.window-20:-self.window]['low'].mean()
        price_up = recent['close'].iloc[-1] > recent['close'].iloc[0]
        is_met = atr > prev_atr and price_up
        confidence = min((atr - prev_atr) / prev_atr * 10, 1.0) if is_met else 0.0
        explanation = f"ATR {atr:.2f} {'>' if is_met else '<='} 前期 {prev_atr:.2f}, 价格{'上涨' if price_up else '未上涨'}"
        return is_met, confidence, explanation

class UptrendRule(Rule):
    """规则：上升趋势（5日均线 > 20日均线 > 50日均线）"""
    def __init__(self, short_window: int = 5, mid_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.mid_window = mid_window
        self.long_window = long_window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        short_ma = features['close'].tail(self.short_window).mean()
        mid_ma = features['close'].tail(self.mid_window).mean()
        long_ma = features['close'].tail(self.long_window).mean()
        is_met = short_ma > mid_ma > long_ma
        confidence = min((short_ma - mid_ma) / mid_ma * 10, 1.0) if is_met else 0.0
        explanation = f"5日均线 {short_ma:.2f}, 20日均线 {mid_ma:.2f}, 50日均线 {long_ma:.2f}"
        return is_met, confidence, explanation

class RSIAbove50Rule(Rule):
    """规则：RSI > 50（动态阈值）"""
    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        # 计算 RSI
        rsi_series = ta.momentum.RSIIndicator(features['close'], window=14).rsi()
        rsi = rsi_series.iloc[-1]
        volatility = features['market_volatility'].iloc[-1] if 'market_volatility' in features else 0.01
        threshold = 50 - 10 * volatility
        is_met = rsi > threshold
        confidence = min((rsi - threshold) / (100 - threshold), 1.0) if is_met else 0.0
        explanation = f"RSI: {rsi:.2f} {'>' if is_met else '<='} 动态阈值 {threshold:.2f}"
        return is_met, confidence, explanation

class MACDPositiveAndAboveSignalRule(Rule):
    """规则：MACD > 0 且 MACD > 信号线"""
    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        # 计算 MACD
        macd = ta.trend.MACD(features['close'])
        macd_val = macd.macd().iloc[-1]
        signal = macd.macd_signal().iloc[-1]
        is_met = macd_val > 0 and macd_val > signal
        confidence = min((macd_val - signal) / abs(signal) * 10, 1.0) if is_met else 0.0
        explanation = f"MACD: {macd_val:.2f}, 信号线: {signal:.2f}"
        return is_met, confidence, explanation

class PriceAboveBollingerUpperRule(Rule):
    """规则：价格 > 上轨布林带"""
    def __init__(self, window: int = 20):
        self.window = window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        # 计算布林带上轨
        bb = ta.volatility.BollingerBands(features['close'], window=self.window)
        upper_band = bb.bollinger_hband().iloc[-1]
        recent = features.tail(self.window)
        latest_close = recent['close'].iloc[-1]
        is_met = latest_close > upper_band
        confidence = min((latest_close - upper_band) / upper_band * 10, 1.0) if is_met else 0.0
        explanation = f"价格 {latest_close:.2f} {'>' if is_met else '<='} 上轨 {upper_band:.2f}"
        return is_met, confidence, explanation

class TrendReversalRule(Rule):
    """规则：价格跌破20日均线且RSI<30"""
    def __init__(self, window: int = 20):
        self.window = window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        # 计算 RSI
        rsi_series = ta.momentum.RSIIndicator(features['close'], window=14).rsi()
        rsi = rsi_series.iloc[-1]
        recent = features.tail(self.window)
        ma = recent['close'].mean()
        latest_close = recent['close'].iloc[-1]
        is_met = latest_close < ma and rsi < 30
        confidence = min((ma - latest_close) / ma * 10 + (30 - rsi) / 30, 1.0) if is_met else 0.0
        explanation = f"价格 {latest_close:.2f} {'<' if is_met else '>='} 均线 {ma:.2f}, RSI: {rsi:.2f} {'<' if rsi < 30 else '>='} 30"
        return is_met, confidence, explanation

class VolumeSpikeRule(Rule):
    """规则：成交量超过20日均量的2倍"""
    def __init__(self, window: int = 20, threshold: float = 2.0):
        self.window = window
        self.threshold = threshold

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        recent = features.tail(self.window)
        avg_volume = recent['volume'].mean()
        latest_volume = recent['volume'].iloc[-1]
        is_met = latest_volume > avg_volume * self.threshold
        confidence = min((latest_volume - avg_volume * self.threshold) / (avg_volume * self.threshold) * 10, 1.0) if is_met else 0.0
        explanation = f"成交量 {latest_volume:.0f} {'>' if is_met else '<='} {self.threshold}x均量 {avg_volume * self.threshold:.0f}"
        return is_met, confidence, explanation

class PriceBreakoutHighRule(Rule):
    """规则：价格突破前20日最高点"""
    def __init__(self, window: int = 20):
        self.window = window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        recent = features.tail(self.window + 1)
        prev_high = recent['high'].iloc[:-1].max()
        latest_close = recent['close'].iloc[-1]
        is_met = latest_close > prev_high
        confidence = min((latest_close - prev_high) / prev_high * 10, 1.0) if is_met else 0.0
        explanation = f"价格 {latest_close:.2f} {'>' if is_met else '<='} 前高 {prev_high:.2f}"
        return is_met, confidence, explanation

class MACDGoldenCrossRule(Rule):
    """规则：MACD上穿信号线"""
    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        # 计算 MACD
        macd = ta.trend.MACD(features['close'])
        macd_val = macd.macd().iloc[-1]
        signal = macd.macd_signal().iloc[-1]
        prev_macd = macd.macd().iloc[-2]
        prev_signal = macd.macd_signal().iloc[-2]
        is_met = macd_val > signal and prev_macd <= prev_signal
        confidence = min((macd_val - signal) / abs(signal) * 10, 1.0) if is_met else 0.0
        explanation = f"MACD {macd_val:.2f} 上穿信号线 {signal:.2f} (前值: {prev_macd:.2f}, {prev_signal:.2f})"
        return is_met, confidence, explanation

class VolatilityContractionRule(Rule):
    """规则：波动率（ATR）连续5天缩小"""
    def __init__(self, window: int = 5):
        self.window = window

    def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
        recent = features.tail(self.window + 1)
        atr = (recent['high'] - recent['low']).diff()
        is_met = all(atr.iloc[-i] < atr.iloc[-i-1] for i in range(1, self.window + 1))
        confidence = 1.0 if is_met else 0.0
        explanation = f"ATR最近{self.window}天{'连续缩小' if is_met else '未连续缩小'}"
        return is_met, confidence, explanation

class ExpertSystem:
    """专家系统，整合规则、市场状态和推理引擎"""
    def __init__(self, rules: List[Rule], rule_weights: Dict[str, float] = None):
        """初始化专家系统
        Args:
            rules (List[Rule]): 规则列表
            rule_weights (Dict[str, float]): 初始权重字典
        """
        self.rules = rules
        self.base_weights = rule_weights or {rule.__class__.__name__: 1.0 for rule in rules}
        self.rule_weights = self.base_weights.copy()
        self.market_volatility = None
        self.knowledge_base = {}

    def adjust_weights(self, datasets: List[pd.DataFrame], conditions: List[MarketCondition]) -> Dict[str, float]:
        """动态调整规则权重"""
        self.rule_weights = self.base_weights.copy()
        for condition in conditions:
            if condition.evaluate(datasets):
                condition.apply_adjustments(self.rule_weights)
                logging.info(f"应用市场状态: {condition.__class__.__name__}")
        logging.info(f"动态调整权重: {self.rule_weights}")
        return self.rule_weights

    def extract_facts(self, features: pd.DataFrame, contract: str) -> Dict[str, Tuple[bool, float, str]]:
        """从特征数据中提取事实，存入知识库"""
        if 'market_volatility' not in features.columns:
            features['market_volatility'] = self.market_volatility
        facts = {}
        for rule in self.rules:
            rule_name = rule.__class__.__name__
            is_met, confidence, explanation = rule.evaluate(features)
            facts[rule_name] = (is_met, confidence, explanation)
            logging.debug(f"{contract} - {rule_name}: {explanation}, 置信度: {confidence:.2f}")
        self.knowledge_base[contract] = facts
        return facts

    def evaluate_dependencies(self, rule_name: str, facts: Dict, dependencies: Dict) -> Tuple[bool, float]:
        """递归评估规则的依赖关系，返回是否满足及增强倍数"""
        if rule_name not in dependencies:
            return True, 1.0
        dep_config = dependencies[rule_name]
        requires = dep_config.get("requires", [])
        logic = dep_config.get("logic", "and")
        boost = dep_config.get("boost", 1.0)

        dep_results = []
        for req in requires:
            if isinstance(req, str):
                is_met, _, _ = facts.get(req, (False, 0.0, ""))
                dep_results.append(is_met)
            else:
                sub_rule = req["rule"]
                sub_met, sub_boost = self.evaluate_dependencies(sub_rule, facts, dependencies)
                dep_results.append(sub_met)
                boost *= sub_boost if sub_met else 1.0

        if logic == "and":
            condition_met = all(dep_results)
        elif logic == "or":
            condition_met = any(dep_results)
        elif logic == "not":
            condition_met = not all(dep_results)
        else:
            condition_met = True
        return condition_met, boost if condition_met else 1.0

    def infer_strength(self, facts: Dict[str, Tuple[bool, float, str]], dependencies: Dict[str, Dict] = None) -> Tuple[float, str]:
        """推理强弱得分，支持独立和依赖模式"""
        strength_score = 0.0
        total_weight = sum(self.rule_weights.values())
        explanation = "推理过程：\n"
        dependencies = dependencies or {}

        for rule_name, (is_met, confidence, rule_explanation) in facts.items():
            weight = self.rule_weights.get(rule_name, 1.0)
            dep_met, boost = self.evaluate_dependencies(rule_name, facts, dependencies)
            dep_explanation = f" (依赖满足，增益 {boost:.2f}x)" if boost != 1.0 and is_met and dep_met else ""

            if is_met and dep_met:
                adjusted_contribution = confidence * weight * boost
            else:
                adjusted_contribution = 0.0

            explanation += f"- {rule_name}: {rule_explanation}, 权重: {weight}, 贡献: {adjusted_contribution:.2f}{dep_explanation}\n"
            strength_score += adjusted_contribution

        final_score = strength_score / total_weight if total_weight > 0 else 0.0
        explanation += f"最终得分: {final_score:.2f}"
        return final_score, explanation

    def evaluate(self, feature_datasets: List[pd.DataFrame], condition_map: Dict[str, type], config_path: str = "rules_config.yaml") -> Dict[str, Tuple[float, str]]:
        """评估所有合约的强弱"""
        config_full_path = os.path.join(os.path.dirname(__file__), config_path)
        with open(config_full_path, "r", encoding='utf-8') as f:
            rules_config = yaml.safe_load(f)
        conditions = [condition_map[cond["type"]](cond["adjustments"]) for cond in rules_config.get("market_conditions", [])]
        dependencies = rules_config.get("dependencies", {})
        
        if rules_config.get("auto_weights", False):
            from ..weight_generator.generate_weights import WeightGenerator
            generator = WeightGenerator()
            self.base_weights = generator.generate("rules", feature_datasets, self.rules)
            self.rule_weights = self.base_weights.copy()
            generator.update_config("rules", self.base_weights, config_full_path)
        else:
            self.base_weights = rules_config.get("weights", {rule.__class__.__name__: 1.0 for rule in self.rules})
            self.rule_weights = self.base_weights.copy()

        self.rule_weights = self.adjust_weights(feature_datasets, conditions)
        market_atr = np.mean([df['high'].tail(20).mean() - df['low'].tail(20).mean() for df in feature_datasets])
        self.market_volatility = market_atr / feature_datasets[0]['close'].mean()
        results = {}
        for i, features in enumerate(feature_datasets):
            contract = f"contract{i+1}"
            facts = self.extract_facts(features, contract)
            score, explanation = self.infer_strength(facts, dependencies)
            results[contract] = (score, explanation)
            logging.info(f"{contract} 强弱评估完成，得分: {score:.2f}\n{explanation}")
        return results


# # src/rules/evaluator.py
# from abc import ABC, abstractmethod
# from typing import List, Dict, Tuple
# import pandas as pd
# import numpy as np
# import yaml
# import logging
# from ..utils.market_conditions import MarketCondition

# class Rule(ABC):
#     """规则基类，每个具体规则需实现 evaluate 方法"""
#     @abstractmethod
#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         """评估规则，返回是否满足、置信度和解释。
#         Args:
#             features (pd.DataFrame): 单个合约的特征数据
#         Returns:
#             Tuple[bool, float, str]: (是否满足, 置信度, 解释)
#         """
#         pass

# class BreakoutMARule(Rule):
#     """规则：价格突破均线，阈值随波动率动态调整"""
#     def __init__(self, window: int = 20, volatility_factor: float = 0.01):
#         self.window = window
#         self.volatility_factor = volatility_factor

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window)
#         ma = recent['close'].mean()
#         latest_close = recent['close'].iloc[-1]
#         volatility = features['market_volatility'].iloc[-1] if 'market_volatility' in features else 0.01
#         threshold = ma * (1 + self.volatility_factor * volatility)
#         is_met = latest_close > threshold
#         confidence = min((latest_close - threshold) / threshold * 10, 1.0) if is_met else 0.0
#         explanation = f"价格 {latest_close:.2f} {'>' if is_met else '<='} 均线+阈值 {threshold:.2f}"
#         return is_met, confidence, explanation

# class VolumeIncreaseRule(Rule):
#     """规则：成交量增加"""
#     def __init__(self, window: int = 20):
#         self.window = window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window)
#         vol_change = (recent['volume'].iloc[-1] - recent['volume'].iloc[0]) / recent['volume'].iloc[0]
#         is_met = vol_change > 0
#         confidence = min(vol_change * 10, 1.0) if is_met else 0.0
#         explanation = f"成交量变化率 {vol_change:.2%}"
#         return is_met, confidence, explanation

# class PositionTrendRule(Rule):
#     """规则：持仓量趋势上升"""
#     def __init__(self, window: int = 20):
#         self.window = window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window)
#         pos_change = (recent['position'].iloc[-1] - recent['position'].iloc[0]) / recent['position'].iloc[0]
#         is_met = pos_change > 0
#         confidence = min(pos_change * 10, 1.0) if is_met else 0.0
#         explanation = f"持仓量变化率 {pos_change:.2%}"
#         return is_met, confidence, explanation

# class VolatilityExpansionRule(Rule):
#     """规则：波动率扩展（ATR增加且价格上涨）"""
#     def __init__(self, window: int = 20):
#         self.window = window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window)
#         atr = (recent['high'] - recent['low']).mean()
#         prev_atr = features.iloc[-self.window-20:-self.window]['high'].mean() - features.iloc[-self.window-20:-self.window]['low'].mean()
#         price_up = recent['close'].iloc[-1] > recent['close'].iloc[0]
#         is_met = atr > prev_atr and price_up
#         confidence = min((atr - prev_atr) / prev_atr * 10, 1.0) if is_met else 0.0
#         explanation = f"ATR {atr:.2f} {'>' if is_met else '<='} 前期 {prev_atr:.2f}, 价格{'上涨' if price_up else '未上涨'}"
#         return is_met, confidence, explanation

# class UptrendRule(Rule):
#     """规则：上升趋势（5日均线 > 20日均线 > 50日均线）"""
#     def __init__(self, short_window: int = 5, mid_window: int = 20, long_window: int = 50):
#         self.short_window = short_window
#         self.mid_window = mid_window
#         self.long_window = long_window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         short_ma = features['close'].tail(self.short_window).mean()
#         mid_ma = features['close'].tail(self.mid_window).mean()
#         long_ma = features['close'].tail(self.long_window).mean()
#         is_met = short_ma > mid_ma > long_ma
#         confidence = min((short_ma - mid_ma) / mid_ma * 10, 1.0) if is_met else 0.0
#         explanation = f"5日均线 {short_ma:.2f}, 20日均线 {mid_ma:.2f}, 50日均线 {long_ma:.2f}"
#         return is_met, confidence, explanation

# class RSIAbove50Rule(Rule):
#     """规则：RSI > 50（动态阈值）"""
#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         rsi = features['rsi'].iloc[-1]
#         volatility = features['market_volatility'].iloc[-1] if 'market_volatility' in features else 0.01
#         threshold = 50 - 10 * volatility
#         is_met = rsi > threshold
#         confidence = min((rsi - threshold) / (100 - threshold), 1.0) if is_met else 0.0
#         explanation = f"RSI: {rsi:.2f} {'>' if is_met else '<='} 动态阈值 {threshold:.2f}"
#         return is_met, confidence, explanation

# class MACDPositiveAndAboveSignalRule(Rule):
#     """规则：MACD > 0 且 MACD > 信号线"""
#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         macd = features['macd'].iloc[-1]
#         signal = features['macd_signal'].iloc[-1]
#         is_met = macd > 0 and macd > signal
#         confidence = min((macd - signal) / abs(signal) * 10, 1.0) if is_met else 0.0
#         explanation = f"MACD: {macd:.2f}, 信号线: {signal:.2f}"
#         return is_met, confidence, explanation

# class PriceAboveBollingerUpperRule(Rule):
#     """规则：价格 > 上轨布林带"""
#     def __init__(self, window: int = 20):
#         self.window = window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window)
#         upper_band = recent['bollinger_upper'].iloc[-1]
#         latest_close = recent['close'].iloc[-1]
#         is_met = latest_close > upper_band
#         confidence = min((latest_close - upper_band) / upper_band * 10, 1.0) if is_met else 0.0
#         explanation = f"价格 {latest_close:.2f} {'>' if is_met else '<='} 上轨 {upper_band:.2f}"
#         return is_met, confidence, explanation

# class TrendReversalRule(Rule):
#     """规则：价格跌破20日均线且RSI<30"""
#     def __init__(self, window: int = 20):
#         self.window = window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window)
#         ma = recent['close'].mean()
#         latest_close = recent['close'].iloc[-1]
#         rsi = features['rsi'].iloc[-1]
#         is_met = latest_close < ma and rsi < 30
#         confidence = min((ma - latest_close) / ma * 10 + (30 - rsi) / 30, 1.0) if is_met else 0.0
#         explanation = f"价格 {latest_close:.2f} {'<' if is_met else '>='} 均线 {ma:.2f}, RSI: {rsi:.2f} {'<' if rsi < 30 else '>='} 30"
#         return is_met, confidence, explanation

# class VolumeSpikeRule(Rule):
#     """规则：成交量超过20日均量的2倍"""
#     def __init__(self, window: int = 20, threshold: float = 2.0):
#         self.window = window
#         self.threshold = threshold

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window)
#         avg_volume = recent['volume'].mean()
#         latest_volume = recent['volume'].iloc[-1]
#         is_met = latest_volume > avg_volume * self.threshold
#         confidence = min((latest_volume - avg_volume * self.threshold) / (avg_volume * self.threshold) * 10, 1.0) if is_met else 0.0
#         explanation = f"成交量 {latest_volume:.0f} {'>' if is_met else '<='} {self.threshold}x均量 {avg_volume * self.threshold:.0f}"
#         return is_met, confidence, explanation

# class PriceBreakoutHighRule(Rule):
#     """规则：价格突破前20日最高点"""
#     def __init__(self, window: int = 20):
#         self.window = window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window + 1)
#         prev_high = recent['high'].iloc[:-1].max()
#         latest_close = recent['close'].iloc[-1]
#         is_met = latest_close > prev_high
#         confidence = min((latest_close - prev_high) / prev_high * 10, 1.0) if is_met else 0.0
#         explanation = f"价格 {latest_close:.2f} {'>' if is_met else '<='} 前高 {prev_high:.2f}"
#         return is_met, confidence, explanation

# class MACDGoldenCrossRule(Rule):
#     """规则：MACD上穿信号线"""
#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         macd = features['macd'].iloc[-1]
#         signal = features['macd_signal'].iloc[-1]
#         prev_macd = features['macd'].iloc[-2]
#         prev_signal = features['macd_signal'].iloc[-2]
#         is_met = macd > signal and prev_macd <= prev_signal
#         confidence = min((macd - signal) / abs(signal) * 10, 1.0) if is_met else 0.0
#         explanation = f"MACD {macd:.2f} 上穿信号线 {signal:.2f} (前值: {prev_macd:.2f}, {prev_signal:.2f})"
#         return is_met, confidence, explanation

# class VolatilityContractionRule(Rule):
#     """规则：波动率（ATR）连续5天缩小"""
#     def __init__(self, window: int = 5):
#         self.window = window

#     def evaluate(self, features: pd.DataFrame) -> Tuple[bool, float, str]:
#         recent = features.tail(self.window + 1)
#         atr = (recent['high'] - recent['low']).diff()
#         is_met = all(atr.iloc[-i] < atr.iloc[-i-1] for i in range(1, self.window + 1))
#         confidence = 1.0 if is_met else 0.0
#         explanation = f"ATR最近{self.window}天{'连续缩小' if is_met else '未连续缩小'}"
#         return is_met, confidence, explanation

# class ExpertSystem:
#     """专家系统，整合规则、市场状态和推理引擎"""
#     def __init__(self, rules: List[Rule], rule_weights: Dict[str, float] = None):
#         """初始化专家系统
#         Args:
#             rules (List[Rule]): 规则列表
#             rule_weights (Dict[str, float]): 初始权重字典
#         """
#         self.rules = rules
#         self.base_weights = rule_weights or {rule.__class__.__name__: 1.0 for rule in rules}
#         self.rule_weights = self.base_weights.copy()
#         self.market_volatility = None
#         self.knowledge_base = {}

#     def adjust_weights(self, datasets: List[pd.DataFrame], conditions: List[MarketCondition]) -> Dict[str, float]:
#         """动态调整规则权重"""
#         self.rule_weights = self.base_weights.copy()
#         for condition in conditions:
#             if condition.evaluate(datasets):
#                 condition.apply_adjustments(self.rule_weights)
#                 logging.info(f"应用市场状态: {condition.__class__.__name__}")
#         logging.info(f"动态调整权重: {self.rule_weights}")
#         return self.rule_weights

#     def extract_facts(self, features: pd.DataFrame, contract: str) -> Dict[str, Tuple[bool, float, str]]:
#         """从特征数据中提取事实，存入知识库"""
#         if 'market_volatility' not in features.columns:
#             features['market_volatility'] = self.market_volatility
#         facts = {}
#         for rule in self.rules:
#             rule_name = rule.__class__.__name__
#             is_met, confidence, explanation = rule.evaluate(features)
#             facts[rule_name] = (is_met, confidence, explanation)
#             logging.debug(f"{contract} - {rule_name}: {explanation}, 置信度: {confidence:.2f}")
#         self.knowledge_base[contract] = facts
#         return facts

#     def evaluate_dependencies(self, rule_name: str, facts: Dict, dependencies: Dict) -> Tuple[bool, float]:
#         """递归评估规则的依赖关系，返回是否满足及增强倍数"""
#         if rule_name not in dependencies:
#             return True, 1.0
#         dep_config = dependencies[rule_name]
#         requires = dep_config.get("requires", [])
#         logic = dep_config.get("logic", "and")
#         boost = dep_config.get("boost", 1.0)

#         dep_results = []
#         for req in requires:
#             if isinstance(req, str):
#                 is_met, _, _ = facts.get(req, (False, 0.0, ""))
#                 dep_results.append(is_met)
#             else:
#                 sub_rule = req["rule"]
#                 sub_met, sub_boost = self.evaluate_dependencies(sub_rule, facts, dependencies)
#                 dep_results.append(sub_met)
#                 boost *= sub_boost if sub_met else 1.0

#         if logic == "and":
#             condition_met = all(dep_results)
#         elif logic == "or":
#             condition_met = any(dep_results)
#         elif logic == "not":
#             condition_met = not all(dep_results)
#         else:
#             condition_met = True
#         return condition_met, boost if condition_met else 1.0

#     def infer_strength(self, facts: Dict[str, Tuple[bool, float, str]], dependencies: Dict[str, Dict] = None) -> Tuple[float, str]:
#         """推理强弱得分，支持独立和依赖模式"""
#         strength_score = 0.0
#         total_weight = sum(self.rule_weights.values())
#         explanation = "推理过程：\n"
#         dependencies = dependencies or {}

#         for rule_name, (is_met, confidence, rule_explanation) in facts.items():
#             weight = self.rule_weights.get(rule_name, 1.0)
#             dep_met, boost = self.evaluate_dependencies(rule_name, facts, dependencies)
#             dep_explanation = f" (依赖满足，增益 {boost:.2f}x)" if boost != 1.0 and is_met and dep_met else ""

#             if is_met and dep_met:
#                 adjusted_contribution = confidence * weight * boost
#             else:
#                 adjusted_contribution = 0.0

#             explanation += f"- {rule_name}: {rule_explanation}, 权重: {weight}, 贡献: {adjusted_contribution:.2f}{dep_explanation}\n"
#             strength_score += adjusted_contribution

#         final_score = strength_score / total_weight if total_weight > 0 else 0.0
#         explanation += f"最终得分: {final_score:.2f}"
#         return final_score, explanation

#     def evaluate(self, feature_datasets: List[pd.DataFrame], condition_map: Dict[str, type], config_path: str = "rules_config.yaml") -> Dict[str, Tuple[float, str]]:
#         """评估所有合约的强弱"""
#         config_full_path = os.path.join(os.path.dirname(__file__), config_path)
#         with open(config_full_path, "r", encoding='utf-8') as f:
#             rules_config = yaml.safe_load(f)
#         conditions = [condition_map[cond["type"]](cond["adjustments"]) for cond in rules_config.get("market_conditions", [])]
#         dependencies = rules_config.get("dependencies", {})
        
#         if rules_config.get("auto_weights", False):
#             from ..weight_generator.generate_weights import WeightGenerator
#             generator = WeightGenerator()
#             self.base_weights = generator.generate("rules", feature_datasets, self.rules)
#             self.rule_weights = self.base_weights.copy()
#             generator.update_config("rules", self.base_weights, config_full_path)
#         else:
#             self.base_weights = rules_config.get("weights", {rule.__class__.__name__: 1.0 for rule in self.rules})
#             self.rule_weights = self.base_weights.copy()

#         self.rule_weights = self.adjust_weights(feature_datasets, conditions)
#         market_atr = np.mean([df['high'].tail(20).mean() - df['low'].tail(20).mean() for df in feature_datasets])
#         self.market_volatility = market_atr / feature_datasets[0]['close'].mean()
#         results = {}
#         for i, features in enumerate(feature_datasets):
#             contract = f"contract{i+1}"
#             facts = self.extract_facts(features, contract)
#             score, explanation = self.infer_strength(facts, dependencies)
#             results[contract] = (score, explanation)
#             logging.info(f"{contract} 强弱评估完成，得分: {score:.2f}\n{explanation}")
#         return results