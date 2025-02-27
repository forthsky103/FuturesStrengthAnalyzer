# Rules Module Documentation

本模块实现了一个基于规则的专家系统，用于分析期货合约的强弱，支持动态适应市场条件和规则间的依赖关系。本文档梳理 `src/rules/` 内的代码逻辑，详细解释关键类和配置文件的使用方法，帮助用户快速配置和扩展。

## 目录结构
src/rules/
├── init.py           # 模块初始化文件
├── evaluator.py          # 专家系统核心（知识库和推理引擎）
├── main_rules.py         # 主函数
├── rules_config.json     # 配置文件
├── rules_module.md       # 本文档
## 代码逻辑梳理

### `evaluator.py`
核心文件，包含规则类、市场状态类和专家系统。

#### 1. `Rule` 类（抽象基类）
- **作用**: 定义规则接口，所有具体规则必须继承并实现 `evaluate` 方法。
- **方法**:
  - `evaluate(features: pd.DataFrame) -> Tuple[bool, float, str]`: 评估规则，返回是否满足、置信度和解释。
- **示例规则**:
  - `BreakoutMARule`: 检查价格是否突破均线，阈值随市场波动率动态调整。
  -  `RSIAbove50Rule`: 检查 RSI 是否高于动态阈值。

#### 2. `MarketCondition` 类（抽象基类）
- **作用**: 封装市场状态判断和权重调整逻辑，支持动态适应市场条件。
- **方法**:
- `evaluate(datasets: List[pd.DataFrame]) -> bool`: 判断是否满足条件。
- `apply_adjustments(weights: Dict[str, float]) -> Dict[str, float]`: 调整规则权重。
- **具体类**:
- `HighVolatilityCondition`: 若 ATR > 均值 + 标准差，提升波动规则权重。
- `TrendMarketCondition`: 若 ADX > 25，提升趋势规则权重。

#### 3. `ExpertSystem` 类
- **作用**: 整合规则、市场状态和推理引擎，评估合约强弱。
- **关键方法**:
- `adjust_weights(datasets, conditions)`: 调用市场状态类列表，动态调整权重。
- `extract_facts(features, contract)`: 从特征数据中提取事实，存入知识库。
- `evaluate_dependencies(rule_name, facts, dependencies)`: 递归解析依赖，支持 AND/OR/NOT 和多级依赖。
- `extract_facts(features, contract)`: 从特征数据中提取事实，存入知识库。
- `evaluate_dependencies(rule_name, facts, dependencies)`: 递归解析依赖，支持 AND/OR/NOT 和多级依赖。
- - `infer_strength(facts, dependencies)`: 推理强弱，支持独立和依赖模式。
- `evaluate(feature_datasets, condition_map, config_path)`: 主方法，协调权重调整和推理，`condition_map` 从外部传入。

### `main_rules.py`
- **作用**: 加载配置，初始化数据和特征，运行专家系统。
- **逻辑**:
1. 加载配置文件，初始化规则和特征。
2. 对齐数据时间，提取特征。
3. 创建 `ExpertSystem` 实例，传入 `condition_map`，运行评估，生成建议。
- **关键函数**:
- `get_feature_objects()`: 定义特征列表，与 `features.py` 协作。

## 配置文件详解：`rules_config.json`

### 文件结构
`rules_config.json` 是规则模块的核心配置文件，控制规则、权重、市场状态和依赖关系。结构如下：
```json
{
"rules": { /* 规则配置 */ },
"weights": { /* 基础权重 */ },
"auto_weights": false, /* 是否自动生成权重 */
"market_conditions": [ /* 市场状态 */ ],
"dependencies": { /* 依赖关系 */ }
}



---

### 重点解释配置文件：`rules_config.json`

#### 为什么重要？
- **核心控制**: 配置文件是模块的“大脑”，决定了规则选择、权重分配、市场状态调整和依赖关系。
- **灵活性**: 通过修改配置文件，用户无需触及代码即可调整系统行为。
- **扩展性**: 支持新增规则、市场状态和依赖，只需扩展配置。

#### 关键字段详解
1. **`rules`**:
   - **功能**: 指定所有规则及其参数，驱动 `ExpertSystem` 的规则评估。
   - **灵活性**: 支持任意规则类，只要在 `evaluator.py` 中定义并在 `main_rules.py` 的 `rule_map` 中注册。
   - **示例场景**: 
     - 想测试短期趋势，添加 `"ShortTrendRule": {"window": 5}`，并实现对应类。
   - **调试技巧**: 减少规则数量（如只留 `"BreakoutMARule"`），观察单一规则效果。

2. **`weights`**:
   - **功能**: 定义规则的基础权重，直接影响强弱得分。
   - **手动 vs 自动**: 
     - 手动：直接编辑，如 `"BreakoutMARule": 1.5`。
     - 自动：设置 `"auto_weights": true`，运行 `generate_weights.py`，生成基于数据的结果。
   - **示例场景**: 
     - 若认为成交量更重要，手动将 `"VolumeIncreaseRule": 0.8` 改为 `1.2`。
   - **调试技巧**: 初始设为均匀权重（如全为 1.0），观察自动生成的变化。

3. **`auto_weights`**:
   - **功能**: 切换手动和自动权重模式。
   - **使用场景**: 
     - 开发初期设为 `false`，手动调整验证。
     - 数据充足后设为 `true`，依赖工具优化。
   - **示例**: 
     - `"auto_weights": true`，运行脚本后检查 `weights` 是否合理。

4. **`market_conditions`**:
   - **功能**: 根据市场状态动态调整权重，增强适应性。
   - **灵活性**: 支持自定义状态类，只需新增类并更新 `condition_map`。
   - **示例场景**: 
     - 添加震荡市场状态，提升动量规则权重：

- **调试技巧**: 减少状态数量，观察单一状态（如高波动）的影响。

5. **`dependencies`**:
- **功能**: 定义规则间的逻辑依赖，提升推理能力。
- **复杂性**: 支持 AND/OR/NOT 和多级嵌套。
- **示例场景**: 
- 要求趋势规则需动量确认：


- **调试技巧**: 先禁用依赖（留空），验证独立推理，再逐步添加。

#### 配置注意事项
- **一致性**: `rules` 和 `weights` 中的规则名必须匹配。
- **路径**: 默认路径为 `"rules_config.json"`，若移动需同步修改 `evaluate` 参数。
- **格式**: JSON 需严格遵守，避免语法错误。

---

### 下一步
- **文件确认**: 请将 `rules_module.md` 粘贴到 `src/rules/` 下，验证是否满足需求。
- **后续改进**: 若需更详细的配置文件示例（如复杂依赖场景），我可以进一步补充。
- **其他需求**: 若有其他调整（如代码优化、`generate_weights.py` 的使用说明），随时告诉我！

你的认可是我最大的动力，有什么想法就告诉我吧！