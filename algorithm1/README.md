# Algorithm 1：十经济体均值网络与秩 `(2,1)`

本目录包含十经济体月度数据、两个网络的秩选择、Algorithm 1 的均值网络及共同因子估计。`data/trade_levels.csv` 和 `data/reer_levels.csv` 是可直接运行的输入；每行一个月份，每列一个经济体。国家为 CHN、USA、DEU、NLD、JPN、FRA、ITA、GBR、KOR、HKG，时间为 2000-01 至 2024-06。

要求 Python 3.12。进入本目录运行：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_algorithm1.py --output results/new_run
python verify_algorithm1.py results/new_run
```

输出目录必须是新目录，程序每次重新读取本目录数据并重新估计。已经完成的示例在 `results/selected/`。

## 数据处理和模型

代码先对水平数据做一阶差分，按国家减去日历月份的均值，再按国家用全样本均值和样本标准差标准化。得到 293 个月度变化；使用一期滞后后，估计样本为 292 个月。数据加载在 `code/data_io.py`。

均值方程为

```text
y_t = Pi_G y_(t-1) beta_1 + Pi_H x_t rho + Lambda f_t + e_t.
```

`code/mean_model.py` 用交替降秩回归更新 `Pi_G` 和 `Pi_H`，用最小二乘更新 `beta_1` 和 `rho`，再用 PCA 更新一个共同因子。使用三个确定性初值，保留残差目标最小的结果。滞后阶数为 1，共同因子数为 1。

## 秩选择和罚参数

`code/rank_selection.py` 对 peer 和 contextual 两个**拟合信号**分别施加核范数惩罚，并在每次条件更新中做奇异值软阈值，随后对剩余矩阵做秩一 PCA。对收敛解的两个无惩罚部分拟合信号，以其奇异值超过各自的 `lambda_G`、`lambda_H` 的数量作为所选秩。

`run_algorithm1.py` 运行记录在脚本中的乘数网格，保存**全部 289 组**结果到 `penalty_grid.csv`。两个罚参数分别为

```text
lambda_j = multiplier_j * sigma_hat * (sqrt(rank(design_j)) + sqrt(N)),
```

其中 `sigma_hat` 来自减去初始秩一 PCA 共同成分后的残差均方根。程序使用乘数 `(1.44,1.20)` 计算两个罚参数，并保存网格上全部计算结果。秩由部分拟合信号的奇异值阈值计算。

对应 `lambda_G=6.2290126489`、`lambda_H=5.1908438741`。两个阈值秩和罚后信号秩都是 `(2,1)`。随后按这两个秩运行 Algorithm 1，重估所得 `Pi_G` 和 `Pi_H` 的实际数值秩也是 `(2,1)`。均值残差目标 `n^{-1}||E||_F^2=4.1694221285`，伴随矩阵谱半径 `0.3301266740`。

## 输出

| 文件 | 内容 |
|---|---|
| `penalty_grid.csv` | 全部罚参数、所选秩、收敛状态、迭代次数和目标值 |
| `algorithm1_fit.npz` | 两个网络、系数、共同因子、拟合值及部分拟合的奇异值 |
| `manifest.json` | 所选罚参数、秩、样本、环境与文件 SHA-256 |
| `verify_algorithm1.py` | 从输入重算阈值秩、网络秩及均值残差目标 |

网络的行表示接受者，列表示传递者。`Pi_G`、`Pi_H` 的 Frobenius 范数为 1；有符号系数另存为 `beta`、`rho`。对不同 BLAS 或 NumPy 版本，非凸迭代可能有数值差异；运行时版本记录在 `manifest.json`。
