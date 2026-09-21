# 十经济体 NVAR 实证复现

本文件夹包含十经济体全样本结构估计的数据、算法、运行入口、结果和核验程序。运行环境为 Python 3.12。

秩固定为 `(r_G,r_H)=(2,1)`，因子数 `r_f=1`、滞后 `p=1`。同期稀疏网络的罚参数 `lambda_0` 在原样本上按 BIC 选择。

`algorithm1/` 提供均值网络 Algorithm 1 的单独复现：先用核范数罚参数选择秩 `(2,1)`，再按该秩重估两个网络和共同因子。该目录含自身的输入数据、代码、README 和运行结果。

## 1. 一次完整运行

在本文件夹中打开终端：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --draws 300 --output results/my_run
python verify.py results/my_run
python plot.py results/my_run
```

Windows 激活命令为 `.venv\Scripts\activate`。若系统命令名为 `python`，用对应的 Python 3.12 解释器替换第一行。

所有相对输出路径均相对于本文件夹，而非当前工作目录。输出目录必须不存在；重复运行需使用新目录，例如 `results/my_run_2`。程序每次从输入数据开始估计。

快速检查可用：

```sh
python run.py --draws 20 --output results/quick_check
python verify.py results/quick_check
```

20 次的区间仅用于程序检查。默认也是 20 次，正式数值回放应显式指定 `--draws 300`。仅估计点估计和 BIC 路径可用 `--draws 0`。

## 2. 已包含的运行结果

`results/reference_300/` 从 CSV 输入重新估计，包含 300 次结构 bootstrap，随机种子为 `314159`。

| 参数 | 点估计 | 95% basic bootstrap 区间 |
|---|---:|---:|
| beta_0 | 1.264106 | [1.231455, 1.745660] |
| beta_1 | 0.649261 | [0.392798, 0.646917] |
| rho | 0.481593 | [0.167798, 0.599035] |

原样本 BIC 选出 `lambda_0=0.20`，支持大小 21，伴随矩阵谱半径 `0.330126674`。300/300 次抽样成功；所选罚目标迭代均达到代码停止条件；点估计和这 300 个重拟合均未达到网络范数边界。停止条件成立不意味着全局最优。

`results/demo/` 是先行的 20 次程序检查。主要参考 `reference_300`。

## 3. 数据与来源

`data/source_workbook.xlsx` 是原项目 `data&code/Z-score标准化.xlsx` 的原样复制，SHA-256：

```text
c766a6b3204e811bf874910593e33f5b647a356eba7e7e8a3581da17c5e4e433
```

运行程序使用两个 CSV：

- `data/trade_levels.csv`：来自工作表 `Trade转置N=10`。
- `data/reer_levels.csv`：来自工作表 `REER转置N=10`，截取与贸易相同的月份。
- `data/metadata.json`：国家、月份、输入状态和文件哈希。

国家顺序固定为 `CHN, USA, DEU, NLD, JPN, FRA, ITA, GBR, KOR, HKG`。两个 CSV 均为 2000-01 至 2024-06，共 294 个月；每行一个月份、每列一个国家。REER 工作表原本延伸到 2024-12，代码按共同月份截取。

输入为工作簿中已经标准化的水平数据。

`code/data_io.py` 每次从 CSV 重做：

1. 对水平数据做一阶差分，**不取对数**。
2. 按国家减去各日历月份的全样本均值，以去除季节性。
3. 按国家全样本中心化，用样本标准差（`ddof=1`）标准化。

得到 2000-02 至 2024-06 的 293 个变化值；留出一期初始滞后后，估计使用 2000-03 至 2024-06 的 292 个响应月份。预处理在全样本上进行。

如果要从随附工作簿重新生成完全相同的 CSV：

```sh
python -m pip install -r requirements-data.txt
python export_data.py
```

该命令重写本目录的两个 CSV 和 `metadata.json`；工作簿保持不变。

## 4. 估计算法

### 均值网络与共同因子

`code/mean_model.py` 实现：给定因子和系数依次更新 peer/contextual 降秩回归，最小二乘更新系数，再用 PCA 更新共同成分，直到目标函数相对变化小于 `1e-6`。每次允许最多 3000 次迭代；使用两个确定性初值及一个共同网络 warm start，保留最低目标值的解。程序检查所选解的收敛与逐块下降。

估计后进行保持拟合乘积不变的正尺度归一化。`F'F/n=1`，共同成分为 `Lambda F'`。

### 同期结构网络

令 `S_u` 为减去均值网络、尚未减去共同因子的残差二阶矩。对候选顺序与罚参数拟合：

```text
||S_u - omega Lambda Lambda' - tau (I-B)^(-1)(I-B)^(-T)||_F^2
    + lambda_0 sum_ij |B_ij|.
```

约束：给定顺序下 B 严格下三角，`||B||op <= 0.995`，`||B||F <= 2`，`0 <= omega <= 2`，`1e-4 <= tau <= 5`。行是接受者、列是传递者。

`code/structural.py` 在五个共同成分扣除比例下，使用标准 Cholesky 等方差评分筛选顺序。每个可行比例检查恒等顺序及 256 个带种子的随机顺序，再做插入式局部搜索，最多保留三个不同顺序。`--random-orders` 可更改随机顺序预算；修改预算是不同的运行设置。

在这些顺序上用带回溯的 L1 软阈值步骤拟合罚目标，剖面更新有界的 `omega,tau`。每个顺序的罚路径从小到大 warm start；每个罚参数保留最低罚目标候选。迭代上限 1200；停止规则为系数变化小于 `1e-6`，或目标相对变化小于 `1e-8` 且系数变化小于 `1e-4`。采用有限预算的非凸局部数值搜索。

以 `|B_ij| > 0.001` 提取支持，随后在该支持上无惩罚重拟合高斯协方差准似然。求解器先用 L-BFGS-B，失败或违反范数约束时用显式约束 SLSQP。BIC 为：

```text
n * [logdet(Sigma) + trace(Sigma^(-1) S_u)]
    + (support_size + 2) * log(n).
```

原样本网格固定为 `[0,.01,.02,.04,.06,.08,.10,.12,.15,.20,.30]`。`selection_path.csv` 保存各候选的边数、BIC、罚目标范围、停止状态和 QML 可行/收敛状态。无可行收敛 QML 解时程序报错。罚目标停止状态单独记录。

### 结构对象

令 `A=I-B`，恢复 `beta_0=||B||F`、`G0=B/beta_0`。对约化式乘积 `D_G=Pi_G beta`、`D_H=Pi_H rho`，报告 `beta_1=||A D_G||F`、`rho=||A D_H||F`，并把各乘积除以其正范数得到 `G,H`。所有方向符号留在矩阵中。

## 5. 递归结构 Bootstrap

在原样本拟合后固定 `p=1`、`(r_G,r_H,r_f)=(2,1,1)`、选定 `lambda_0`、REER 路径、初始贸易变化及拟合因子路径。每次生成：

```text
eta_t* ~ iid N(0,I_N)
e_t* = sqrt(tau_hat) (I-B_hat)^(-1) eta_t*
y_t* = Pi_G_hat y_(t-1)* beta_hat
       + Pi_H_hat x_t rho_hat
       + sqrt(omega_hat) Lambda_hat f_hat_t + e_t*.
```

随机创新协方差为 `tau_hat (I-B_hat)^(-1)(I-B_hat)^(-T)`。共同成分按 `sqrt(omega_hat)` 缩放。

每次重新估计均值网络、因子、顺序、支持和无惩罚协方差 refit；**不重新搜索 lambda 网格，不重新做 BIC 选 lambda**。BIC 表达式仍可用于记录单个候选，但 bootstrap 路径长度严格为 1。每次生成及拟合的种子由基础种子和抽样编号确定。

Basic 区间按 `2*theta_hat - quantile(theta_star,[.975,.025])` 计算，不额外重复扣除 bootstrap 均值。每个抽样都保存完整网络矩阵和 theta；失败会记录 traceback、返回非零退出码，并停止报告区间，不静默丢弃失败抽样。

## 6. 文件与核验

| 文件 | 用途 |
|---|---|
| `run.py` | 从 CSV 开始重新估计并运行 bootstrap |
| `code/data_io.py` | 相对路径读数和预处理 |
| `code/mean_model.py` | 交替降秩回归、PCA、递归生成 |
| `code/structural.py` | 顺序筛选、受约束罚目标、QML refit |
| `code/estimation.py` | BIC 选择、结构归一化、结构 bootstrap 组合 |
| `verify.py` | 源码/数据哈希、DAG/范数约束、生成器恒等式及区间回放 |
| `plot.py` | 从 `point.npz` 绘制网络图 |
| `results/reference_300/point.npz` | 所有点估计和因子 |
| `results/reference_300/draw_*.npz` | 每个 bootstrap 完整估计 |
| `results/reference_300/bootstrap.npz` | theta 和所有抽样的汇总 |
| `results/reference_300/strengths.csv` | 三个强度与 basic 区间 |
| `results/reference_300/manifest.json` | 种子、环境、设置、源码和数据哈希 |
| `results/reference_300/summary.json` | 状态、点估计、选定罚参数和运行时间 |
| `VALIDATION.json` | 本次核验及搬移后重跑结果 |

结果采用 NPZ 格式保存。theta 布局为 `[beta0,beta1,rho, vec(peer),vec(contextual),vec(B)]`，矩阵按行展开；N=10 时向量长度为 303。区间 CSV 不包含对挑选后的边的同时推断。

程序不读本文件夹以外的数据或结果。固定种子下，同一数值环境应复现数值；跨 BLAS、平台及依赖版本的非凸优化结果可能有差异，因此记录了测试环境和依赖版本。`verify.py` 中的文件哈希针对同一版本代码；改动代码应重新运行。
