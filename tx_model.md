### 数据传输模型

- 传输资源配置
  - 频域子载波数 `S = 624/96`
  - 时域符号数 `T = 12`
  - 发送天线 `Nt = 2/32` (这个似乎没啥用)
  - 接收天线 `Nr = 2/4`
  - 传输层数 `L = 2/4`
  - 每符号比特数 `M = 4/6`

- 通信模型
  - 发送方: `S = sqrt(W)*D + sqrt(V)*P` (数据-导频非正交叠加)
    - `D` 为数据比特经 QAM 编码后的复数信号，`QAM(tx_bits): [L, T, S, c=2] float16`
    - `P` 为导频信号，`[L, T, S, c=2] vset {-1, +1}`
    - `W` 和 `V` 为权重 `[L, T, S] float`
      - scene1
        - const: `W=0.9, V=0.1`
      - scene2 
        - const: `W=0.6`
        - stripped: `t=l+4*k` ? `V=1.6` : `V=0`
          - `layer=1`, `token=1,5,9`
          - `layer=2`, `token=2,6,10`
          - `layer=3`, `token=3,7,11`
          - `layer=4`, `token=4,8,12`
  - 接收方: `Y = H * S + N`
    - `H` 为等效信道模型 `[L, T, S, Nr]`
    - `N` 为高斯白噪声 `[L, T, S, Nr]`

----

- 训练数据格式
  - tx_bits: 发送数据比特=预期解码比特，即 `invQAM(D): [L, T, S, M] bit`
  - rx_signal: 接收信号, 即 `Y: [L, T, S, c] float16`
    - scene1: `vrng [-9.04, 9.16]`
    - scene2: `vrng [-25.11, 23.86]`
  - pilot: 导频信号，即 `P: [L, T, S, c=2] vset {-1, 0, +1}` (无导频时为 `0`)

- 训练目标
  - 根据总公式 `Y = H * (sqrt(W) * QAM(bits) + sqrt(V) * P) + N`，已知 `Y, W, V, P` 求 `bits`
  - 则解析解 `bits = invQAM(((Y - N) * invH - sqrt(V) * P) / sqrt(W))` 中需要建模的部分
    - `N`
    - `invH`
    - `invQAM`
