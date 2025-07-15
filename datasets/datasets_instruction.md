# 数据集说明 (中文版)

## 数据集概览
该数据集包含首届WiFi感知大赛家庭场景的相关数据。数据提供了每个房间的CSI信号，用于检测房间内是否有人以及房间内的人数。该数据集可用于基于WiFi的人体存在检测研究。

## 数据文件描述
- **文件格式:** `txt`
- **压缩包格式:** `ZIP`
- **数据集大小:** 342MB
- **数据量:** 4组样本数据
- **数据类型:** CSI数据
- **数据采集时长:** 20分钟
- **文件命名规则:** `csi_time_number`
- **详细描述:** 数据集附带的 .txt 文件描述了每条数据的采集场景和干扰情况。
- **数据格式:**
    - **家庭场景1:** 一个数据包包含1008个数据 [ts: 时, 分, 秒(3个), rssi(2个), mcs(1个), gain(2个), csi(4*250)]。
    - **家庭场景2:** 一个数据包包含1000个数据 [ts: 时, 分, 秒(3个), rssi(2个), mcs(1个), gain(2个), csi(4*248)]。

## 采集场景及设备信息
- **采集场景:** 家庭场景，可分为家庭场景1和2。场景中的干扰包括风吹绿植干扰和扫地机器人干扰。
- **采集时间:** 2023.06 - 2023.10
- **频段:** 5GHz
- **带宽:** 160 MHz (248个子载波数据), 80 MHz (250个子载波数据)
- **协议:** 802.11ax
- **波形:** OFDM
- **采样率:** 约20Hz
- **天线:** 每个设备2根天线
- **分段/时间段长度:** 总时长在299秒到300秒之间。每2秒是一个时间窗口，对应一个真值。

---

## 数据分析摘要

### 数据加载逻辑
数据处理脚本 (`src/utils.py`) 已更新，以处理此数据集的特定格式。

- **单行格式:** 脚本可以正确读取所有数据点都在一个单行内，并以空格分隔的文件。这适用于CSI数据文件和真值标签文件。
- **数据包重塑:** 脚本使用每个数据包的已知元素数量（场景1为1008，场景2为1000），将长的一维数组重塑为 `[数据包数量, 每个数据包的元素数量]` 的二维数组。
- **接收端数据同步:** 脚本通过将两个接收器（`room_A` 和 `room_B`）的数据截断到相同的较短长度，来处理它们之间微小的数据包数量差异，从而确保数据在拼接前正确对齐。

### 多对一映射：CSI数据包到真值标签
CSI数据包与真值标签之间并非一对一的对应关系，而是多个CSI数据包对应一个标签。代码通过为每个标签分组一个CSI数据包窗口，然后从该组中创建一个训练样本来实现此逻辑。每个标签对应的平均数据包数量在下面的详细分析中显示。

---

## 详细文件大小分析
*此分析由 `src/analyze_dataset.py` 生成。*

### 家庭场景1分析

**CSI数据文件**
| 文件路径 | 总元素数 | 计算出的数据包数 |
| :--- | :--- | :--- |
| `.../room_A/csi_2023_10_30_1.txt` | 6,739,488 | 6,686 |
| `.../room_A/csi_2023_10_30_2.txt` | 6,746,544 | 6,693 |
| `.../room_B/csi_2023_10_30_1.txt` | 6,717,312 | 6,664 |
| `.../room_B/csi_2023_10_30_2.txt` | 6,832,224 | 6,778 |

**每个标签对应的数据包**
| 实验 | 最小数据包数 (A & B) | 总标签数 | 平均数据包/标签 |
| :--- | :--- | :--- | :--- |
| `csi_2023_10_30_1` | 6,664 | 150 | 44.43 |
| `csi_2023_10_30_2` | 6,693 | 150 | 44.62 |

---
### 家庭场景2分析

**CSI数据文件**
| 文件路径 | 总元素数 | 计算出的数据包数 |
| :--- | :--- | :--- |
| `.../room_A/csi_2023_10_30_1.txt` | 6,544,000 | 6,544 |
| `.../room_A/csi_2023_10_30_2.txt` | 6,714,000 | 6,714 |
| `.../room_B/csi_2023_10_30_1.txt` | 6,500,000 | 6,500 |
| `.../room_B/csi_2023_10_30_2.txt` | 6,656,000 | 6,656 |

**每个标签对应的数据包**
| 实验 | 最小数据包数 (A & B) | 总标签数 | 平均数据包/标签 |
| :--- | :--- | :--- | :--- |
| `csi_2023_10_30_1` | 6,500 | 150 | 43.33 |
| `csi_2023_10_30_2` | 6,656 | 150 | 44.37 |

---
<br>

# Dataset Instruction (English Version)

## Dataset Overview
The dataset contains the data related to the home scenario of the first WiFi sensing contest. The dataset provides the CSI signals of each room to detect the existence of human bodies in the room and the number of people in the room. This dataset can be used for research on WiFi-based human presence detection.

## Data File Description
- **File format:** `txt`
- **Compressed package format:** `ZIP`
- **Dataset size:** 342MB
- **Data volume:** 4 groups of sample data
- **Data type:** `CSI data`
- **Data collection duration:** 20 min
- **File naming rules:** `csi_time_number`
- **Detailed description:** The .txt file attached to the data set describes the collection scenario and interference of each piece of data.
- **Data format:**
    - **Home scenario1:** One data packet contains 1008 data [ts: hour, minute, second(3 data records), rssi(2 data records), mcs(1 data record), gain(2 data records), csi(4*250)].
    - **Home scenario2:** One data packet contains 1000 data [ts: hour, minute, second(3 data records), rssi(2 data records), mcs(1 data record), gain(2 data records), csi(4*248)].

## Collecting Scenarios And Device Information
- **Collection scenarios:** The collection scenario is home scenario, which can be classified into home scenario 1, 2. The interference in the scenario includes wind blowing and green plant interference and floor sweeping robot interference.
- **Collection time:** 2023.06-2023.10
- **Frequency band:** 5Ghz
- **Bandwidth:** 160 MHz (data of 248 subcarriers), 80 MHz (data of 250 subcarriers)
- **Protocol:** 802.11ax
- **Waveform:** OFDM
- **Sampling rate:** About 20Hz
- **Antenna:** 2 antennas per device
- **Segment flag/Time segment length:** The total duration ranges from 299s to 300s. Every 2s is a time window, which corresponds to a true value.

---

## Data Analysis Summary

### Data Loading Logic
The data processing script (`src/utils.py`) has been updated to handle the specific format of this dataset.

- **Single-Line Format:** The script correctly reads files where all data points are on a single line, separated by spaces. This applies to both CSI data files and truth label files.
- **Packet Reshaping:** It uses the known number of elements per packet (1008 for Scenario 1, 1000 for Scenario 2) to reshape the long 1D array of numbers into a 2D array of `[num_packets, elements_per_packet]`.
- **RX Data Synchronization:** It handles minor differences in packet counts between `room_A` and `room_B` receivers by truncating both to the shorter length, ensuring proper data alignment before concatenation.

### Many-to-One Mapping: CSI Packets to Truth Labels
There is no one-to-one correspondence between a CSI packet and a truth label. Instead, many CSI packets correspond to a single label. The code implements this by grouping a window of CSI packets for each label and then creating one training sample from that group. The calculated average number of packets per label is shown in the detailed analysis below.

---

## Detailed File Size Analysis
*This analysis was generated by `src/analyze_dataset.py`.*

### Home Scenario 1 Analysis

**CSI Data Files**
| File Path | Total Elements | Calculated Packets |
| :--- | :--- | :--- |
| `.../room_A/csi_2023_10_30_1.txt` | 6,739,488 | 6,686 |
| `.../room_A/csi_2023_10_30_2.txt` | 6,746,544 | 6,693 |
| `.../room_B/csi_2023_10_30_1.txt` | 6,717,312 | 6,664 |
| `.../room_B/csi_2023_10_30_2.txt` | 6,832,224 | 6,778 |

**Packets per Label**
| Experiment | Min Packets (A & B) | Total Labels | Avg. Packets/Label |
| :--- | :--- | :--- | :--- |
| `csi_2023_10_30_1` | 6,664 | 150 | 44.43 |
| `csi_2023_10_30_2` | 6,693 | 150 | 44.62 |

---
### Home Scenario 2 Analysis

**CSI Data Files**
| File Path | Total Elements | Calculated Packets |
| :--- | :--- | :--- |
| `.../room_A/csi_2023_10_30_1.txt` | 6,544,000 | 6,544 |
| `.../room_A/csi_2023_10_30_2.txt` | 6,714,000 | 6,714 |
| `.../room_B/csi_2023_10_30_1.txt` | 6,500,000 | 6,500 |
| `.../room_B/csi_2023_10_30_2.txt` | 6,656,000 | 6,656 |

**Packets per Label**
| Experiment | Min Packets (A & B) | Total Labels | Avg. Packets/Label |
| :--- | :--- | :--- | :--- |
| `csi_2023_10_30_1` | 6,500 | 150 | 43.33 |
| `csi_2023_10_30_2` | 6,656 | 150 | 44.37 |