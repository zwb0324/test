import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import warnings
import os

warnings.filterwarnings('ignore')

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_acf_and_plot():
    """
    读取Excel文件，计算自相关函数并绘制结果图
    """
    try:
        # 1. 读取Excel文件
        file_path = r"C:\Users\123456\Desktop\pachong\co2(0ns).xlsx"
        print("正在读取Excel文件...")
        df = pd.read_excel(file_path)

        # 检查数据列
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")

        # 假设第一列是序号，第二列是数目
        # 如果列名不是默认的，可以根据实际情况调整
        if len(df.columns) >= 2:
            numbers_column = df.iloc[:, 1]  # 第二列（数目列）
        else:
            raise ValueError("Excel文件需要至少包含两列数据")

        print(f"数据样本前5行:")
        print(numbers_column.head())

        # 2. 计算自相关函数
        print("\n正在计算自相关函数...")

        # 计算ACF，设置合适的滞后阶数
        # 通常滞后阶数取数据长度的1/4到1/2
        nlags = min(100, len(numbers_column) // 2)
        acf_values = acf(numbers_column, nlags=nlags, fft=True)

        print(f"计算了 {len(acf_values)} 个滞后点的ACF值")
        print(f"ACF值范围: {acf_values.min():.4f} ~ {acf_values.max():.4f}")

        # 3. 创建时间轴（基于0.2ns的间隔）
        time_lags = np.arange(len(acf_values)) * 0.2  # 转换为ns单位

        # 4. 绘制自相关函数图
        plt.figure(figsize=(12, 8))

        # 主图：ACF函数
        plt.subplot(2, 1, 1)
        # 更新stem用法，避免use_line_collection参数警告
        markerline, stemlines, baseline = plt.stem(time_lags, acf_values, basefmt=" ")
        plt.setp(stemlines, 'linewidth', 1)
        plt.setp(markerline, 'markersize', 3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

        # 添加置信区间（95%）
        conf_int = 1.96 / np.sqrt(len(numbers_column))
        plt.axhline(y=conf_int, color='gray', linestyle='--', alpha=0.5, label='95% 置信区间')
        plt.axhline(y=-conf_int, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('时间延迟 (ns)')
        plt.ylabel('自相关函数')
        plt.title('自相关函数 (ACF) 分析')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 子图：原始数据
        plt.subplot(2, 1, 2)
        time_original = np.arange(len(numbers_column)) * 0.2
        plt.plot(time_original, numbers_column)
        plt.xlabel('时间 (ns)')
        plt.ylabel('数目')
        plt.title('原始数据序列')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 自动保存图片
        plot_filename = 'ACF_co2(0ns).png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"ACF图表已保存为 '{plot_filename}'")

        plt.show()

        # 5. 保存ACF结果到新的Excel文件
        acf_df = pd.DataFrame({
            '时间延迟_ns': time_lags,
            'ACF值': acf_values
        })

        excel_filename = 'ACF_co2(0ns).xlsx'
        acf_df.to_excel(excel_filename, index=False)
        print(f"ACF计算结果已保存到 '{excel_filename}'")

        # 6. 打印一些统计信息
        print("\n=== 统计信息 ===")
        print(f"数据点数: {len(numbers_column)}")
        print(f"数据均值: {numbers_column.mean():.4f}")
        print(f"数据标准差: {numbers_column.std():.4f}")
        print(f"ACF第一个零点位置: {find_first_zero_crossing(acf_values, time_lags)}")

        return acf_values, time_lags

    except FileNotFoundError:
        print("错误: 未找到文件，请确保文件存在于指定目录")
        return None, None
    except Exception as e:
        print(f"发生错误: {e}")
        return None, None


def find_first_zero_crossing(acf_values, time_lags):
    """
    找到ACF第一次穿过零点的位置
    """
    for i in range(1, len(acf_values)):
        if acf_values[i] * acf_values[i - 1] <= 0 and acf_values[i] <= 0:
            return f"{time_lags[i]:.2f} ns (滞后 {i})"
    return "未找到零点"


# 高级分析：计算不同统计量
def advanced_acf_analysis(acf_values, time_lags):
    """
    进行更深入的ACF分析
    """
    print("\n=== 高级分析 ===")

    # 找到相关性衰减到0.1的时间
    decay_threshold = 0.1
    decay_index = np.where(np.abs(acf_values) < decay_threshold)[0]
    if len(decay_index) > 0:
        decay_time = time_lags[decay_index[0]]
        print(f"相关性衰减到 {decay_threshold} 的时间: {decay_time:.2f} ns")

    # 计算积分相关时间
    integral_time = np.trapz(np.abs(acf_values), time_lags)
    print(f"积分相关时间: {integral_time:.4f} ns")

    return integral_time


# 运行主函数
if __name__ == "__main__":
    # 计算ACF并绘图
    acf_values, time_lags = calculate_acf_and_plot()

    # 如果成功计算ACF，进行高级分析
    if acf_values is not None:
        integral_time = advanced_acf_analysis(acf_values, time_lags)