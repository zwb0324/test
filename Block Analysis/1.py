import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import warnings
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')


def calculate_standard_error(data, group_size, include_remainder=True):
    n = len(data)
    if n < 2:  # 至少需要2个数据点
        return np.nan

    # 计算完整组和剩余数据
    num_full_groups = n // group_size
    remainder = n % group_size

    # 确定是否包含剩余数据作为一组
    total_groups = num_full_groups
    if include_remainder and remainder > 0:
        total_groups += 1

    # 至少需要3组才能计算标准误差
    if total_groups < 3:
        return np.nan

    # 分割数据
    groups = []
    start = 0

    # 添加完整组
    for i in range(num_full_groups):
        end = start + group_size
        groups.append(data[start:end])
        start = end

    # 如果条件满足，添加剩余数据作为一组
    if include_remainder and remainder > 0:
        groups.append(data[start:start + remainder])

    # 计算每组平均值
    group_means = [np.mean(group) for group in groups]

    # 计算标准误差 = 标准差 / sqrt(组数)
    standard_error = np.std(group_means, ddof=1) / np.sqrt(len(group_means))

    return standard_error


def main():
    # 1. 读取Excel文件（使用绝对文件路径）
    file_path = r"C:\Users\123456\Desktop\pachong\co2渗透数目.xlsx"
    try:
        df = pd.read_excel(file_path)
        print("成功读取Excel文件")
        print(f"文件路径: {file_path}")
        print(f"数据形状: {df.shape}")
        print(df.head())
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 2. 提取数目列数据
    if len(df.columns) < 2:
        print("错误: Excel文件需要至少两列数据")
        return

    # 第一列是序号，第二列是数目
    serial_col = df.columns[0]
    count_col = df.columns[1]

    print(f"序号列: {serial_col}")
    print(f"数目列: {count_col}")

    # 提取数目数据
    count_data = df[count_col].values
    n = len(count_data)
    print(f"总数据点数: {n}")

    # 3. 计算不同分组大小的标准误差
    max_group_size = min(n // 2, 100)  # 最大分组大小为n//2或50中的较小值
    group_sizes = list(range(2, max_group_size + 1))
    standard_errors = []

    # 用于记录组大小≥10时每个组数对应的候选分组大小
    group_count_candidates = {}

    print("\n正在计算标准误差...")
    print("分组大小\t时间(ns)\t标准误差\t总组数\t完整组数\t剩余数据\t分组类型")
    print("-" * 90)

    # 第一遍：收集所有可能的分组
    for group_size in group_sizes:
        # 计算分组信息
        num_full_groups = n // group_size
        remainder = n % group_size

        # 确定分组类型和是否计算
        group_type = ""
        calculate = False

        if group_size < 10:
            group_type = "小分组(2-9)"
            # 组大小2-9的分组总是计算，包含剩余数据
            calculate = True
            include_remainder = True
        else:
            # 组大小≥10的情况
            # 总组数 = 完整组数 + (1 if remainder > 0 else 0)
            total_groups = num_full_groups + (1 if remainder > 0 else 0)

            # 只考虑总组数≥3的情况
            if total_groups >= 3:
                if total_groups not in group_count_candidates:
                    group_count_candidates[total_groups] = []

                group_count_candidates[total_groups].append({
                    'size': group_size,
                    'remainder': remainder,
                    'divisible': (remainder == 0)
                })
                group_type = f"候选{total_groups}组"
            else:
                group_type = f"跳过(组数<3)"

        # 对于小分组，直接计算标准误差
        if calculate:
            se = calculate_standard_error(count_data, group_size, include_remainder)
            standard_errors.append(se)

            if not np.isnan(se):
                total_groups_calc = num_full_groups + (1 if remainder > 0 and include_remainder else 0)
                print(
                    f"{group_size}\t\t{group_size * 0.2:.1f}\t\t{se:.6f}\t{total_groups_calc}\t{num_full_groups}\t\t{remainder}\t\t{group_type}")
            else:
                total_groups_calc = num_full_groups + (1 if remainder > 0 and include_remainder else 0)
                print(
                    f"{group_size}\t\t{group_size * 0.2:.1f}\t\t无法计算\t{total_groups_calc}\t{num_full_groups}\t\t{remainder}\t\t{group_type}")
        else:
            standard_errors.append(np.nan)
            total_groups_calc = num_full_groups + (1 if remainder > 0 else 0)
            print(
                f"{group_size}\t\t{group_size * 0.2:.1f}\t\t跳过\t{total_groups_calc}\t{num_full_groups}\t\t{remainder}\t\t{group_type}")

    # 第二遍：处理组大小≥10的情况，选择每个组数下的最优分组
    print("\n处理组大小≥10的候选分组:")
    print("总组数\t候选分组大小\t选择的分组\t选择原因")
    print("-" * 60)

    selected_group_sizes = set()

    for group_count, candidates in sorted(group_count_candidates.items()):
        if not candidates:
            continue

        # 检查是否有整除的分组
        divisible_candidates = [c for c in candidates if c['divisible']]

        if divisible_candidates:
            # 选择整除的分组
            selected = divisible_candidates[0]  # 选择第一个整除的分组
            selected_group_sizes.add(selected['size'])
            reason = "整除分组"
        else:
            # 没有整除分组，选择剩余数据量最大的分组
            selected = max(candidates, key=lambda x: x['remainder'])
            selected_group_sizes.add(selected['size'])
            reason = f"剩余数据最大({selected['remainder']})"

        # 计算并记录标准误差（对于组大小≥10，总是包含剩余数据）
        se = calculate_standard_error(count_data, selected['size'], include_remainder=True)

        # 更新standard_errors列表
        index = group_sizes.index(selected['size'])
        standard_errors[index] = se

        # 打印选择信息
        candidate_sizes = [c['size'] for c in candidates]
        print(f"{group_count}\t{candidate_sizes}\t\t{selected['size']}\t\t{reason}")

        # 更新输出信息
        num_full_groups = n // selected['size']
        remainder = n % selected['size']
        if not np.isnan(se):
            print(
                f"{selected['size']}\t\t{selected['size'] * 0.2:.1f}\t\t{se:.6f}\t{group_count}\t{num_full_groups}\t\t{remainder}\t\t最优{group_count}组")
        else:
            print(
                f"{selected['size']}\t\t{selected['size'] * 0.2:.1f}\t\t无法计算\t{group_count}\t{num_full_groups}\t\t{remainder}\t\t最优{group_count}组")

    # 4. 数据可视化
    plt.figure(figsize=(12, 8))

    # 过滤掉NaN值
    valid_indices = [i for i, se in enumerate(standard_errors) if not np.isnan(se)]
    valid_sizes = [group_sizes[i] for i in valid_indices]
    valid_errors = [standard_errors[i] for i in valid_indices]

    # 将横坐标乘以0.2，单位改为ns
    valid_sizes_ns = [size * 0.2 for size in valid_sizes]

    if len(valid_sizes_ns) == 0:
        print("错误: 没有有效的标准误差数据可绘制")
        return

    # 绘制散点图
    plt.scatter(valid_sizes_ns, valid_errors, color='red', s=50, zorder=5, label='数据点')

    # 创建光滑曲线（只有在有足够数据点时）
    if len(valid_sizes_ns) >= 3:
        # 使用样条插值创建光滑曲线
        x_smooth = np.linspace(min(valid_sizes_ns), max(valid_sizes_ns), 300)

        try:
            # 使用scipy的make_interp_spline创建光滑曲线
            spl = make_interp_spline(valid_sizes_ns, valid_errors)
            y_smooth = spl(x_smooth)
            plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='光滑曲线')
        except:
            # 如果样条插值失败，使用多项式拟合
            try:
                z = np.polyfit(valid_sizes_ns, valid_errors, min(3, len(valid_sizes_ns) - 1))
                p = np.poly1d(z)
                y_smooth = p(x_smooth)
                plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='拟合曲线')
            except:
                # 最后尝试直接连接点
                plt.plot(valid_sizes_ns, valid_errors, 'b-', linewidth=2, label='连接线')
    elif len(valid_sizes_ns) >= 2:
        # 如果只有2个点，直接连接
        plt.plot(valid_sizes_ns, valid_errors, 'b-', linewidth=2, label='连接线')

    # 设置图表属性
    plt.xlabel('block time (ns)', fontsize=12)
    plt.ylabel('SE', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 设置x轴刻度
    if len(valid_sizes_ns) > 0:
        # 选择一些主要刻度，避免过于密集
        if len(valid_sizes_ns) > 10:
            step = len(valid_sizes_ns) // 10
            x_ticks = [valid_sizes_ns[i] for i in range(0, len(valid_sizes_ns), step)]
        else:
            x_ticks = valid_sizes_ns
        # 添加最后三个数据点的刻度
        last_three = valid_sizes_ns[-3:] if len(valid_sizes_ns) >= 3 else valid_sizes_ns
        for x in last_three:
            if x not in x_ticks:  # 避免重复添加
                x_ticks.append(x)

                # 排序并去重
        x_ticks = sorted(set(x_ticks))

        plt.xticks(x_ticks, [f"{x:.1f}" for x in x_ticks], fontsize=10, rotation=45)
    plt.yticks(fontsize=10)

    # 添加数据标签（只添加部分标签，避免过于密集）
    if len(valid_sizes_ns) <= 20:  # 如果点不多，添加所有标签
        for i, (x, y) in enumerate(zip(valid_sizes_ns, valid_errors)):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)
    else:  # 如果点很多，只添加最小值和最大值附近的标签
        min_error = min(valid_errors)
        max_error = max(valid_errors)
        min_index = valid_errors.index(min_error)
        max_index = valid_errors.index(max_error)

        # 添加开头和结尾的几个标签
        for i in range(min(3, len(valid_sizes_ns))):
            plt.annotate(f'{valid_errors[i]:.2f}', (valid_sizes_ns[i], valid_errors[i]),
                         textcoords="offset points", xytext=(-10, 0), ha='center', fontsize=8)
        for i in range(max(0, len(valid_sizes_ns) - 3), len(valid_sizes_ns)):
            plt.annotate(f'{valid_errors[i]:.2f}', (valid_sizes_ns[i], valid_errors[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()

    # 5. 保存结果
    # 为每个分组大小计算详细信息
    group_info = []
    for group_size in group_sizes:
        num_full_groups = n // group_size
        remainder = n % group_size

        # 确定分组类型
        group_type = ""
        if group_size < 10:
            group_type = "小分组(2-9)"
        else:
            total_groups = num_full_groups + (1 if remainder > 0 else 0)
            if total_groups >= 3:
                if group_size in selected_group_sizes:
                    group_type = f"最优{total_groups}组"
                else:
                    group_type = "跳过"
            else:
                group_type = "跳过(组数<3)"

        # 计算标准误差
        if group_type in ["小分组(2-9)"] or group_type.startswith("最优"):
            include_remainder = True  # 对于小分组和选中的大分组，总是包含剩余数据
            se = calculate_standard_error(count_data, group_size, include_remainder)
        else:
            se = np.nan

        group_info.append({
            '分组大小': group_size,
            '时间(ns)': group_size * 0.2,
            '标准误差': se,
            '总组数': num_full_groups + (1 if remainder > 0 and include_remainder else 0),
            '完整组数': num_full_groups,
            '剩余数据量': remainder,
            '分组类型': group_type
        })

    result_df = pd.DataFrame(group_info)

    # 保存到Excel
    excel_output_path = r"C:\Users\123456\Desktop\pachong\co2渗透数目标准误差.xlsx"
    result_df.to_excel(excel_output_path, index=False)
    print(f"\n结果已保存到 '{excel_output_path}'")

    # 保存图表
    chart_output_path = r"C:\Users\123456\Desktop\pachong\co2渗透数目.jpg"
    plt.savefig(chart_output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到 '{chart_output_path}'")

    plt.show()

    # 6. 打印统计信息
    print("\n统计信息:")
    print(f"总数据量: {n}")
    print(f"有效计算的分组大小: {len(valid_sizes_ns)}")
    if len(valid_errors) > 0:
        min_error = min(valid_errors)
        max_error = max(valid_errors)
        min_index = valid_errors.index(min_error)
        max_index = valid_errors.index(max_error)
        print(
            f"最小标准误差: {min_error:.6f} (时间: {valid_sizes_ns[min_index]:.1f} ns, 分组大小: {valid_sizes[min_index]})")
        print(
            f"最大标准误差: {max_error:.6f} (时间: {valid_sizes_ns[max_index]:.1f} ns, 分组大小: {valid_sizes[max_index]})")

        # 找出最优分组（标准误差最小的分组）
        optimal_index = valid_errors.index(min_error)
        optimal_group_size = valid_sizes[optimal_index]
        optimal_time = valid_sizes_ns[optimal_index]

        # 计算最优分组的具体信息
        num_full_groups = n // optimal_group_size
        remainder = n % optimal_group_size
        total_groups = num_full_groups + (1 if remainder > 0 else 0)

        print(f"推荐分组: {optimal_group_size} 个数据点一组")
        print(f"对应时间: {optimal_time:.1f} ns")
        print(f"总组数: {total_groups} 组 ({num_full_groups}完整组 + {remainder if remainder > 0 else 0}剩余数据)")
        print(f"标准误差: {min_error:.6f}")


if __name__ == "__main__":
    main()