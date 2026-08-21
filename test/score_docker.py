"""
本脚本用于将预测出的五支股票与实际的股票数据进行对比，计算加权收益，形成最终得分。
"""
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
parser = argparse.ArgumentParser(description='Calculate stock prediction score.')
parser.add_argument('team_name', type=str, help='The name of the team (used for file naming).')
args = parser.parse_args()
output_path = f'./test/results_output/{args.team_name}.csv'
test_data_path = './data/test.csv'


def write_failed_score() -> None:
    Path('./temp').mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        {
            "Team Name": [args.team_name],
            "Final Score": [-999],
        }
    )
    result.to_csv("./temp/tmp.csv", index=False)


def is_valid_prediction(prediction_data):
    """
    验证选手输出的结果是否合法：需要包含最多五支股票，并且权重之和0到1之间.
    """
    id_col = 'stock_id' if 'stock_id' in prediction_data.columns else '股票代码' if '股票代码' in prediction_data.columns else None
    weight_col = 'weight' if 'weight' in prediction_data.columns else '权重' if '权重' in prediction_data.columns else None
    if id_col is None or weight_col is None:
        raise ValueError('预测结果缺少必要字段，必须包含 stock_id/股票代码 和 weight/权重。')

    if not 1 <= len(prediction_data) <= 5:
        raise ValueError('预测结果不合法：必须包含一到五支股票。')

    if prediction_data[id_col].astype(str).duplicated().any():
        raise ValueError('预测结果不合法：股票代码不得重复。')

    weights = pd.to_numeric(prediction_data[weight_col], errors='coerce')
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('预测结果不合法：权重必须是非负有限数值。')
    weight_sum = weights.sum()
    if not (0 <= float(weight_sum) <= 1.0):
        raise ValueError(f"预测结果不合法：权重之和必须为0到1之间. 当前权重之和为 {weight_sum}.")


def calculate_return(group):
    group = group.sort_values('日期')
    if len(group) != 5:
        raise ValueError(f"股票 {group.iloc[0]['股票代码']} 的测试记录不是5条")
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']


def calculate_predict_weight_score(output_data, test_data):
    # 选择输出指定的5个股票
    test_data = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
    # 只选最后五个记录，并显式按日期聚合，避免依赖输入文件顺序。
    test_data = test_data.sort_values(['股票代码', '日期']).groupby('股票代码').tail(5)
    result = test_data.groupby('股票代码', as_index=False).agg(
        起始开盘=('开盘', 'first'),
        结束开盘=('开盘', 'last'),
        记录数=('开盘', 'size'),
    )
    if len(result) != len(output_data) or (result['记录数'] != 5).any():
        raise ValueError('测试数据未覆盖组合中每只股票的完整5个交易日')
    result['收益率'] = (result['结束开盘'] - result['起始开盘']) / result['起始开盘']
    result = result.merge(output_data, on='股票代码')
    # 计算加权收益率
    final_score = (result['收益率'] * result['权重']).sum()
    return final_score


# 读取测试数据
try:
    test_data = pd.read_csv(test_data_path)
    raw_output_data = pd.read_csv(output_path)
    is_valid_prediction(raw_output_data)
except Exception as e:
    print(f"Error reading test data or validating prediction: {e}")
    write_failed_score()
    sys.exit(0)

test_data = test_data[['股票代码', '日期', '开盘', '收盘']]
# 读取输出数据
output_data = raw_output_data.rename(columns={'stock_id': '股票代码', 'weight': '权重'})

required_columns = {'股票代码', '权重'}
if not required_columns.issubset(output_data.columns):
    print('Error reading test data or validating prediction: 输出结果缺少股票代码或权重字段。')
    write_failed_score()
    sys.exit(0)

# 第一步：计算预测股票的加权收益率
predict_weight_score = calculate_predict_weight_score(output_data, test_data)


# 保存结果到 CSV 文件
Path('./temp').mkdir(parents=True, exist_ok=True)
result = pd.DataFrame(
    {
        "Team Name": [args.team_name],
        "Final Score": [predict_weight_score],
    }
)
result.to_csv("./temp/tmp.csv", index=False)
print(f"预测股票的加权收益率得分: {predict_weight_score}")
