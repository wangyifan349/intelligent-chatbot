"""
名称：简易命令行记账本（Python/JSON版）
功能说明：
本程序是一个供个人使用的本地命令行记账工具。
它支持添加账目、查看账单、修改账目、删除账目、实现总体统计、查看收入和支出类别排行以及按月份统计收支。
所有账目信息存储于同目录下的账本.json文件中（JSON格式），无需依赖数据库。支出为负数金额，收入为正数金额。
原理描述
1. 数据以列表形式存储为JSON对象，每笔账目用字典（含日期、类别、金额、备注）表达。
2. 所有操作（增删改查统计）都通过读取整个JSON文件到内存并写回文件实现，无需数据库。
3. 支持用户在交互式命令行输入指令，按菜单提示操作。
4. 统计模块遍历所有账目信息，计算总收入、总支出、最大/最小金额、类别排名以及每月收支等结果直接输出。
适用范围：
本程序适用于需要本地管理个人账目的用户，可以直接运行，无额外依赖。
"""
import json
import os
from datetime import datetime
DATABASE_FILE = '账本.json'
def load_data():
    if not os.path.exists(DATABASE_FILE):
        return []  # 文件不存在时，返回空账单数据
    with open(DATABASE_FILE, 'r', encoding='utf-8') as file:
        try:
            return json.load(file)  # 正常读取json数据
        except:
            return []  # 文件损坏或空时返回空数据

def save_data(bill_data_list):
    with open(DATABASE_FILE, 'w', encoding='utf-8') as file:
        json.dump(bill_data_list, file, ensure_ascii=False, indent=2)  # 保存账单

def add_entry():
    date = input("日期 (YYYY-MM-DD，留空为今天): ").strip()
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')  # 默认使用今天日期
    category = input("类别: ").strip()
    amount = float(input("金额 (收入正，支出负): ").strip())
    note = input("备注: ").strip()
    bill_record = {"日期": date, "类别": category, "金额": amount, "备注": note}
    bill_data_list = load_data()
    bill_data_list.append(bill_record)  # 添加账单
    save_data(bill_data_list)
    print("添加成功！")

def view_entries():
    bill_data_list = load_data()
    if not bill_data_list:
        print("暂无账目。")
        return
    print("{:<4} {:<12} {:<10} {:>10}   {}".format("序号", "日期", "类别", "金额", "备注"))
    print('-'*50)
    for index, bill_record in enumerate(bill_data_list):
        print("{:<4} {:<12} {:<10} {:>10.2f}   {}".format(
            index+1, bill_record['日期'], bill_record['类别'], bill_record['金额'], bill_record['备注']))  # 输出每条账单

def modify_entry():
    bill_data_list = load_data()
    if not bill_data_list:
        print("暂无账目。")
        return
    view_entries()
    record_number = input("请选择要修改的序号（空取消）: ").strip()
    if not record_number.isdigit():
        print("取消。")
        return
    record_index = int(record_number) - 1
    if 0 <= record_index < len(bill_data_list):
        bill_record = bill_data_list[record_index]
        print("当前：", bill_record)
        date = input(f"新日期[{bill_record['日期']}]: ").strip() or bill_record["日期"]
        category = input(f"新类别[{bill_record['类别']}]: ").strip() or bill_record["类别"]
        amount_input = input(f"新金额[{bill_record['金额']}]: ").strip()
        amount = float(amount_input) if amount_input else bill_record["金额"]
        note = input(f"新备注[{bill_record['备注']}]: ").strip() or bill_record["备注"]
        bill_data_list[record_index] = {"日期": date, "类别": category, "金额": amount, "备注": note}
        save_data(bill_data_list)
        print("修改完成！")
    else:
        print("无效序号。")

def delete_entry():
    bill_data_list = load_data()
    if not bill_data_list:
        print("暂无账目。")
        return
    view_entries()
    record_number = input("请选择要删除的序号（空取消）: ").strip()
    if not record_number.isdigit():
        print("取消。")
        return
    record_index = int(record_number)-1
    if 0 <= record_index < len(bill_data_list):
        print("已删除：", bill_data_list.pop(record_index))  # 删除所选账单
        save_data(bill_data_list)
    else:
        print("无效序号。")

def base_statistics():
    bill_data_list = load_data()
    if not bill_data_list:
        print("暂无数据。")
        return
    total_income = sum(bill_record["金额"] for bill_record in bill_data_list if bill_record["金额"] > 0)
    total_expense = sum(abs(bill_record["金额"]) for bill_record in bill_data_list if bill_record["金额"] < 0)
    net_total = total_income - total_expense
    print("总收入：%.2f  总支出：%.2f  结余：%.2f" % (total_income, total_expense, net_total))
    
    if total_income + total_expense > 0:
        print("支出占比：%.1f%%   收入占比：%.1f%%" % 
              (total_expense / (total_income + total_expense) * 100, 
               total_income / (total_income + total_expense) * 100))

    expense_list = [bill_record for bill_record in bill_data_list if bill_record["金额"] < 0]
    income_list = [bill_record for bill_record in bill_data_list if bill_record["金额"] > 0]
    if expense_list:
        max_expense_record = min(expense_list, key=lambda record: record["金额"])
        print("最大一笔支出：%.2f（%s | %s | %s）" % 
              (max_expense_record["金额"], max_expense_record["日期"], max_expense_record["类别"], max_expense_record["备注"]))
    if income_list:
        max_income_record = max(income_list, key=lambda record: record["金额"])
        print("最大一笔收入：%.2f（%s | %s | %s）" % 
              (max_income_record["金额"], max_income_record["日期"], max_income_record["类别"], max_income_record["备注"]))

    if bill_data_list:
        date_set = sorted(set(bill_record["日期"] for bill_record in bill_data_list))
        print("记账天数：", len(date_set))
        average_expense = total_expense / len(date_set)
        average_income = total_income / len(date_set)
        print("平均每日收入：%.2f，平均每日支出：%.2f" % (average_income, average_expense))

def category_rank():
    bill_data_list = load_data()
    if not bill_data_list:
        print("暂无记录。")
        return
    category_expense_total = {}
    category_income_total = {}
    for bill_record in bill_data_list:
        if bill_record["金额"] < 0:
            category_expense_total[bill_record["类别"]] = category_expense_total.get(bill_record["类别"], 0) + abs(bill_record["金额"])
        elif bill_record["金额"] > 0:
            category_income_total[bill_record["类别"]] = category_income_total.get(bill_record["类别"], 0) + bill_record["金额"]
    print("\n最高支出类别排行：")
    for category, amount_sum in sorted(category_expense_total.items(), key=lambda item: -item[1])[:5]:
        print("%-8s %.2f" % (category, amount_sum))
    print("\n最高收入类别排行：")
    for category, amount_sum in sorted(category_income_total.items(), key=lambda item: -item[1])[:5]:
        print("%-8s %.2f" % (category, amount_sum))

def month_stat():
    bill_data_list = load_data()
    if not bill_data_list:
        print("暂无记录。")
        return
    month_record_dict = {}
    for bill_record in bill_data_list:
        month_string = bill_record["日期"][:7]  # 获取年月
        month_record_dict.setdefault(month_string, []).append(bill_record)
    for month_string, record_list in sorted(month_record_dict.items()):
        income_sum = sum(bill_record["金额"] for bill_record in record_list if bill_record["金额"] > 0)
        expense_sum = sum(-bill_record["金额"] for bill_record in record_list if bill_record["金额"] < 0)
        print(f"{month_string}: 收入{income_sum:.2f} 支出{expense_sum:.2f} 结余{income_sum-expense_sum:.2f}")

def main():
    while True:
        print('''
================= 记账本 ================
1. 添加账目        2. 查看账目
3. 修改账目        4. 删除账目
5. 总体统计        6. 类别排行
7. 月份统计        0. 退出
=========================================
''')
        user_choice = input("请选择功能: ").strip()
        if user_choice == '1':
            add_entry()
        elif user_choice == '2':
            view_entries()
        elif user_choice == '3':
            modify_entry()
        elif user_choice == '4':
            delete_entry()
        elif user_choice == '5':
            base_statistics()
        elif user_choice == '6':
            category_rank()
        elif user_choice == '7':
            month_stat()
        elif user_choice == '0':
            print("谢谢使用")
            break
        else:
            print("输入无效。")
if __name__ == '__main__':
    main()

====================================================================

如需进一步完善或扩展功能，可以随时继续提问。
