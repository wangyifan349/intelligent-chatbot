#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
file_organizer_cli.py

功能说明：
1. 这是一个交互式文件整理工具，不需要命令行参数，运行后根据菜单提示输入即可。
2. 支持输入多个源目录，以及一个目标目录。
3. 支持两种整理模式：
   - 复制模式：所有匹配到的文件都会复制到目标目录。
   - 移动模式：普通文件会移动到目标目录，但 Python 相关文件永远只复制，不会移动。
4. 支持演练模式：
   - 演练模式开启时，只打印计划执行的复制 / 移动动作，不会真正修改文件。
   - 建议第一次运行时开启演练模式，确认无误后再关闭演练模式执行。
5. 支持自动分类：
   - Images：图片文件
   - Audio：音频文件
   - Video：视频文件
   - Office：Office、PDF、文本、CSV 等文档文件
   - Python：.py、.pyw、.ipynb 文件
   - Others：未匹配到上述类型的其他文件
6. 支持自适应重命名：
   - 如果目标文件不存在，则直接使用原文件名。
   - 如果目标文件已存在且内容相同，则跳过，避免重复整理。
   - 如果目标文件已存在但内容不同，则在文件名后追加 SHA256 哈希前缀，避免覆盖。
7. 支持保留源目录结构：
   - 开启后，目标路径会包含源目录名和源文件相对路径。
   - 这样可以降低不同源目录下同名文件集中到同一目录造成冲突的概率。
8. 支持 Windows 和 Linux：
   - Windows 示例路径：D:\\Downloads
   - Linux 示例路径：/home/user/Downloads
9. 安全规则：
   - 永远不会覆盖已有文件。
   - Python 文件在用户选择移动模式时也只复制，避免误删源码。
   - 如果目标目录位于源目录内部，会自动跳过目标目录，避免递归整理自身。
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path
FILE_TYPES = {
    "Images": {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
        ".heic", ".heif", ".raw", ".cr2", ".nef", ".arw", ".dng",
    },
    "Audio": {
        ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma", ".aiff",
        ".alac", ".opus",
    },
    "Video": {
        ".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
        ".mpeg", ".mpg", ".3gp",
    },
    "Office": {
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".pdf", ".txt", ".rtf", ".csv", ".odt", ".ods", ".odp",
    },
    "Python": {
        ".py", ".pyw", ".ipynb",
    },
}

PYTHON_EXTENSIONS = FILE_TYPES["Python"]  # Python 文件扩展名集合，用于强制复制保护规则。
def get_file_hash(file_path: Path) -> str:
    """计算文件的 SHA256 哈希值，用于判断两个同名文件内容是否完全一致。"""
    sha256 = hashlib.sha256()  # 创建 SHA256 哈希对象，用于逐块计算文件摘要。
    with file_path.open("rb") as file_object:  # 以二进制方式读取文件，避免文本编码影响哈希结果。
        while True:
            chunk = file_object.read(1024 * 1024)  # 每次读取 1MB，避免大文件一次性读入内存。
            if not chunk:  # 读到空内容表示文件已经读取完毕。
                break
            sha256.update(chunk)  # 将当前文件块加入哈希计算。

    return sha256.hexdigest()  # 返回十六进制字符串形式的 SHA256 值。
def get_category(file_path: Path) -> str:
    """根据文件扩展名判断文件分类。"""
    file_extension = file_path.suffix.lower()  # 获取文件后缀并转成小写，兼容 .JPG、.Py 等大小写混用情况。

    for category_name, extensions in FILE_TYPES.items():  # 遍历预设分类规则，查找当前后缀属于哪一类。
        if file_extension in extensions:  # 如果当前文件后缀命中了某个分类的扩展名集合。
            return category_name  # 返回对应分类名，用于创建目标分类目录。

    return "Others"  # 没有命中任何分类时，统一归入 Others。


def ask_yes_no(message: str, default_yes: bool = False) -> bool:
    """交互式询问是否，支持 y/n、yes/no、中文和 1/0。"""
    default_text = "Y/n" if default_yes else "y/N"  # 根据默认值显示提示，Y 表示直接回车时默认选是。

    while True:
        answer = input(f"{message} [{default_text}]：").strip().lower()  # 读取用户输入并标准化，方便后续判断。

        if not answer:  # 用户直接回车时，使用默认选项。
            return default_yes

        if answer in {"y", "yes", "是", "对", "好", "1"}:  # 支持英文、中文和数字形式的肯定回答。
            return True

        if answer in {"n", "no", "否", "不", "0"}:  # 支持英文、中文和数字形式的否定回答。
            return False

        print("[WARN] 请输入 y 或 n。")  # 输入无法识别时提示用户重新输入。


def main() -> None:
    print("\n========== 文件整理工具 ==========")
    print("功能：多个源目录 -> 一个目标目录")
    print("支持：复制 / 移动 / 演练模式 / 自适应重命名")
    print("特殊规则：Python 文件永远只复制，不移动")
    print("系统支持：Windows / Linux\n")

    print("分类规则：")
    for category_name, extensions in FILE_TYPES.items():  # 打印当前内置的分类规则，方便用户运行前确认。
        print(f"  - {category_name}: {', '.join(sorted(extensions))}")
    print("  - Others: 未匹配到以上类型的其他文件\n")

    source_directories = []  # 保存用户输入的多个源目录，后续会逐个扫描。

    print("请输入源目录，每行一个。输入空行结束。")
    print("Windows 示例：D:\\Downloads")
    print("Linux 示例：/home/user/Downloads\n")

    while True:
        raw_source_path = input(f"源目录 {len(source_directories) + 1}：").strip().strip('"').strip("'")  # 允许用户拖拽路径到终端，自动去掉首尾引号。

        if not raw_source_path:  # 空输入表示当前源目录输入结束。
            if source_directories:  # 至少已经添加过一个源目录时，才允许结束输入。
                break
            print("[WARN] 至少需要输入一个源目录。")
            continue

        source_directory = Path(raw_source_path).expanduser()  # 将用户输入转换为 Path，并支持 Linux 下的 ~ 用户目录。

        if not source_directory.exists():  # 检查源目录是否存在，避免后续扫描时报错。
            print(f"[WARN] 源目录不存在：{source_directory}")
            if not ask_yes_no("是否仍然添加这个目录", default_yes=False):  # 允许用户保留暂时不存在的路径，但默认不添加。
                continue

        elif not source_directory.is_dir():  # 源路径必须是目录，普通文件不作为源目录处理。
            print(f"[WARN] 这不是目录，已跳过：{source_directory}")
            continue

        if source_directory in source_directories:  # 避免同一个源目录重复添加，导致文件重复扫描。
            print(f"[WARN] 该源目录已经添加过：{source_directory}")
            continue

        source_directories.append(source_directory)  # 将有效源目录加入待扫描列表。
        print(f"[OK] 已添加源目录：{source_directory}")

    raw_target_path = input("\n请输入目标目录：").strip().strip('"').strip("'")  # 读取目标目录，同样去除可能存在的引号。
    while not raw_target_path:  # 目标目录是必须项，不能为空。
        print("[WARN] 目标目录不能为空。")
        raw_target_path = input("请输入目标目录：").strip().strip('"').strip("'")

    target_directory = Path(raw_target_path).expanduser()  # 将目标目录转换为 Path，并支持 ~ 用户目录。

    print("\n请选择操作模式：")
    print("  1. 复制文件，保留源文件")
    print("  2. 移动文件，但 Python 文件永远只复制")

    while True:
        operation_choice = input("请选择 1 或 2：").strip()  # 读取用户选择的操作模式。
        if operation_choice == "1":
            requested_operation = "copy"  # 用户选择复制模式，所有文件都复制。
            break
        if operation_choice == "2":
            requested_operation = "move"  # 用户选择移动模式，但 Python 文件后续会被强制改为复制。
            break
        print("[WARN] 无效选择，请输入 1 或 2。")

    dry_run_enabled = ask_yes_no("\n是否开启演练模式，不实际复制或移动", default_yes=True)  # 默认开启演练模式，避免误操作。

    preserve_source_tree = ask_yes_no(
        "是否保留源目录结构，降低同名文件聚集风险",
        default_yes=True,
    )  # 默认保留源目录结构，降低来自不同目录的同名文件冲突概率。

    print("\n========== 配置确认 ==========")
    print("源目录：")
    for source_directory in source_directories:  # 展示所有源目录，方便用户在真正执行前核对。
        print(f"  - {source_directory}")

    print(f"目标目录：{target_directory}")
    print(f"用户选择操作：{'复制' if requested_operation == 'copy' else '移动'}")
    print(f"演练模式：{'开启' if dry_run_enabled else '关闭'}")
    print(f"保留源目录结构：{'开启' if preserve_source_tree else '关闭'}")
    print("固定规则：Python / pyw / ipynb 文件永远只复制")
    print("重名规则：不覆盖；相同内容跳过；不同内容自动改名")

    if not ask_yes_no("\n确认开始执行", default_yes=False):  # 最终确认，默认不执行，进一步降低误操作风险。
        print("[WARN] 用户取消，程序退出。")
        return
    if dry_run_enabled:  # 演练模式下只打印计划动作，不会真正复制或移动。
        print("\n[WARN] 当前是演练模式，不会真正复制、移动或删除任何文件。")
    target_directory.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，不存在则自动创建。
    scanned_count = 0  # 扫描到的文件总数。
    copied_count = 0  # 计划复制或实际复制的文件数。
    moved_count = 0  # 计划移动或实际移动的文件数。
    skipped_duplicate_count = 0  # 因目标已有相同内容文件而跳过的数量。
    renamed_count = 0  # 因重名且内容不同而自动改名的数量。
    python_forced_copy_count = 0  # 用户选择移动时，Python 文件被强制复制的数量。
    error_count = 0  # 处理过程中发生异常的文件数量。

    print("\n========== 开始扫描和处理 ==========")

    resolved_target_directory = target_directory.resolve()  # 获取目标目录绝对路径，用于判断是否需要跳过目标目录自身。

    for source_directory in source_directories:  # 逐个扫描用户输入的多个源目录。
        try:
            resolved_source_directory = source_directory.resolve()  # 获取源目录绝对路径，方便计算相对路径和避免路径歧义。
        except OSError:
            resolved_source_directory = source_directory.absolute()  # 如果 resolve 失败，则退回到 absolute，增强兼容性。
        print(f"\n[INFO] 开始扫描源目录：{resolved_source_directory}")
        if not resolved_source_directory.exists():  # 二次检查源目录是否存在，处理用户之前强制添加不存在路径的情况。
            print(f"[WARN] 源目录不存在，跳过：{resolved_source_directory}")
            continue
        if not resolved_source_directory.is_dir():  # 二次确认源路径是目录。
            print(f"[WARN] 源路径不是目录，跳过：{resolved_source_directory}")
            continue
        for current_root, child_directory_names, file_names in os.walk(resolved_source_directory):  # 递归遍历源目录下的所有子目录和文件。
            current_root_path = Path(current_root)  # 将当前扫描目录转换为 Path，方便后续路径运算。
            try:
                resolved_current_root = current_root_path.resolve()  # 当前扫描目录的绝对路径。
            except OSError:
                resolved_current_root = current_root_path.absolute()  # resolve 失败时使用 absolute 兜底。
            try:
                resolved_current_root.relative_to(resolved_target_directory)  # 判断当前目录是否位于目标目录内部。
                print(f"[WARN] 跳过目标目录自身，避免递归整理：{resolved_current_root}")
                child_directory_names[:] = []  # 清空子目录列表，阻止 os.walk 继续深入目标目录。
                continue
            except ValueError:
                pass  # 当前目录不在目标目录内部，可以正常处理。
            filtered_child_directory_names = []  # 用于保存允许继续递归扫描的子目录名。
            for child_directory_name in child_directory_names:  # 检查每一个子目录，避免进入目标目录。
                child_directory_path = resolved_current_root / child_directory_name  # 拼出子目录完整路径。

                try:
                    child_directory_path.resolve().relative_to(resolved_target_directory)  # 判断子目录是否属于目标目录。
                    print(f"[WARN] 跳过目标子目录：{child_directory_path}")
                    continue
                except ValueError:
                    filtered_child_directory_names.append(child_directory_name)  # 子目录不在目标目录内，保留给 os.walk 继续递归。
                except OSError:
                    filtered_child_directory_names.append(child_directory_name)  # 路径解析失败时保守处理，继续扫描该子目录。
            child_directory_names[:] = filtered_child_directory_names  # 原地修改 os.walk 的目录列表，真正实现跳过目标目录。
            for file_name in file_names:  # 遍历当前目录下的所有文件名。
                source_file_path = resolved_current_root / file_name  # 拼出源文件完整路径。
                scanned_count += 1  # 每遇到一个文件，扫描计数加一。
                try:
                    category_name = get_category(source_file_path)  # 根据扩展名获取文件分类。
                    source_extension = source_file_path.suffix.lower()  # 获取小写扩展名，用于判断是否是 Python 文件。
                    is_python_source = source_extension in PYTHON_EXTENSIONS  # Python 文件保护判断，命中则永远只复制。

                    if preserve_source_tree:  # 如果用户选择保留源目录结构。
                        try:
                            relative_parent = source_file_path.parent.resolve().relative_to(
                                resolved_source_directory
                            )  # 计算源文件父目录相对于源目录根目录的路径。
                            target_parent_directory = (
                                target_directory
                                / category_name
                                / resolved_source_directory.name
                                / relative_parent
                            )  # 目标目录包含分类名、源目录名和相对路径，便于追踪来源。
                        except ValueError:
                            target_parent_directory = target_directory / category_name  # 无法计算相对路径时退回到分类目录。
                    else:
                        target_parent_directory = target_directory / category_name  # 不保留源结构时，直接放入对应分类目录。

                    target_parent_directory.mkdir(parents=True, exist_ok=True)  # 确保当前文件对应的目标父目录存在。

                    destination_file_path = target_parent_directory / source_file_path.name  # 默认目标文件名保持和源文件一致。
                    destination_status = "new"  # 默认认为目标文件是新文件，没有发生重名。

                    if destination_file_path.exists():  # 如果目标位置已经存在同名文件，需要判断是重复还是冲突。
                        same_content = False  # 默认认为不是相同内容，只有大小和哈希都一致才改为 True。

                        try:
                            if source_file_path.stat().st_size == destination_file_path.stat().st_size:  # 先比较文件大小，大小不同则无需计算哈希。
                                same_content = (
                                    get_file_hash(source_file_path)
                                    == get_file_hash(destination_file_path)
                                )  # 大小相同再比较 SHA256，确认是否为完全相同的文件。
                        except OSError as exception:
                            print(f"[WARN] 比较文件失败，将按不同文件处理：{exception}")

                        if same_content:  # 同名且内容完全一致，说明目标已有重复文件。
                            skipped_duplicate_count += 1  # 重复跳过计数加一。
                            print()
                            print(f"[INFO] 扫描到文件：{source_file_path}")
                            print(f"[INFO] 文件分类：{category_name}")
                            print(f"[WARN] 目标位置已存在相同内容文件，跳过：{destination_file_path}")
                            continue

                        file_stem = source_file_path.stem  # 原文件名主体，不包含扩展名。
                        file_suffix = source_file_path.suffix  # 原文件扩展名，保留大小写形式。

                        try:
                            hash_prefix = get_file_hash(source_file_path)[:10]  # 取源文件哈希前 10 位作为重命名后缀，降低冲突概率。
                        except OSError:
                            hash_prefix = "unknown"  # 如果哈希计算失败，使用 unknown 作为兜底后缀。

                        destination_file_path = target_parent_directory / f"{file_stem}_{hash_prefix}{file_suffix}"  # 第一次自适应重命名尝试。
                        destination_status = "renamed"  # 标记发生了重名改名。
                        rename_index = 1  # 如果哈希后缀文件名仍然存在，则继续追加递增编号。
                        while destination_file_path.exists():  # 只要候选目标路径仍然存在，就继续处理冲突。
                            same_content = False  # 每轮都重新判断当前候选目标是否与源文件相同。
                            try:
                                if source_file_path.stat().st_size == destination_file_path.stat().st_size:  # 先比较大小以减少哈希计算成本。
                                    same_content = (
                                        get_file_hash(source_file_path)
                                        == get_file_hash(destination_file_path)
                                    )  # 大小相同时再用哈希确认是否完全一致。
                            except OSError:
                                same_content = False  # 读取失败时按不同文件处理，继续寻找新名字。

                            if same_content:  # 如果某个已存在的重命名文件内容相同，也可以直接跳过。
                                skipped_duplicate_count += 1
                                print()
                                print(f"[INFO] 扫描到文件：{source_file_path}")
                                print(f"[INFO] 文件分类：{category_name}")
                                print(f"[WARN] 已存在相同内容的重命名文件，跳过：{destination_file_path}")
                                break
                            destination_file_path = (
                                target_parent_directory
                                / f"{file_stem}_{hash_prefix}_{rename_index:03d}{file_suffix}"
                            )  # 继续生成带递增编号的新候选路径。
                            rename_index += 1  # 递增编号，避免无限尝试同一个名字。
                        else:
                            renamed_count += 1  # while 正常结束说明最终找到了不存在的新名字，重命名计数加一。

                        if same_content:  # 如果 while 内发现重复内容并 break，则跳过当前源文件。
                            continue

                    effective_operation = requested_operation  # 默认执行用户选择的复制或移动。

                    if is_python_source:  # Python 相关文件触发保护规则。
                        effective_operation = "copy"  # 不管用户选择什么，Python 文件最终都只复制。

                    if requested_operation == "move" and is_python_source:  # 用户选择移动但当前文件是 Python 文件。
                        python_forced_copy_count += 1  # 记录 Python 强制复制次数，便于最终统计。

                    operation_text = "复制" if effective_operation == "copy" else "移动"  # 用中文描述最终实际操作，方便日志阅读。

                    print()
                    print(f"[INFO] 扫描到文件：{source_file_path}")
                    print(f"[INFO] 文件分类：{category_name}")
                    print(f"[INFO] 目标目录：{target_parent_directory}")

                    if destination_status == "renamed":
                        print(f"[WARN] 检测到重名文件，已自适应重命名：{destination_file_path.name}")
                    else:
                        print(f"[INFO] 目标文件名：{destination_file_path.name}")

                    if requested_operation == "move" and is_python_source:
                        print("[WARN] Python 文件保护规则生效：用户选择移动，但该文件强制复制。")

                    if dry_run_enabled:  # 演练模式只打印计划动作。
                        print(f"[INFO] [演练] 将会{operation_text}：")
                        print(f"[INFO]        从：{source_file_path}")
                        print(f"[INFO]        到：{destination_file_path}")
                    else:
                        print(f"[INFO] 正在{operation_text}文件...")
                        destination_file_path.parent.mkdir(parents=True, exist_ok=True)  # 再次确保目标父目录存在，防止并发或手动删除导致失败。

                        if effective_operation == "copy":  # 最终操作为复制。
                            shutil.copy2(source_file_path, destination_file_path)  # 复制文件，并尽量保留修改时间等元数据。
                        else:
                            shutil.move(str(source_file_path), str(destination_file_path))  # 移动文件，源文件会从原位置移走。

                        print(f"[OK] {operation_text}完成：{destination_file_path}")

                    if effective_operation == "copy":  # 根据最终实际操作更新统计。
                        copied_count += 1
                    else:
                        moved_count += 1

                except Exception as exception:
                    error_count += 1  # 当前文件处理失败，错误计数加一，并继续处理后续文件。
                    print()
                    print(f"[ERROR] 处理失败：{source_file_path}")
                    print(f"[ERROR] 失败原因：{exception}")

    print("\n========== 完成统计 ==========")
    print(f"扫描文件数：{scanned_count}")
    print(f"复制文件数：{copied_count}")
    print(f"移动文件数：{moved_count}")
    print(f"Python 强制复制数：{python_forced_copy_count}")
    print(f"跳过重复数：{skipped_duplicate_count}")
    print(f"自适应重命名数：{renamed_count}")
    print(f"错误数量：{error_count}")
    if dry_run_enabled:  # 最终再次提醒用户，本次没有真正修改文件。
        print()
        print("[WARN] 本次为演练模式，上述复制/移动统计表示计划执行数量，并未真正修改文件。")

    print("\n[OK] 程序执行结束。")
if __name__ == "__main__":
    try:
        main()  # 程序入口，启动交互式文件整理流程。
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断，程序退出。")
        sys.exit(130)  # 使用 130 作为 Ctrl+C 中断退出码，符合常见命令行约定。
