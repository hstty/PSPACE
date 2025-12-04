#paperspaceのjupyter lab上のコンソールで実行するプログラム。
#ipynbとWindowsには対応しなくていい。

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import subprocess
import sys
import shutil
import argparse
import collections.abc
import copy
import zipfile
import glob

import toml
import html as _html
import signal
import time

from threading import Thread

def deep_update(d, u):
    """
    ネストされた辞書を再帰的に更新する。
    u のキーと値を d にマージする。
    値が '**delete**' の場合、そのキーを d から削除する。
    """
    for k, v in u.items():
        if v == "**delete**":
            d.pop(k, None)
        elif isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def compare_configs(original, updated, path=""):
    """
    2つの設定辞書を再帰的に比較し、変更点を文字列のリストとして返す。
    """
    changes = []
    # 追加されたキーと変更されたキーをチェック
    for key, updated_value in updated.items():
        new_path = f"{path}.{key}" if path else key
        if key not in original:
            changes.append(f"  [追加] {new_path} = {updated_value}")
        else:
            original_value = original[key]
            if isinstance(updated_value, dict) and isinstance(original_value, dict):
                changes.extend(compare_configs(original_value, updated_value, path=new_path))
            elif original_value != updated_value:
                changes.append(f"  [変更] {new_path}: {original_value} -> {updated_value}")
    
    # 削除されたキーをチェック
    for key in original.keys():
        if key not in updated:
            new_path = f"{path}.{key}" if path else key
            changes.append(f"  [削除] {new_path}")
    
    return changes

def run_command_and_stream_output(command, folder_name):
    """
    コマンドを実行し、出力をリアルタイムで表示する。
    完了後、標準出力と標準エラーの全文を返す。
    """
    print(f"実行コマンド: {command}", flush=True)
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        bufsize=1 # 行バッファリングを有効にする
    )

    stdout_lines = []
    stderr_lines = []

    # 出力表示用のロック
    print_lock = None
    try:
        from threading import Lock
        print_lock = Lock()
    except ImportError:
        pass

    # 共有ステータス (epoch情報など)
    status_info = {'epoch': ''}

    def reader(pipe, container, stream_name):
        try:
            with pipe:
                for line in iter(pipe.readline, ''):
                    # 受け取った出力はそのまま保持しておく
                    try:
                        container.append(line)
                    except Exception as e:
                        print(f"Error appending line to container: {e}", flush=True)
                        continue

                    try:
                        # ロックを取得して出力を同期
                        if print_lock:
                            print_lock.acquire()
                        
                        try:
                            stripped_line = line.strip()
                            
                            # epoch行の検出と保存 (表示はしない)
                            if stripped_line.startswith("epoch") and "/" in stripped_line:
                                status_info['epoch'] = stripped_line
                                continue # この行は表示せずにスキップ

                            # プログレスバーの行か判定 (tqdmの一般的な出力形式を想定)
                            if stripped_line.startswith("steps:") and ("it/s" in line or "s/it" in line):
                                # epoch情報があれば結合して表示
                                display_line = stripped_line
                                if status_info['epoch']:
                                    display_line = f"{status_info['epoch']} | {display_line}"
                                
                                # 行頭に戻り、内容を表示し、行末までクリアする (ANSIエスケープシーケンス)
                                # \r: 行頭に戻る
                                # \033[K: カーソル位置から行末までクリア
                                sys.stdout.write(f'\r{display_line}\033[K')
                            else:
                                # それ以外の行（ログなど）は、行頭に戻ってから出力し、改行する
                                # 前の行（プログレスバー）の残骸を消すために行末クリアを入れる
                                sys.stdout.write(f'\r{line.rstrip()}\033[K\n')
                            
                            sys.stdout.flush()
                        finally:
                            if print_lock:
                                print_lock.release()

                    except Exception:
                        try:
                            print(line, end='', flush=True)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Error in reader thread: {e}", flush=True)

    stdout_thread = Thread(target=reader, args=[process.stdout, stdout_lines, "stdout"])
    stderr_thread = Thread(target=reader, args=[process.stderr, stderr_lines, "stderr"])
    stdout_thread.start()
    stderr_thread.start()

    try:
        process.wait() # プロセスの終了を待つ
    except KeyboardInterrupt:
        # ユーザーによる中断時、サブプロセスを確実に終了させる
        print(f"\n[中断] ユーザーによる中断を検知しました。サブプロセス (PID: {process.pid}) を終了します...", flush=True)
        try:
            # まずは優しく終了
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 応答がない場合は強制終了
                print(f"[中断] サブプロセスが応答しません。強制終了 (kill) します...", flush=True)
                process.kill()
                process.wait()
        except Exception as e:
            print(f"[中断] サブプロセスの終了中にエラーが発生しました: {e}", flush=True)
        raise # 上位に例外を伝播させる
    finally:
        # 念のため、プロセスがまだ生きていれば終了させる
        if process.poll() is None:
             try:
                process.terminate()
                process.wait(timeout=1)
             except:
                try:
                    process.kill()
                except:
                    pass

    stdout_thread.join()
    stderr_thread.join()

    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines)
    )

def run_training_with_retry(command, temp_config_file, config, program_directory, folder_name):
    """
    学習コマンドを実行します。メモリ不足エラーが発生した場合、
    'outofmemory.toml' の設定を読み込んで再試行します。
    出力をリアルタイムで表示します。
    """
    print(f"[{folder_name}] の学習を開始します...", flush=True)
    
    try:
        # 初回の実行
        result = run_command_and_stream_output(command, folder_name)

        # メモリ不足エラーのキーワード
        oom_keywords = ["CUDA out of memory", "torch.cuda.OutOfMemoryError"]

        # メモリ不足エラーを検知した場合のみ再試行
        if result.returncode != 0 and any(keyword in result.stderr for keyword in oom_keywords):
            print(f"[{folder_name}] メモリ不足エラーを検出しました。'outofmemory.toml' の設定で再試行します。")
            print("----- Stderr Summary (Initial) -----")
            print("OOM Error detected. Retrying with new settings...")
            print("------------------------------------")

            oom_config_path = os.path.join(program_directory, 'outofmemory.toml')

            if not os.path.exists(oom_config_path):
                print(f"警告: 'outofmemory.toml' が '{program_directory}' に見つかりません。再試行をスキップします。")
                return result

            try:
                with open(oom_config_path, 'r', encoding='utf-8') as f:
                    oom_config = toml.load(f)
                
                # outofmemory.toml の全キーをそのまま反映する
                print(f"'outofmemory.toml' の内容で設定を更新します: {list(oom_config.keys())}", flush=True)
                
                # デバッグ用：変更前後の設定値をログ出力
                for key in oom_config.keys():
                    old_value = config.get(key, '<未設定>')
                    new_value = oom_config[key]
                    if old_value != new_value:
                        print(f"  [{key}] {old_value} -> {new_value}", flush=True)
                
                config.update(oom_config)

                # 更新したconfigで一時ファイルを再度書き込み
                with open(temp_config_file, 'w', encoding='utf-8') as f:
                    toml.dump(config, f)

                print(f"[{folder_name}] 設定を更新して学習を再実行します...", flush=True)
                # コマンドは同じものを再利用（config_fileの中身が変わっているため）
                result = run_command_and_stream_output(command, folder_name)

            except Exception as e:
                print(f"[{folder_name}] 再試行中に予期せぬエラーが発生しました: {e}")
                # resultは最初の失敗のまま返す
                return result
        
        return result

    except Exception as e:
        print(f"[{folder_name}] 学習コマンドの実行中に予期せぬエラーが発生しました: {e}")
        # ダミーの失敗resultを返す
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=str(e))



# 引数を解析
parser = argparse.ArgumentParser(description='LoRA学習スクリプト')
parser.add_argument('--add', type=str, help='追加で読み込むTOML設定ファイル')
parser.add_argument('--test', action='store_true', help='テストモード。学習完了後にフォルダを削除しません。')
args = parser.parse_args()

# ヘッダー表示
print("\n" + "="*54)
print("||" + " LoRA学習プロセスを開始します ".center(50) + "||")
print("="*54 + "\n")

# 環境設定ファイルを読み込む
env_config_file = "PSPACE_env.toml"
try:
    with open(env_config_file, 'r', encoding='utf-8') as f:
        env_config = toml.load(f)
except FileNotFoundError:
    print(f"エラー: 環境設定ファイル '{env_config_file}' が見つかりません。")
    sys.exit(1)

print("[1] 環境設定の読み込み")
print(f"- メイン環境設定: {env_config_file}")

paths = env_config.get('paths', {})
acc_opts = env_config.get('accelerate_options', {})
train_opts = env_config.get('train_options', {})
makelora_settings = env_config.get('makelora_settings', {})

output_suffix = makelora_settings.get('output_suffix')
train_config_file = makelora_settings.get('train_config_file')

if not output_suffix or not train_config_file:
    print(f"エラー: '{env_config_file}' に 'output_suffix' と 'train_config_file' を設定してください。")
    sys.exit(1)

# --addが指定された場合、output_suffixを変更
if args.add:
    add_filename_without_ext = os.path.splitext(os.path.basename(args.add))[0]
    new_output_suffix = f"{output_suffix}-{add_filename_without_ext}"
    print(f"-> 'output_suffix' を '{output_suffix}' -> '{new_output_suffix}' に変更しました。")
    output_suffix = new_output_suffix

print(f"- 基本学習設定: {train_config_file}")
if args.add:
    print(f"- 追加学習設定: {args.add}")
print("-" * 20, flush=True)

# パス設定
base_directory = paths.get('base_directory', '/notebooks')
working_directory = os.path.join(base_directory, paths.get('working_directory', 'training'))
kohya_directory = paths.get('kohya_directory', '/kohya_ss')
program_directory = os.path.join(base_directory, paths.get('program_directory', 'program'))
temp_directory = paths.get('temp_directory', '/tmp')
accelerate_path = paths.get('accelerate_path', '/venv/bin/accelerate')
train_script_path = paths.get('train_script_path', '/kohya_ss/sd-scripts/sdxl_train_network.py')

# Google Drive のゴミ箱を空にする
try:
    rclone_config = env_config.get('rclone', {})
    # PSPACE_env.toml の [rclone] セクションで empty_trash_on_start = false が指定されていない限り実行
    if rclone_config.get('empty_trash_on_start', True):
        print("\n[+] リモートのゴミ箱を空にします")
        rclone_config_path = os.path.join(program_directory, 'rclone.conf')
        remote_path = rclone_config.get('remote_path', 'google:lora')
        try:
            remote_name = remote_path.split(':', 1)[0]
        except Exception:
            remote_name = remote_path

        cleanup_cmd = f"rclone --config {rclone_config_path} cleanup {remote_name}:"
        print(f"- 実行コマンド: rclone cleanup {remote_name}:")
        cleanup_result = subprocess.run(cleanup_cmd, shell=True, capture_output=True, text=True)

        if cleanup_result.returncode == 0:
            print("- ゴミ箱を空にしました。")
        else:
            print(f"- 警告: ゴミ箱を空にする際にエラーが発生しました。\n  stdout: {cleanup_result.stdout}\n  stderr: {cleanup_result.stderr}")
except Exception as e:
    print(f"- エラー: rclone cleanup 処理中に予期しないエラーが発生しました: {e}")
print("-" * 20, flush=True)

# rcloneでリモートからファイルをダウンロード
print("\n[2] リモートからファイルをダウンロード")
try:
    rclone_config_path = os.path.join(program_directory, 'rclone.conf')
    remote_path = env_config.get('rclone', {}).get('remote_path', 'google:lora')
    remote_training_path = f"{remote_path.rstrip('/')}/training"
    
    print(f"- リモートパス: {remote_training_path}")
    print(f"- ダウンロード先: {working_directory}")
    
    # rclone copy コマンドでファイルをダウンロード
    download_command = f"rclone --config {rclone_config_path} copy {remote_training_path} {working_directory}"
    print(f"- 実行コマンド: {download_command}")
    download_result = subprocess.run(download_command, shell=True, capture_output=True, text=True)
    
    if download_result.returncode == 0:
        print("- ダウンロードが正常に完了しました。")
        
        # ダウンロード成功後、リモートの中身を全て削除（trainingフォルダ自体は残す）
        print(f"- リモートの中身を削除します（{remote_training_path}フォルダは残します）")
        
        # ステップ1: 全てのファイルを削除
        delete_command = f"rclone --config {rclone_config_path} delete {remote_training_path}"
        delete_result = subprocess.run(delete_command, shell=True, capture_output=True, text=True)
        
        if delete_result.returncode == 0:
            print("  - ファイルを削除しました。")
        else:
            print(f"  - 警告: ファイル削除に失敗しました。")
            print(f"    stdout: {delete_result.stdout}")
            print(f"    stderr: {delete_result.stderr}")
        
        # ステップ2: 空のサブディレクトリを削除
        # rmdirs は親フォルダも削除する可能性があるため、実行後にmkdirで再作成してフォルダ維持を保証する
        rmdirs_command = f"rclone --config {rclone_config_path} rmdirs {remote_training_path}"
        rmdirs_result = subprocess.run(rmdirs_command, shell=True, capture_output=True, text=True)
        
        if rmdirs_result.returncode == 0:
            print("  - サブディレクトリを削除しました。")
        else:
            print(f"  - 警告: サブディレクトリ削除に失敗しました。")
            print(f"    stdout: {rmdirs_result.stdout}")
            print(f"    stderr: {rmdirs_result.stderr}")

        # ステップ3: trainingフォルダ自体を再作成（存在保証）
        # これにより、rmdirsでtrainingフォルダごと消えてしまっても復活させる
        mkdir_command = f"rclone --config {rclone_config_path} mkdir {remote_training_path}"
        subprocess.run(mkdir_command, shell=True, capture_output=True, text=True)
        
        print(f"- リモートの中身を削除しました（{remote_training_path}フォルダは残っています）。")
    else:
        print(f"- 警告: ダウンロードに失敗しました。")
        print(f"  stdout: {download_result.stdout}")
        print(f"  stderr: {download_result.stderr}")
        
except Exception as e:
    print(f"- エラー: rcloneダウンロード処理中に予期しないエラーが発生しました: {e}")
print("-" * 20, flush=True)

# ZIPファイルの自動解凍処理
print("\n[3] ZIPファイルの解凍処理")
print(f"- 検索ディレクトリ: {working_directory}")
try:
    zip_files = glob.glob(os.path.join(working_directory, '*.zip'))
    if not zip_files:
        print("- 解凍対象のZIPファイルはありませんでした。")
    else:
        for zip_path in zip_files:
            zip_filename = os.path.basename(zip_path)
            print(f"- ZIPファイルを検出: {zip_filename}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(working_directory)
                print(f"  - {zip_filename} を正常に解凍しました。")
                os.remove(zip_path)
                print(f"  - 元のファイル {zip_filename} を削除しました。")
            except zipfile.BadZipFile:
                print(f"  - エラー: {zip_filename} は壊れているか、無効なZIPファイルです。")
            except Exception as e:
                print(f"  - エラー: {zip_filename} の処理中に問題が発生しました: {e}")
except Exception as e:
    print(f"- ZIPファイルの検索中にエラーが発生しました: {e}")
print("-" * 20, flush=True)


processed_folders = []
skipped_folders = []

# フォルダ一覧取得
print("\n[4] 処理対象の検出")
print(f"- ワーキングディレクトリ: {working_directory}")
try:
    all_entries = os.listdir(working_directory)
    folders = []
    for entry in all_entries:
        full_path = os.path.join(working_directory, entry)
        if os.path.isdir(full_path):
            if entry.startswith('.'):
                skipped_folders.append(f"{entry} (ドット始まりのためスキップ)")
            else:
                folders.append(entry)
        else:
            # ディレクトリでないエントリはスキップ対象として記録
            skipped_folders.append(f"{entry} (ファイルのためスキップ)")

    print("- 処理対象フォルダ:")
    if folders:
        for folder in folders:
            print(f"  - {folder}")
    else:
        print("  (なし)")

    if skipped_folders:
        print("- 処理対象外:")
        for item in skipped_folders:
            print(f"  - {item}")
    print("-" * 20, flush=True)

except FileNotFoundError:
    print(f"エラー: working_directory '{working_directory}' が見つかりません。")
    sys.exit(1)

# ベースとなる設定を準備
try:
    with open(os.path.join(program_directory, train_config_file), 'r', encoding='utf-8') as f:
        base_config = toml.load(f)
except FileNotFoundError:
    print(f"エラー: 基本学習設定ファイル '{os.path.join(program_directory, train_config_file)}' が見つかりません。")
    sys.exit(1)

# --add引数が指定されている場合、設定をマージ
if args.add:
    print("\n[5] 設定のマージと確認")
    additional_config_path = os.path.join(program_directory, args.add)
    try:
        with open(additional_config_path, 'r', encoding='utf-8') as f:
            additional_config = toml.load(f)
        
        print(f"- '{args.add}' の内容を基本設定にマージします。")
        
        original_config = copy.deepcopy(base_config)
        base_config = deep_update(base_config, additional_config)

        changes = compare_configs(original_config, base_config)
        if changes:
            print("- 変更された設定:")
            for change in changes:
                print(f"  {change}")
        else:
            print("- 設定の変更はありませんでした。")
        print("-" * 20, flush=True)

    except FileNotFoundError:
        print(f"警告: 追加設定ファイル '{additional_config_path}' が見つかりません。スキップします。")
    except Exception as e:
        print(f"警告: 追加設定ファイル '{additional_config_path}' の読み込みまたはマージ中にエラーが発生しました: {e}")

# ディレクトリ移動
try:
    os.chdir(working_directory)
except FileNotFoundError:
    # このエラーは上でキャッチされるはずだが念のため
    print(f"エラー: working_directory '{working_directory}' が見つかりません。")
    sys.exit(1)

os.chdir(kohya_directory)

print("\n" + "="*54)
print("||" + " 各フォルダの学習を開始します ".center(50) + "||")
print("="*54 + "\n")

# 各フォルダに対して処理を実行
for folder in folders:
    temp_config_file = os.path.join(temp_directory, f'{folder}_{output_suffix}.toml')
    original_model_path = None
    temp_model_path = None
    result = None
    should_skip = False

    try:
        # ベース設定をコピーして、フォルダ固有の設定を追加
        config = copy.deepcopy(base_config)

        config['train_data_dir'] = os.path.join(working_directory, folder)
        config['output_name'] = f'{folder}_{output_suffix}'
        config['output_dir'] = os.path.join(base_directory, paths.get('output_dir'))

        # モデルファイルの移動処理
        # `PSPACE_env.toml` では `pretrained_model_name_or_path` をファイル名のみで指定する想定
        # そのため `model_dir` と結合して実際のパスを作成する
        model_dir_setting = paths.get('model_dir', '.')
        pretrained_name = paths.get('pretrained_model_name_or_path')
        if not pretrained_name:
            print(f"[{folder}] 警告: pretrained_model_name_or_path が設定されていません。モデルの移動をスキップします。", flush=True)
            temp_model_path = None
            should_skip = True
        else:
            original_model_path = os.path.join(base_directory, model_dir_setting, pretrained_name)
            config['pretrained_model_name_or_path'] = original_model_path

            if not os.path.isabs(original_model_path):
                print(f"[{folder}] 警告: 作成した original_model_path が絶対パスではありません。モデルの移動をスキップします。", flush=True)
                temp_model_path = None
            else:
                model_filename = os.path.basename(original_model_path)
                temp_model_path = os.path.join(temp_directory, model_filename)

            if os.path.abspath(original_model_path) != os.path.abspath(temp_model_path):
                if os.path.exists(original_model_path):
                    print(f"[{folder}] モデルファイル {model_filename} を一時ディレクトリに移動します...", flush=True)
                    shutil.move(original_model_path, temp_model_path)
                    config['pretrained_model_name_or_path'] = temp_model_path
                elif os.path.exists(temp_model_path):
                    print(f"[{folder}] モデルファイル {model_filename} は既に一時ディレクトリに存在します。パスを更新します。", flush=True)
                    config['pretrained_model_name_or_path'] = temp_model_path
                else:
                    print(f"[{folder}] エラー: モデルファイルが見つかりません: {original_model_path}", flush=True)
                    skipped_folders.append(f"{folder} (モデルファイル欠落のためスキップ)")
                    should_skip = True
            else:
                print(f"[{folder}] モデルファイルは既に一時ディレクトリにあります。", flush=True)
        
        if should_skip:
            continue

        with open(temp_config_file, 'w', encoding='utf-8') as f:
            toml.dump(config, f)

        # コマンドを構築
        command = (
            f'{accelerate_path} launch '
            f'--dynamo_backend {acc_opts.get("dynamo_backend", "no")} '
            f'--dynamo_mode {acc_opts.get("dynamo_mode", "default")} '
            f'--mixed_precision {acc_opts.get("mixed_precision", "bf16")} '
            f'--num_processes {acc_opts.get("num_processes", 1)} '
            f'--num_machines {acc_opts.get("num_machines", 1)} '
            f'--num_cpu_threads_per_process {acc_opts.get("num_cpu_threads_per_process", 2)} '
            f'"{train_script_path}" '
            f'--config_file "{temp_config_file}" '
            f'--log_prefix={train_opts.get("log_prefix", "xl-loha")} '
        )

        result = run_training_with_retry(command, temp_config_file, config, program_directory, folder)

    finally:
        # モデルを元の場所に戻す
        if original_model_path and temp_model_path and os.path.exists(temp_model_path):
            if os.path.abspath(original_model_path) != os.path.abspath(temp_model_path):
                print(f"[{folder}] モデルファイル {os.path.basename(temp_model_path)} を元の場所に戻します...", flush=True)
                
                # クリーンアップ中のSIGINT (CTRL+C) を一時的にブロックして、移動処理を保護する
                original_sigint_handler = signal.getsignal(signal.SIGINT)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                
                try:
                    # リトライロジック付きで移動
                    max_retries = 5
                    for i in range(max_retries):
                        try:
                            os.makedirs(os.path.dirname(original_model_path), exist_ok=True)
                            shutil.move(temp_model_path, original_model_path)
                            print(f"[{folder}] モデルファイルを正常に復元しました。", flush=True)
                            break
                        except PermissionError:
                            if i < max_retries - 1:
                                print(f"[{folder}] ファイルがロックされています。1秒後に再試行します... ({i+1}/{max_retries})", flush=True)
                                time.sleep(1)
                            else:
                                raise
                        except Exception:
                            raise
                except Exception as e:
                    print(f"[{folder}] 警告: モデルファイルの復元に失敗しました: {e}", flush=True)
                    print(f"[{folder}] 重要: モデルファイルは一時ディレクトリに残っています: {temp_model_path}", flush=True)
                finally:
                    # シグナルハンドラを復元
                    signal.signal(signal.SIGINT, original_sigint_handler)

    if should_skip:
        continue

    # 実行結果のハンドリング
    if result is None or result.returncode != 0:
        print(f"処理失敗: {folder}。フォルダは削除されませんでした。")
        if result and result.stderr:
            print("----- Stderr -----")
            print(result.stderr.strip())
            print("--------------------")
        try:
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
        except Exception as e:
            print(f"警告: {temp_config_file} の削除に失敗しました: {e}", flush=True)
        continue

    # 成功した場合の出力を表示
    print(f"[{folder}] の学習が正常に完了しました。")
    if result.stdout:
        print("----- Stdout -----")
        print(result.stdout.strip())
        print("--------------------")

    # rcloneでファイルをアップロード
    try:
        output_dir = os.path.join(base_directory, paths.get('output_dir'))
        rclone_config_path = os.path.join(program_directory, 'rclone.conf')
        print("学習済みモデルをアップロードします...", flush=True)
        remote_path = env_config.get('rclone', {}).get('remote_path', 'google:runpod/AI')
        rclone_target = f"{remote_path.rstrip('/')}/output"
        rclone_command = f"rclone --config {rclone_config_path} copy {output_dir} {rclone_target}"
        rclone_result = subprocess.run(rclone_command, shell=True, capture_output=True, text=True)
        if rclone_result.returncode != 0:
            rclone_output = (rclone_result.stdout or "") + "\n" + (rclone_result.stderr or "")
            if 'storageQuotaExceeded' in rclone_output or "Drive storage quota" in rclone_output or "The user's Drive storage quota has been exceeded" in rclone_output:
                print("エラー: Google Drive の容量が超過しています。アップロードを中止し、プログラムを終了します。フォルダは削除されません。", flush=True)
                print("rclone 出力:", flush=True)
                print(rclone_output, flush=True)
                sys.exit(1)
            else:
                print(f"アップロードに失敗しました（returncode={rclone_result.returncode}）。フォルダは削除されません。出力:", flush=True)
                print(rclone_output, flush=True)
                try:
                    os.remove(temp_config_file)
                except Exception as e:
                    print(f"警告: {temp_config_file} の削除に失敗しました: {e}", flush=True)
                continue
    except Exception as e:
        print(f"警告: rclone アップロード処理中に予期しないエラーが発生しました: {e}", flush=True)
        try:
            os.remove(temp_config_file)
        except Exception as e2:
            print(f"警告: {temp_config_file} の削除に失敗しました: {e2}", flush=True)
        continue

    print("アップロードが完了しました。", flush=True)

    # Google Drive のゴミ箱を空にする
    try:
        empty_trash = env_config.get('rclone', {}).get('empty_trash_after_upload', True)
        if empty_trash:
            try:
                remote_name = remote_path.split(':', 1)[0]
            except Exception:
                remote_name = remote_path
            print(f"rclone による Google Drive のゴミ箱空にする処理を実行します: {remote_name}:", flush=True)
            cleanup_cmd = f"rclone --config {rclone_config_path} cleanup {remote_name}:"
            cleanup_result = subprocess.run(cleanup_cmd, shell=True, capture_output=True, text=True)
            if cleanup_result.returncode == 0:
                print("Google Drive のゴミ箱を空にしました（rclone cleanup 成功）。", flush=True)
            else:
                print("警告: Google Drive のゴミ箱を空にする際にエラーが発生しました。rclone 出力:", flush=True)
                print((cleanup_result.stdout or "") + "\n" + (cleanup_result.stderr or ""), flush=True)
    except Exception as e:
        print(f"警告: rclone cleanup 処理中に予期しないエラーが発生しました: {e}", flush=True)

    # /workspace/output の中身を削除
    try:
        output_dir_to_clear = output_dir
        print(f"'{output_dir_to_clear}' の中身を消去します...", flush=True)
        for item in os.listdir(output_dir_to_clear):
            item_path = os.path.join(output_dir_to_clear, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f'{item_path} の削除中にエラーが発生しました: {e}', flush=True)
        print(f"'{output_dir_to_clear}' の中身を消去しました。", flush=True)
    except Exception as e:
        print(f"警告: output_dir の消去処理中に予期しないエラーが発生しました: {e}", flush=True)

    # 成功時のみフォルダ削除
    try:
        folder_path = os.path.join(working_directory, folder)
        if args.test:
             print(f"テストモード: {folder_path} の削除をスキップします。", flush=True)
             processed_folders.append(folder)
        else:
            try:
                shutil.rmtree(folder_path)
                print(f'正常終了: {folder_path} を削除しました。', flush=True)
                processed_folders.append(folder)
            except OSError as e:
                print(f"エラー: {folder_path} の削除に失敗しました - {e}", flush=True)
        
        try:
            os.remove(temp_config_file)
        except Exception as e:
            print(f"警告: {temp_config_file} の削除に失敗しました: {e}", flush=True)
    except Exception as e:
        print(f"警告: フォルダ削除処理中に予期しないエラーが発生しました: {e}", flush=True)


# 全ての処理が完了した後にメッセージを表示
print("\n" + "="*50)
print("|| 全ての処理が完了しました。結果の要約: ||")
print("="*50)

print(f"\n正常に処理され、削除されたフォルダ ({len(processed_folders)}件):")
if processed_folders:
    for item in processed_folders:
        print(f"- {item}")
else:
    print("  なし")

# 最終的に残ったフォルダを計算
remaining_folders = [f for f in folders if f not in processed_folders]

print(f"\n処理が完了しなかった、またはエラーで残ったフォルダ ({len(remaining_folders)}件):")
if remaining_folders:
    for item in remaining_folders:
        print(f"- {item}")
else:
    print("  なし")

# 最初にスキップされたエントリも表示
print(f"\n最初から処理対象外だったエントリ ({len(skipped_folders)}件):")
if skipped_folders:
    for item in skipped_folders:
        print(f"- {item}")
else:
    print("  なし")

print("\n" + "="*50)
print("|| 出来上がった LORAファイルをダウンロード ||")
print("="*50 + "\n")
