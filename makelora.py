import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import subprocess
import sys
import shutil
import argparse

import toml
import html as _html

from threading import Thread

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

    # Jupyter 環境向けに、出力表示を上書きできる DisplayHandle を作る
    display_handle = None
    try:
        # IPython があれば display_id を使って上書き表示を試みる
        from IPython.display import display, HTML
        display_handle = display(HTML("<pre></pre>"), display_id=True)
    except Exception:
        display_handle = None

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

                    # Jupyter の表示ハンドルがあれば、全バッファを HTML の <pre> で更新して上書き表示する
                    if display_handle is not None:
                        try:
                            # HTML に埋める前にエスケープ
                            text = _html.escape(''.join(container))
                            display_handle.update(HTML(f"<pre>{text}</pre>"))
                        except Exception as e:
                            # 失敗したら通常出力へ落とす（以降は display_handle を使わない）
                            try:
                                sys.stdout.write(line)
                                sys.stdout.flush()
                            except Exception:
                                try:
                                    print(line, end='', flush=True)
                                except Exception:
                                    pass
                    else:
                        # 通常のターミナルではそのまま出力（改行や\rを保持）
                        try:
                            sys.stdout.write(line)
                            sys.stdout.flush()
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

    process.wait() # プロセスの終了を待つ

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
parser.add_argument('--body', action='store_true', help='全身画像用の設定を適用（face_crop_aug_rangeを削除）')
args = parser.parse_args()

print("\n" + "="*50)
if args.body:
    print("|| 全身（体用）モードで学習を開始します。 ||")
else:
    print("|| 顔用モードで学習を開始します。 ||")
print("="*50 + "\n")

# 環境設定ファイルを読み込む
env_config_file = "PSPACE_env.toml"
with open(env_config_file, 'r', encoding='utf-8') as f:
    env_config = toml.load(f)

paths = env_config.get('paths', {})
acc_opts = env_config.get('accelerate_options', {})
train_opts = env_config.get('train_options', {})
makelora_settings = env_config.get('makelora_settings', {})

output_suffix = makelora_settings.get('output_suffix')
train_config_file = makelora_settings.get('train_config_file')

if not output_suffix or not train_config_file:
    print(f"エラー: '{env_config_file}' に 'output_suffix' と 'train_config_file' を設定してください。")
    sys.exit(1)

# パス設定
base_directory = paths.get('base_directory', '/notebooks')
working_directory = os.path.join(base_directory, paths.get('working_directory', 'training'))
kohya_directory = paths.get('kohya_directory', '/kohya_ss')
program_directory = os.path.join(base_directory, paths.get('program_directory', 'program'))
temp_directory = paths.get('temp_directory', '/tmp')
accelerate_path = paths.get('accelerate_path', '/venv/bin/accelerate')
train_script_path = paths.get('train_script_path', '/kohya_ss/sd-scripts/sdxl_train_network.py')

processed_folders = []
skipped_folders = []

# ディレクトリ移動とフォルダ一覧取得
try:
    # working_directory のフルパスで一覧を取得してから移動するようにする
    all_entries = os.listdir(working_directory)
    print(f"working_directory='{working_directory}' から検出したエントリ: {all_entries}")

    folders = []
    for entry in all_entries:
        full_path = os.path.join(working_directory, entry)
        if os.path.isdir(full_path):
            if entry.startswith('.'):
                skipped_folders.append(f"{entry} (ドット始まりのためスキップ)")
            else:
                folders.append(entry)
        else:
            # ディレクトリでないエントリはスキップ対象として記録（任意）
            # skipped_folders.append(f"{entry} (ファイルのためスキップ)")
            pass

    print(f"処理対象フォルダ: {folders}", flush=True)
    if skipped_folders:
        print(f"処理対象外のエントリ: {skipped_folders}", flush=True)

    os.chdir(working_directory)
except FileNotFoundError:
    print(f"エラー: working_directory '{working_directory}' が見つかりません。")
    sys.exit(1)

os.chdir(kohya_directory)

# 各フォルダに対して処理を実行
for folder in folders:
    temp_config_file = os.path.join(temp_directory, f'{folder}_{output_suffix}.toml')
    
    # 学習設定ファイルを読み込み、内容を更新
    try:
        with open(os.path.join(program_directory, train_config_file), 'r', encoding='utf-8') as f:
            config = toml.load(f)
    except FileNotFoundError:
        print(f"エラー: 学習設定ファイル '{os.path.join(program_directory, train_config_file)}' が見つかりません。")
        continue

    # --body引数が指定されている場合、face_crop_aug_rangeを削除
    if args.body:
        if 'face_crop_aug_range' in config:
            del config['face_crop_aug_range']
            print("全身画像モード: face_crop_aug_range を設定から削除しました。")


    config['train_data_dir'] = os.path.join(working_directory, folder)
    config['output_name'] = f'{folder}_{output_suffix}'
    config['output_dir'] = os.path.join(base_directory, paths.get('output_dir'))
    config['pretrained_model_name_or_path'] = os.path.join(base_directory, paths.get('pretrained_model_name_or_path'))


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

    # print(command) # デバッグ用にコマンド全体を表示する場合はコメントアウトを解除
    result = run_training_with_retry(command, temp_config_file, config, program_directory, folder)

    # 実行結果のハンドリング
    if result.returncode != 0:
        print(f"処理失敗: {folder}。フォルダは削除されませんでした。")
        if result.stderr:
            print("----- Stderr -----")
            print(result.stderr.strip())
            print("--------------------")
        os.remove(temp_config_file)
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
        # PSPACE_env.toml の [rclone].remote_path を使い、末尾に /output を追加
        remote_path = env_config.get('rclone', {}).get('remote_path', 'google:runpod/AI')
        rclone_target = f"{remote_path.rstrip('/')}/output"
        rclone_command = f"rclone --config {rclone_config_path} copy {output_dir} {rclone_target}"
        # rclone の出力を取得してエラーの種類を判定する
        rclone_result = subprocess.run(rclone_command, shell=True, capture_output=True, text=True)
        if rclone_result.returncode != 0:
            rclone_output = (rclone_result.stdout or "") + "\n" + (rclone_result.stderr or "")
            # Google Drive の容量超過を示すエラーを検出したら、ユーザーに知らせてプログラムを終了する
            if 'storageQuotaExceeded' in rclone_output or "Drive storage quota" in rclone_output or "The user's Drive storage quota has been exceeded" in rclone_output:
                print("エラー: Google Drive の容量が超過しています。アップロードを中止し、プログラムを終了します。フォルダは削除されません。", flush=True)
                print("rclone 出力:", flush=True)
                print(rclone_output, flush=True)
                sys.exit(1)
            else:
                print(f"アップロードに失敗しました（returncode={rclone_result.returncode}）。フォルダは削除されません。出力:", flush=True)
                print(rclone_output, flush=True)
                # 他のエラーの場合はこのフォルダは削除せず次へ進む
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

    # Google Drive のゴミ箱を空にする（rclone cleanup）
    # PSPACE_env.toml の [rclone].empty_trash_after_upload で制御（デフォルト True）
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
