
import subprocess
import os
import sys
import shutil


def check_rclone_and_reauthenticate():
    """
    rcloneの接続をチェックし、必要であれば再認証を促します。
    """
    print("\n--- rclone接続チェック ---")
    # スクリプトと同じフォルダにある rclone.conf を優先して使う
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rclone_config_path = os.path.join(script_dir, 'rclone.conf')

    # PSPACE_env.toml の [rclone].remote_path を優先的に使う
    env_path = os.path.join(script_dir, 'PSPACE_env.toml')
    remote_path = "google:runpod/AI"  # デフォルト
    try:
        import toml
        if os.path.exists(env_path):
            try:
                cfg = toml.load(env_path)
                remote_path = cfg.get('rclone', {}).get('remote_path', remote_path)
            except Exception as e:
                print(f"PSPACE_env.toml の読み込み中に問題が発生しました: {e}。デフォルトの remote_path を使用します。")
        else:
            print(f"PSPACE_env.toml が {env_path} に見つかりません。デフォルトの remote_path を使用します。")
    except Exception:
        # toml が無い場合は警告してデフォルトを使う
        print("'toml' パッケージが利用できないため PSPACE_env.toml を読み込めません。デフォルトの remote_path を使用します。")

    # チェック用のコマンド (lsd: リモートのディレクトリをリスト)
    check_command = [
        "rclone",
        "--config", rclone_config_path,
        "lsd",
        "--low-level-retries", "1",
        remote_path,
    ]

    print("rclone の接続をテストしています...")
    print(f"コマンド: {' '.join(check_command)}")

    try:
        subprocess.run(check_command, check=True, capture_output=True, text=True)
        print("\n[成功] rclone の接続は正常です。")

    except subprocess.CalledProcessError as e:
        print("\n[警告] rclone の接続に失敗しました。")
        print("--------------------------------------------------")
        print("エラー内容:")
        error_output = e.stderr if e.stderr else e.stdout
        print(error_output)
        print("--------------------------------------------------")
        print("\n認証が切れているか、設定に問題がある可能性があります。")
        print("rclone の再認証手順を案内します。")

        # reconnect に渡すのはリモート名のみ (remote:subpath の形式から取り出す)
        remote_name = remote_path.split(':', 1)[0] if ':' in remote_path else remote_path
        reauth_command = [
            "rclone",
            "--config", rclone_config_path,
            "config",
            "reconnect",
            f"{remote_name}:",
        ]

        print("\n以下のコマンドを手動で実行して、再認証を完了してください。")
        print("ターミナルに表示されるURLをブラウザで開き、認証コードを貼り付ける必要があります。")
        print("--------------------------------------------------")
        print(f"{' '.join(reauth_command)}")
        print("--------------------------------------------------")


def apply_train_util_patch():
    """
    ワークスペース内の `train_util.patch` を使って
    `/kohya_ss/sd-scripts/library/train_util.py` にパッチを適用します。
    """
    script_dir_local = os.path.dirname(os.path.abspath(__file__))
    patch_file = os.path.join(script_dir_local, 'train_util.patch')
    
    # PSPACE_env.toml から kohya_directory を取得
    env_path = os.path.join(script_dir_local, 'PSPACE_env.toml')
    kohya_dir = '/kohya_ss' # デフォルト
    try:
        import toml
        if os.path.exists(env_path):
            try:
                cfg = toml.load(env_path)
                kohya_dir = cfg.get('paths', {}).get('kohya_directory', kohya_dir)
            except Exception:
                pass # 読み込み失敗時はデフォルトを使用
    except ImportError:
        pass # tomlがない場合はデフォルトを使用

    target_file = os.path.normpath(os.path.join(kohya_dir, 'sd-scripts/library/train_util.py'))
    working_dir = os.path.normpath(kohya_dir)

    print(f"\n--- `train_util.py` にパッチを適用します ---")

    if not os.path.exists(patch_file):
        print(f"パッチファイルが見つかりません: {patch_file}")
        return

    if not os.path.exists(target_file):
        print(f"ターゲットファイルが見つかりません: {target_file}")
        return

    if not os.path.exists(working_dir):
        print(f"作業ディレクトリが見つかりません: {working_dir}")
        return

    # git apply コマンドの構築
    # --ignore-whitespace: 空白の違いを無視 (CRLF/LF対策)
    # --verbose: 詳細出力
    command = ["git", "apply", "--verbose", "--ignore-whitespace", patch_file]
    
    try:
        print(f"作業ディレクトリ: {working_dir}")
        print(f"コマンド: {' '.join(command)}")
        
        subprocess.run(command, cwd=working_dir, check=True)
        print(f"[成功] パッチを適用しました。")

    except subprocess.CalledProcessError as e:
        print(f"[エラー] パッチの適用に失敗しました。既適用か、ファイルが一致しない可能性があります。")
    except Exception as e:
        print(f"[エラー] 予期せぬエラーが発生しました: {e}")


def remove_dot_hidden_dirs_from_base(env_toml_path=None, dry_run=False, verbose=True):
    """
    PSPACE_env.toml の [paths].base_directory を読み、
    先頭が '.' のディレクトリを全て削除します。

    - `env_toml_path` が None の場合はスクリプトと同じフォルダの `PSPACE_env.toml` を使います。
    - `dry_run=True` の場合は削除は行わず一覧表示のみ行います。
    - 戻り値は削除（または削除予定）のパスのリスト。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if env_toml_path is None:
        env_toml_path = os.path.join(script_dir, 'PSPACE_env.toml')

    if not os.path.exists(env_toml_path):
        print(f"PSPACE_env.toml が見つかりません: {env_toml_path}")
        return []

    # toml 読み込み互換処理 (tomllib or toml)
    def _load_toml(path):
        try:
            import tomllib
            with open(path, 'rb') as f:
                return tomllib.load(f)
        except Exception:
            try:
                import toml
                with open(path, 'r', encoding='utf-8') as f:
                    return toml.load(f)
            except Exception as e:
                raise RuntimeError(f"toml の読み込みに失敗しました: {e}")

    cfg = _load_toml(env_toml_path)
    base_dir = cfg.get('paths', {}).get('base_directory')
    if not base_dir:
        print("PSPACE_env.toml の [paths].base_directory が設定されていません。")
        return []

    base_dir = os.path.expanduser(base_dir)
    if not os.path.isabs(base_dir):
        base_dir = os.path.abspath(os.path.join(script_dir, base_dir))

    abs_base = os.path.abspath(base_dir)
    # 安全対策: ルートや短すぎるパスを誤って削除しない
    if abs_base in ('/', '\\') or len(abs_base) <= 3:
        print(f"[安全停止] ベースディレクトリが危険なパスの可能性があります: {abs_base}")
        return []

    if not os.path.exists(abs_base):
        print(f"ベースディレクトリが見つかりません: {abs_base}")
        return []

    removed = []
    for name in os.listdir(abs_base):
        if name.startswith('.'):
            p = os.path.join(abs_base, name)
            if os.path.isdir(p):
                if dry_run:
                    if verbose:
                        print(f"[DRY-RUN] 削除予定: {p}")
                    removed.append(p)
                else:
                    try:
                        shutil.rmtree(p)
                        if verbose:
                            print(f"[削除] {p}")
                        removed.append(p)
                    except Exception as e:
                        print(f"[エラー] {p} の削除に失敗しました: {e}")

    return removed


def upload_program_files():
    """
    スクリプトと同じフォルダにある全ファイルを
    PSPACE_env.toml の [rclone].remote_path + '/program/PSPACE' にアップロードします。
    """
    print("\n--- プログラムファイルのアップロード ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rclone_config_path = os.path.join(script_dir, 'rclone.conf')
    env_path = os.path.join(script_dir, 'PSPACE_env.toml')

    # デフォルトのリモートパス
    remote_base = "google:runpod/AI"
    
    try:
        import toml
        if os.path.exists(env_path):
            try:
                cfg = toml.load(env_path)
                remote_base = cfg.get('rclone', {}).get('remote_path', remote_base)
            except Exception:
                pass
    except ImportError:
        pass

    # アップロード先パスの構築
    # 末尾のスラッシュ処理などはrcloneがよしなにやってくれるが、念のため綺麗に結合
    # remote_base が "google:LORA" なら "google:LORA/program/PSPACE" になる
    destination = f"{remote_base}/program/PSPACE".replace('//', '/')

    print(f"アップロード元: {script_dir}")
    print(f"アップロード先: {destination}")

    # rclone copy コマンド
    # --config: 設定ファイル指定
    # --exclude: .gitフォルダなどを除外したい場合はここに追加
    # ここでは "スクリプトと同じフォルダの全ファイル" なのでそのまま copy
    command = [
        "rclone",
        "--config", rclone_config_path,
        "copy",
        script_dir,
        destination,
        "--exclude", ".*/**",      # .で始まる隠しフォルダ/ファイルを除外 (.git, .envなど)
        "--exclude", "__pycache__/**",
        "--exclude", "*.pyc",
        "--exclude", "venv/**",
        "--verbose"
    ]

    try:
        subprocess.run(command, check=True)
        print("[成功] ファイルのアップロードが完了しました。")
    except subprocess.CalledProcessError as e:
        print(f"[エラー] ファイルのアップロードに失敗しました: {e}")
    except Exception as e:
        print(f"[エラー] 予期せぬエラーが発生しました: {e}")


# --- メイン処理 ---
# このスクリプトが直接実行された場合に、以下の処理を開始します。
if __name__ == "__main__":
    # 先に requirements.txt があればインストールする
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        req_path = os.path.join(script_dir, 'requirements.txt')
        if os.path.exists(req_path):
            print(f"requirements.txt を検出しました。パッケージをインストールします: {req_path}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_path])
        else:
            print("requirements.txt が見つかりません。インストールをスキップします。")
    except subprocess.CalledProcessError as e:
        print(f"パッケージのインストール中にエラーが発生しました: {e}")
        print("必要なパッケージが不足している可能性がありますが、処理は継続します。")

    check_rclone_and_reauthenticate()

    # プログラムファイルのアップロードを実行
    upload_program_files()

    # 実行時に kohya_ss 用の train_util.py にパッチを適用する（失敗しても継続）
    try:
        apply_train_util_patch()
    except Exception:
        print('`train_util.py` へのパッチ適用でエラーが発生しましたが、処理を継続します。')

    # base_directory 内の先頭が '.' の隠しフォルダを削除（デフォルトは実行）
    try:
        print("\n--- base_directory 内の '.' で始まる隠しフォルダを削除します（デフォルト: 実行） ---")
        removed = remove_dot_hidden_dirs_from_base(dry_run=False, verbose=True)
        if removed:
            print(f"\n{len(removed)} 個の隠しフォルダを削除しました。問題があればログを確認してください。")
        else:
            print("\n削除対象の隠しフォルダは見つかりませんでした。")
    except Exception as e:
        print(f"隠しフォルダ削除処理でエラーが発生しました: {e}")
