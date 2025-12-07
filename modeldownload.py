import os
import tempfile
import shutil
import sys
import subprocess
import importlib.util
import getpass
from logging import getLogger, basicConfig, INFO

# ロギングの設定
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

# パッケージのインポート（自動インストールは行いません）
import toml
from huggingface_hub import hf_hub_download


def _load_toml_file(path):
    """TOMLファイルを読み込む。Python 3.11+ の tomllib を優先し、なければ toml パッケージを使用する。"""
    try:
        # Python 3.11+ の場合
        import tomllib  # type: ignore

        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        # フォールバックで toml パッケージを使う
        with open(path, "r", encoding="utf-8") as f:
            return toml.load(f)

def is_ipython_or_jupyter():
    """IPythonまたはJupyter環境で実行されているかを検出する"""
    try:
        # 方法1: get_ipython()関数が存在するかチェック
        get_ipython()  # type: ignore
        return True
    except NameError:
        pass
    
    try:
        # 方法2: __IPYTHON__変数が存在するかチェック
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        pass
    
    # 方法3: sys.modulesにIPythonが含まれているかチェック
    if 'IPython' in sys.modules:
        return True
    
    return False

def download_model(repo_id, filename, model_dir, token=None):
    """
    Hugging Face Hubから指定されたモデルファイルをダウンロードする。
    保存先は `model_dir` を使います（`save_dir` は廃止）。
    """
    try:
        # 保存先ディレクトリが存在しない場合は作成
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"作成されたディレクトリ: {model_dir}")

        logger.info(f"'{repo_id}/{filename}' のダウンロードを開始します（キャッシュは永続化しません）...")

        # 一時ディレクトリをキャッシュ先として使用し、ダウンロード完了後に目的の保存先へコピーする。
        # これにより恒久的な Hugging Face キャッシュを残さない。
        tmp_cache = tempfile.mkdtemp(prefix="hf_cache_")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=tmp_cache,
                force_download=True,
                token=token
            )

            logger.info(f"ダウンロード完了（一時キャッシュ）: {downloaded_path}")

            final_path = os.path.join(model_dir, filename)
            # ダウンロード済ファイルを保存先へコピー（上書き）
            shutil.copy2(downloaded_path, final_path)
            logger.info(f"モデルを '{final_path}' に保存しました。")
        finally:
            # 一時キャッシュを削除して永続キャッシュを残さない
            try:
                shutil.rmtree(tmp_cache)
            except Exception:
                pass


    except Exception as e:
        logger.error(f"'{repo_id}/{filename}' のダウンロード中にエラーが発生しました: {e}")

def main():
    """
    `PSPACE_env.toml` の `[modeldownload]` セクションから設定を読み込み、モデルをダウンロードする。
    """
    cfg_path = "PSPACE_env.toml"
    if not os.path.exists(cfg_path):
        logger.error(f"設定ファイル '{cfg_path}' が見つかりません。'")
        return

    try:
        config = _load_toml_file(cfg_path)
        logger.info(f"設定ファイル '{cfg_path}' を読み込みました。")
    except Exception as e:
        logger.error(f"設定ファイル '{cfg_path}' の読み込み中にエラーが発生しました: {e}")
        return

    model_section = config.get("modeldownload")
    if not model_section or not isinstance(model_section, dict):
        logger.error("'[modeldownload]' セクションが PSPACE_env.toml に見つかりません。")
        return

    hf_token = model_section.get("token")
    
    # 環境変数からトークンを読み取る（Jupyter notebook対応）
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if hf_token:
            logger.info("環境変数 HF_TOKEN からトークンを読み込みました。")
    
    if not hf_token:
        logger.info("[modeldownload] に API トークンが設定されていません。")
        logger.info("以下のいずれかの方法でトークンを設定してください:")
        logger.info("  1. PSPACE_env.toml の [modeldownload] に token = \"your_token\" を追加")
        logger.info("  2. 環境変数 HF_TOKEN を設定（Jupyter: %env HF_TOKEN=your_token）")
        logger.info("  3. 対話式入力（以下のプロンプト）")
        
        # Jupyter/IPython環境かどうかをチェック
        in_jupyter = is_ipython_or_jupyter()
        
        if in_jupyter:
            # Jupyter環境では必ずinput()を使用
            logger.info("Jupyter/IPython環境を検出しました。")
            logger.warning("!python で実行している場合は入力できません。")
            logger.warning("代わりに以下のように実行してください:")
            logger.warning("  %env HF_TOKEN=your_token_here")
            logger.warning("  !python modeldownload.py")
            try:
                hf_token = input("Hugging Face API トークンを入力してください（不要な場合はEnterキー）: ").strip()
            except (EOFError, OSError) as e:
                logger.error(f"標準入力が利用できません: {e}")
                logger.error("環境変数 HF_TOKEN を設定してください。")
                hf_token = ""
            except Exception as e:
                logger.error(f"トークン入力に失敗しました: {e}")
                hf_token = ""
        else:
            # 通常の環境でも、まずinput()を試す（より互換性が高い）
            try:
                # 環境変数でgetpassを強制する場合のみgetpassを使用
                use_getpass = os.environ.get('USE_GETPASS', '').lower() in ('true', '1', 'yes')
                
                if use_getpass:
                    hf_token = getpass.getpass("Hugging Face API トークンを入力してください（不要な場合はEnterキー）: ").strip()
                else:
                    hf_token = input("Hugging Face API トークンを入力してください（不要な場合はEnterキー）: ").strip()
            except Exception as e:
                logger.warning(f"トークン入力エラー: {e}")
                try:
                    hf_token = input("Hugging Face API トークンを入力してください（不要な場合はEnterキー）: ").strip()
                except Exception as e2:
                    logger.error(f"トークン入力に失敗しました（フォールバック）: {e2}")
                    hf_token = ""
        
        if not hf_token:
            logger.warning("API トークンが入力されませんでした。プライベートリポジトリのダウンロードは失敗する可能性があります。")
            hf_token = None

    # paths セクションから base_directory と model_dir を取得し、保存先ディレクトリを決定する
    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}
    base_directory = paths_cfg.get("base_directory", ".")
    paths_model_dir = paths_cfg.get("model_dir", "model")

    # 組み立てルール: `paths.model_dir` が絶対パスであればそのまま使用。
    # そうでなければ `base_directory` と結合して使用する。
    if os.path.isabs(paths_model_dir):
        default_model_dir = paths_model_dir
    else:
        default_model_dir = os.path.join(base_directory, paths_model_dir)

    # `filename` を各モデルエントリで指定する方法を廃止します。
    # 代わりに `[paths].pretrained_model_name_or_path` を使ってファイル名を指定してください。
    paths_pretrained = paths_cfg.get("pretrained_model_name_or_path")
    if not paths_pretrained:
        logger.error("'[paths].pretrained_model_name_or_path' が設定されていません。ダウンロードするファイル名を指定してください。")
        return

    # [modeldownload] セクションから直接 repo_id を取得
    repo_id = model_section.get("repo_id")
    if not repo_id:
        logger.error("'[modeldownload].repo_id' が設定されていません。")
        return

    model_dir = default_model_dir
    filename = paths_pretrained

    if repo_id and filename:
        download_model(repo_id, filename, model_dir, token=hf_token)
    else:
        logger.error("設定項目が不足しています（repo_idまたはpaths.pretrained_model_name_or_path）")


if __name__ == "__main__":
    main()