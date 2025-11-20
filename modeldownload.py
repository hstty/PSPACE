import os
import tempfile
import shutil
import sys
import subprocess
import importlib.util
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

def download_model(repo_id, filename, save_dir, token=None):
    """
    Hugging Face Hubから指定されたモデルファイルをダウンロードする。
    """
    try:
        # 保存先ディレクトリが存在しない場合は作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"作成されたディレクトリ: {save_dir}")

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

            final_path = os.path.join(save_dir, filename)
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
    if not hf_token:
        logger.warning("[modeldownload] に API トークンが設定されていません。プライベートリポジトリのダウンロードは失敗する可能性があります。")

    models = model_section.get("models")
    if models and isinstance(models, list):
        for model_info in models:
            repo_id = model_info.get("repo_id")
            filename = model_info.get("filename")
            save_dir = model_info.get("save_dir", ".")  # デフォルトはカレントディレクトリ

            if repo_id and filename:
                download_model(repo_id, filename, save_dir, token=hf_token)
            else:
                logger.warning(f"設定項目が不足しています（repo_idまたはfilename）: {model_info}")
    else:
        logger.error("'[modeldownload].models' のリストが PSPACE_env.toml に見つかりません。")


if __name__ == "__main__":
    main()