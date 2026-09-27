#!/usr/bin/env bash
# 依 backend/models.lock 從 GitHub Release 下載模型 artifact 並驗證 sha256。
# 模型檔不進 git（.gitignore），雲端排程與新環境靠這支取得「與 lock 完全相同」的檔案。
# 需要 gh CLI（Actions 內建，GH_TOKEN=github.token）。已存在且雜湊相符的檔案不重抓。
set -euo pipefail
cd "$(dirname "$0")/.."

while read -r tag asset sha path; do
  [[ -z "${tag}" || "${tag}" == \#* ]] && continue
  if ! echo "${sha}  ${path}" | sha256sum --check --status 2>/dev/null; then
    mkdir -p "$(dirname "${path}")"
    # 大檔（數百 MB）偶爾下載到一半連線被重設：重試 3 次
    for attempt in 1 2 3; do
      gh release download "${tag}" --pattern "${asset}" --output "${path}" --clobber && break
      [[ ${attempt} -eq 3 ]] && exit 1
      echo "下載 ${asset} 失敗，第 ${attempt} 次重試…" >&2
      sleep $((attempt * 10))
    done
  fi
  echo "${sha}  ${path}" | sha256sum --check -
done < backend/models.lock
