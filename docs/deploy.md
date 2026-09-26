# 部署：stock.sekinv.com（靜態站＋每日排程）

2026-09 起公開站改為**全靜態**，每日排程也從 Windows 工作排程器（`daily_predict.bat`）搬到
GitHub Actions：每個交易日收盤後記錄線上預測、把 API 的回應匯出成 JSON，Next.js 靜態輸出後
部署到 Cloudflare Pages。本機開發（uvicorn＋`next dev`）不受影響。

```
GitHub Actions（daily.yml，台北 15:20；16:50 補跑）
  還原狀態（state 分支：predictions.db、價格快取、報告、上一版網站資料）
  → record_predictions / record_rank_predictions（--require-complete）→ report_*
  → export_static：TestClient 呼叫現有 API → frontend/public/data/*.json（含發布關卡）
  → next build（NEXT_PUBLIC_STATIC_DATA=1）→ wrangler pages deploy → 保存狀態
```

## 重點設計

| 問題 | 做法 |
|---|---|
| 靜態站沒有後端 | `stockta/export_static.py` 用 TestClient 逐一呼叫現有端點、原樣寫檔；數字與本機 API 同源 |
| 帶參數的端點 | 匯出最大範圍、前端切片：K 線近 5 年、歷史回放近 730 天（(start, end] 篩選並重算命中率）、全池掃描與排名保留最近 60 個交易日（查詢日對齊到 ≤ 它的交易日，超出範圍明確報錯） |
| 匯出途中抓到新 K 棒 | 匯出時資料源換成只讀快取的 provider；`/prediction` 的寫入副作用關閉（只由 record 寫入） |
| 還原的快取被當成最新 | `ParquetCache.is_fresh` 看檔案修改時間；還原後設為 3 天前，即時路徑才會重抓 |
| 部分標的抓取失敗 | 自癒改以「標的×日」為單位、缺當日 K 棒不記；有缺漏時 exit 2 → 16:50 補跑 |
| 把假資料發布出去 | 匯出關卡：模型 mock、覆蓋率 < 90%、資料日期倒退 → exit 1，不部署，線上維持上一版 |
| 資料庫要跨次保存 | 孤立分支 `state`，每次覆寫成單一 commit；另存 30 天 artifact |
| 模型不能進 git | GitHub Release `models-v1` ＋ `backend/models.lock`（sha256） |
| pickle 模型對版本敏感 | `backend/requirements-ci.txt` 鎖定與 Windows 開發機相同的版本 |
| 新聞情緒卡 | 讀 `https://news.sekinv.com/data/stocks/<ADR>/sentiment.json`（新聞站以 `_headers` 開放跨站） |

## Workflows

| 檔案 | 觸發 | 做什麼 |
|---|---|---|
| `daily.yml` | 平日 07:20、08:50 UTC、手動 | 上述完整流程；第二次（補跑）若當天已完成就略過；手動可指定 Pages 分支（非 master＝preview） |
| `deploy.yml` | master 上 `frontend/**` 變動、手動 | 用 state 分支上的上一版資料重新 build 部署，不跑 Python |
| `leakage.yml` | master push、PR | 洩漏回歸測試 |

## Secrets

`CLOUDFLARE_API_TOKEN`（Account → Cloudflare Pages → Edit）、`CLOUDFLARE_ACCOUNT_ID`。

## 常用操作

```bash
# 把雲端最新的線上實證資料庫拿回本機
git fetch github state && git show github/state:predictions.db > backend/predictions.db

# 本機產生靜態站預覽
cd backend && python -m stockta.export_static --out ../frontend/public/data
cd ../frontend && NEXT_PUBLIC_STATIC_DATA=1 npm run build && npx wrangler pages dev out

# 重訓換模型：上傳新的 Release（新 tag）→ 更新 backend/models.lock 的 tag 與 sha256
```

## 回復

- 某次執行把資料庫弄壞：到該次之前的 run 下載 `state-<run_id>` artifact，解開後覆寫 state 分支。
- 網站內容有誤：Cloudflare Pages 專案 → Deployments → 對上一版按 Rollback。
