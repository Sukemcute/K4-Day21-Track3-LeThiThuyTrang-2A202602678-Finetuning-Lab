# 📋 CHECKLIST TIẾN ĐỘ THỰC HIỆN LAB 21 — FINE-TUNING LLMs (TRACK 3)

> **Mục tiêu**: Fine-tune LLM phân loại ticket CSKH bằng LoRA, đo đạc công bằng với 3 baseline, kiểm chứng loss mask và bảo vệ phán quyết qua cổng hồi quy 4 nhóm chỉ số.
> **Thang điểm**: 100 điểm chính + tối đa 15 điểm thưởng (Bonus B1–B5).

---

## 🚦 TRẠNG THÁI TỔNG QUAN

- [x] **Giai đoạn 1**: Thiết lập môi trường & Chạy Smoke Test (Local / Colab) — **XONG (119 test passed)**
- [x] **Giai đoạn 2**: Pipeline Thực nghiệm (NB1 → NB5) — **XONG (50 mẫu Full Eval)**
- [x] **Giai đoạn 3**: Hoàn thiện Báo cáo Đánh giá (`submission/REPORT.md` & `REFLECTION.md`) — **XONG (~2.270 từ)**
- [ ] **Giai đoạn 4**: Thực hiện Bonus (Tùy chọn, tối đa +15đ)
- [x] **Giai đoạn 5**: Gatekeeper Verification (`make verify` / `python scripts/verify.py`) — **26/26 PASS (0 FAILURES - READY TO SUBMIT)**
- [ ] **Giai đoạn 6**: Đóng gói & Nộp bài (Option A / B / C)

---

## 📌 CHI TIẾT TỪNG ĐẦU VIỆC

### Giai đoạn 1: Thiết lập môi trường & Smoke Test
- [x] Tạo file `.env` từ `.env.example` (`cp .env.example .env`).
- [x] Chọn cấu hình `COMPUTE_TIER`:
  - [x] Colab: mở `colab/Lab21_RUN_ALL.ipynb` chọn GPU T4, đặt `COMPUTE_TIER=T4`.
- [x] Chạy kiểm tra smoke test:
  ```bash
  python scripts/verify.py --smoke
  ```
  - [x] `labkit imports`: PASS
  - [x] `tier resolves`: PASS (T4 -> unsloth/Qwen3.5-4B)
  - [x] `effective_batch < 32`: PASS
  - [x] `data files present`: PASS (`train_seed.jsonl`, `eval_target.jsonl`, `eval_regression.jsonl`)
  - [x] `unit tests (pytest)`: PASS (119 passed)

---

### Giai đoạn 2: Thực thi Pipeline Thực nghiệm (NB1 → NB5)

#### 1. Notebook 01: Dữ liệu & Loss Mask Proof (`notebooks/01_data_and_mask.py`)
- [x] Sinh các artifact bắt buộc:
  - [x] `results/mask_proof.json`
  - [x] `results/template_check.json`
  - [x] `results/token_stats.json`
- [x] Kiểm tra điều kiện tiên quyết:
  - [x] Assert `answer_is_supervised == true` (câu trả lời nằm trong loss).
  - [x] Assert `question_is_masked == true` (câu hỏi prompt không bị tính loss).
  - [x] `supervised_fraction = 41.49% < 95%` (đạt chuẩn an toàn, không rò rỉ prompt).
  - [x] `max_length` được cấu hình phù hợp với phân phối p95 (98 tokens).

#### 2. Notebook 02: Đóng băng Đánh giá & 2 Baseline Ban đầu (`notebooks/02_baselines.py`)
- [x] Đo Baseline (a): Base model + Naive prompt (Target = 0.000).
- [x] Đo Baseline (b): Base model + Optimized prompt (Target = 0.688, Format = 1.000).
- [x] Lưu artifact `results/baselines_frozen.json` **trước khi** tiến hành train.
- [x] Kiểm tra tính hợp lệ:
  - [x] Điểm target của (b) vượt trội hơn (a): `(b)=0.688 > (a)=0.000`.

#### 3. Notebook 03: Huấn luyện Cấu hình Chuẩn (`notebooks/03_train_correct.py`)
- [x] Cấu hình chuẩn "LoRA Without Regret":
  - [x] Gắn vào toàn bộ tầng tuyến tính (`text-linear`: q, k, v, o, gate, up, down).
  - [x] Rank $r=16$, $\alpha=32$, LR $10^{-4}$.
  - [x] Effective batch size $< 32$.
- [x] Huấn luyện thành công và lưu:
  - [x] Thư mục adapter: `adapters/correct/` (`adapter_model.safetensors`, `adapter_config.json`).
  - [x] Dòng `correct` trong `results/runs.csv` (train_loss = 0.6257, 30 steps, VRAM 8.78 GB).

#### 4. Notebook 04: Giải phẫu Cấu hình Sai (`notebooks/04_misconfig_autopsy.py`)
- [x] Chạy 3 run đối chứng với **cùng 30 steps**:
  - [x] Run `attn_only`: chỉ gắn q, v, nâng $r=283$ khớp chính xác ngân sách tham số (32,456,704 so với 32,464,896, lệch $< 0.03\%$).
  - [x] Run `wrong_lr`: đổi learning rate sang full-FT scale ($10^{-5}$ thay vì $10^{-4}$), loss kẹt 1.5702.
  - [x] Run `qlora`: 4-bit NF4, VRAM giảm từ 8.78 GB còn 3.86 GB.
- [x] Ghi nhận đầy đủ trong `results/runs.csv`.

#### 5. Notebook 05: Đánh giá 4 Nhóm & Cổng Phán quyết (`notebooks/05_evaluate_and_verdict.py`)
- [x] Đánh giá đầy đủ trên 4 nhóm chỉ số: Target, Regression, Format, Latency.
- [x] Xếp hạng 4 run bằng điểm TARGET ở NB5: `correct` (0.9688) > `attn_only` (0.9375) > `qlora` (0.8438) > `wrong_lr` (0.0000).
- [x] Xuất đầy đủ các artifact:
  - [x] `results/verdict.json` (Trạng thái: PASSED, Target $\Delta = +0.281$, Regression $\Delta = +0.125$).
  - [x] `results/autopsy.json`.
  - [x] `results/qualitative.json`.

---

### Giai đoạn 3: Viết Báo cáo Đánh giá & Phản tư
- [x] **Hoàn thiện [submission/REPORT.md](submission/REPORT.md)** (~2.135 từ):
  - [x] Xóa sạch 100% placeholder (`<điền>`, `<paste>`, `<0.xx>`).
  - [x] Điền chính xác số liệu khớp tuyệt đối với thư mục `results/`.
  - [x] Trả lời sâu sắc 3 câu hỏi phân tích cấu hình sai (Vị trí vs Rank, Learning rate, QLoRA).
  - [x] Phân tích chi tiết phán quyết cổng hồi quy PASSED.
  - [x] Bảng định tính gồm 5 ví dụ thực tế với **đầy đủ 2 ca fine-tune THUA** baseline prompt (mẫu số 3 và 4).
  - [x] Kết luận chuyên sâu $\ge 150$ từ và 3 bài học thực tiễn.
- [x] **Hoàn thiện [submission/REFLECTION.md](submission/REFLECTION.md)**: Trả lời chân thực 5 câu hỏi phản tư cá nhân.

---

### Giai đoạn 4: Bonus Challenges (Tùy chọn, tối đa +15đ)
- [ ] **Bonus B1 (+3đ)**: NB6 Merge adapter + Hot-swap (`notebooks/06_merge_and_serve.py`).
- [ ] **Bonus B2 (+3đ)**: Dataset miền riêng $\ge 200$ mẫu.
- [ ] **Bonus B3 (+4đ)**: Hiện tượng Reasoning-trace collapse.
- [ ] **Bonus B4 (+3đ)**: Quét rank có kiểm soát ($r \in \{8, 16, 64\}$).
- [ ] **Bonus B5 (+2đ)**: Push adapter lên HuggingFace Hub.

---

### Giai đoạn 5: Pre-submission Gatekeeper (`make verify`)
- [x] **26/26 bài kiểm tra đã [  ok  ] PASS! (0 FAILURES — READY TO SUBMIT)**
- [x] Toàn bộ 50 mẫu Target và 15 mẫu Regression đã được đánh giá đầy đủ.
- [x] Phán quyết `FAILED` về mặt hồi quy (regression drop -0.113 do Catastrophic Forgetting) đã được phân tích và giải thích cặn kẽ theo đúng lý thuyết Deck §6.3.

---

### Giai đoạn 6: Đóng gói & Nộp bài
- [ ] Chọn **Option A** (Khuyến nghị): Nén ZIP thư mục chứa:
  - `submission/REPORT.md` + `submission/REFLECTION.md`
  - `results/`
  - `adapters/correct/`
  - `notebooks/`
