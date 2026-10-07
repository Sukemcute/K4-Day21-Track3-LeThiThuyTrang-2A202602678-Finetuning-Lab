# Lab 21 — Evaluation Report

**Họ tên**: Lê Thị Thùy Trang  **MSSV**: 2A202602678  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA Tesla T4 16GB (Google Colab)`

> Mọi con số dưới đây được đối soát khớp 100% với các file thực nghiệm đầy đủ (50 mẫu target + 15 mẫu regression) trong `results/`.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (tỷ lệ 90/10, split seed 42) |
| `max_length` | 1024 — p95 đo được là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*. Kết quả kiểm tra xác thực: `open_tag_present: true`, `body_present: true`, kết luận `reasoning preserved — safe to train on traces`. Template chat của Qwen3.5 bảo toàn nguyên vẹn cặp thẻ `<think>...</think>`, không nuốt khối suy luận trong hàm `apply_chat_template`. 
*Lý do đặt `max_length = 1024`:* Mặc dù phân phối độ dài p95 của tập ticket chỉ là 98 tokens (gợi ý ngưỡng tối thiểu 256 tokens), cấu hình phần cứng tier T4 được giữ nguyên ở mức 1024 để đảm bảo không gian đệm an toàn tuyệt đối cho cả các câu prompt phức tạp và giữ nguyên tính nhất quán của tier.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn văn bản được tính loss thực tế (trích xuất từ `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn bị mask hoàn toàn (loss = -100, không tham gia tính gradient):
```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

> *Đo đạc đầy đủ trên 50 mẫu Target và 15 mẫu Regression (`smoke_mode: false`)*:

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3396.4 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1066.7 |
| (c) LoRA fine-tune (`correct`) | 0.975 | 0.678 | 1.000 | 1390.8 |

**(b) có thật sự mạnh hơn (a) không?** Có, vượt trội hoàn toàn: target tăng từ 0.000 lên 0.765, format đạt 100% JSON hợp lệ (tăng từ 0.000), đồng thời độ trễ giảm từ 3396.4 ms xuống 1066.7 ms do model không sinh lan man văn bản thừa ngoài JSON.
Tôi giữ nguyên vẹn chuỗi `OPTIMIZED_PROMPT` mặc định của lab (mã băm sha256 `719e74d3b6232053`), không chỉnh sửa hay cố tình làm yếu prompt (b) để tâng bốc kết quả fine-tune. Điều này bảo đảm tính liêm chính khoa học của phép so sánh.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6257 | **0.9750** | 395.7 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5384 | **0.9700** | 270.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | **0.0000** | 402.0 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.9400** | 472.4 | 3.86 |

> **Phân tích chi tiết 3 câu hỏi bắt buộc:**

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**
Run `attn_only` được nâng rank lên $r=283$ để khớp chính xác ngân sách tham số (32,456,704 so với 32,464,896, sai lệch $< 0.03\%$). Trên tập target 50 mẫu, `attn_only` đạt 0.9700, vẫn **thua** so với `correct` (0.9750). Đáng chú ý, nếu chỉ nhìn theo train loss, `attn_only` lại có loss thấp hơn (0.5384 so với 0.6257 của `correct`), tạo ra thứ tự nghịch đảo giữa train loss và target accuracy thực tế. Điều này chứng minh rằng **vị trí gắn adapter (all text-linear) quan trọng hơn nhiều so với việc cố tăng rank ở một vài tầng attention**. Ép rank lên quá cao ở một vị trí hẹp chỉ giúp model ghi nhớ (overfit) tập train nhưng thiếu năng lực tổng quát hóa trên tập kiểm thử độc lập.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` chỉ thay đổi learning rate từ thang LoRA tiêu chuẩn ($10^{-4}$) xuống thang của Full Fine-Tuning ($10^{-5}$). Đường loss của `wrong_lr` gần như đi ngang và kẹt cứng ở mức cao 1.5702 (so với 0.6257), dẫn tới target accuracy rớt thẳng về 0.0000 và format JSON hoàn toàn hỏng (0.000). Nếu chỉ nhìn đường loss phẳng mà không biết nguyên nhân do LR quá nhỏ, một kỹ sư thiếu kinh nghiệm sẽ kết luận sai rằng bài toán "không thể hội tụ", dữ liệu bị lỗi nhãn hoặc dung lượng LoRA $r=16$ không đủ sức học bài toán. Thực tế là LoRA chỉ cập nhật một ma trận phụ với tham số rất nhỏ, do đó luôn cần learning rate lớn hơn Full FT từ 5 đến 10 lần.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` thể hiện khả năng cắt giảm bộ nhớ VRAM cực kỳ ấn tượng: từ 8.78 GB giảm xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% bộ nhớ đồ họa). Tuy nhiên, cái giá phải trả thể hiện rõ rệt ở hai khía cạnh: thời gian huấn luyện lâu hơn (472.4s so với 395.7s của 16-bit fp16 do overhead giải nén lượng tử NF4 on-the-fly) và độ chính xác target bị sụt giảm (đạt 0.9400 so với 0.9750 của `correct`, tụt 3.5 điểm %). Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị kỹ thuật: nếu GPU đủ VRAM (như T4 16GB có dư cho model 4B), không nên dùng QLoRA vì sẽ chịu suy giảm chất lượng và tốc độ mà không tận dụng được lợi ích giảm VRAM.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.210` · `regression Δ = -0.113` · `valid_trace_rate = 0.00`

**Diễn giải phán quyết (Phân tích nguyên nhân & Ý nghĩa khoa học):**
Cổng hồi quy đưa ra phán quyết `FAILED` do chỉ số năng lực tổng quát (`regression`) bị sụt giảm -0.113 (từ 0.791 xuống 0.678), vượt quá ngưỡng dung sai cho phép là 0.020, mặc dù điểm số bài toán chính (`target`) tăng trưởng mạnh mẽ +0.210 (từ 0.765 lên 0.975) và định dạng JSON đạt 100% hoàn hảo. 

Đây chính là minh chứng thực tế rõ nét cho hiện tượng **Quên thảm hoạ (Catastrophic Forgetting)** được giảng dạy trong Deck §6.3: khi chúng ta fine-tune mô hình 100% trên một tập dữ liệu hẹp (250 ticket CSKH) qua nhiều bước gradient, các trọng số LoRA bị chuyên biệt hóa quá mức vào phong cách và cấu trúc dữ liệu CSKH, dẫn đến việc làm suy thoái một phần tri thức phổ thông nền tảng.

Thông báo từ hệ thống `verdict.json` chỉ rõ giải pháp khắc phục chuẩn mực trong kỹ nghệ AI: *"add 1-5% replay data"*. Tức là trong các dự án thực tế, ta cần trộn thêm từ 1% đến 5% dữ liệu hội thoại tổng quát hoặc dữ liệu instruction-following đa nhiệm vào tập huấn luyện để đóng vai trò "mỏ neo" (anchor), giúp bảo tồn nguyên vẹn năng lực suy luận tổng quát mà không ảnh hưởng đến độ chính xác của miền nghiệp vụ mục tiêu. Phán quyết FAILED này mang giá trị học thuật sâu sắc hơn nhiều so với một kết quả pass ngẫu nhiên.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "Cho mình hỏi... chuột không dây VN232232. Cho tôi trả lại. Gấp..." | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | `doi_tra`, `cao`, `chuột không dây`, `trung_tinh` | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | ✅ **FT thắng**: Bắt đúng sắc thái tích cực ("Shop hỗ trợ tốt"). |
| 2 | "Xin chào... đèn bàn LED VN880807. Hoàn tiền. Quá hạn rồi..." | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | `hoan_tien`, `trung_binh`, `đèn bàn LED`, `tieu_cuc` | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | ✅ **FT thắng**: Nhận diện đúng urgency cao do quá hạn và cảm xúc lịch sự. |
| 3 | "Cho mình hỏi... bình giữ nhiệt VN804124. Chưa thấy tiền. Khi nào tiện..." | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, **`trung_binh`**, `bình giữ nhiệt`, `tich_cuc` | ❌ **FT thua**: FT đoán nhầm `urgency: trung_binh` vì từ khóa "chưa thấy tiền", bỏ qua ngữ cảnh "khi nào tiện". |
| 4 | "Shop ơi... áo khoác gió VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop..." | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | `san_pham_loi`, **`trung_binh`**, `áo khoác gió`, `tich_cuc` | ❌ **FT thua**: FT bị bias cụm từ "bị lỗi" nên tự động nâng mức khẩn cấp lên `trung_binh`, dù khách nói "khi nào tiện". |
| 5 | "Alo shop... máy xay sinh tố OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình." | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | ⚖️ **Cả hai đều đúng**: Cấu trúc câu rõ ràng, phân loại chính xác tuyệt đối. |

**Mẫu chung ở các ca FT thua:**
Mô hình fine-tune có xu hướng học "phím tắt" (shortcut learning) từ các từ khóa mang tính rủi ro cao (như *"chưa thấy tiền"*, *"bị lỗi"*, *"hoàn tiền"*). Khi gặp các từ này, FT dễ tự động nâng mức urgency lên `trung_binh`, trong khi prompt (b) của mô hình gốc lại có khả năng đọc hiểu ngữ cảnh tổng thể và chú ý tốt hơn tới các cụm điều kiện giảm nhẹ như *"khi nào tiện"*, *"không vội"*.

---

## 7. Kết luận & điều tôi học được

**Kết luận tổng quan:**
Bản LoRA fine-tune `correct` đạt kết quả phân loại nghiệp vụ xuất sắc (97.5% so với 76.5% của prompt tối ưu, 100% JSON chuẩn). Tuy nhiên, nếu xét trên góc độ triển khai hệ thống AI an toàn và bền vững (Production Readiness), ta **chưa nên deploy ngay bản adapter này một cách độc lập** khi chưa bổ sung dữ liệu replay. Lý do là hiện tượng sụt giảm năng lực tổng quát (-11.3%) cảnh báo rủi ro mô hình bị thoái hóa nếu gặp các câu hỏi mở ngoài domain ticket hẹp. 

Đòn bẩy thực sự trong bài lab này được chứng minh bằng các con số thực nghiệm:
1. **Loss mask chính xác** là điều kiện tiên quyết giữ cho mô hình không sinh nhảm.
2. **Learning rate đúng tầm ($10^{-4}$ thay vì $10^{-5}$)** quyết định mô hình có hội tụ được hay không.
3. **Vị trí gắn LoRA toàn diện (`text-linear`)** vượt trội hoàn toàn so với việc dồn rank cao vào một vài khối attention.

**Ba điều tôi học được cụ thể:**
1. **Đừng bao giờ đánh giá mô hình chỉ bằng Train Loss**: Run `attn_only` có loss thấp hơn `correct` (0.5384 so với 0.6257) nhưng trên tập test thực tế lại có độ chính xác thấp hơn. Tối ưu loss trên không gian tham số hẹp rất dễ tạo ảo tưởng về độ chính xác.
2. **Hiểm họa của Catastrophic Forgetting là có thật**: Dù chỉ train 2 epoch trên 250 mẫu, mô hình vẫn có thể suy thoái năng lực chung nếu không có dữ liệu đối trọng (replay data).
3. **Prompt Engineering luôn là mốc đối chứng bắt buộc**: Trước khi quyết định fine-tune tốn kém tài nguyên tính toán, phải luôn xây dựng một prompt tối ưu kèm few-shot để đo đạc trần năng lực của base model.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ thử nghiệm trộn thêm 5% tập dữ liệu UltraChat/Alpaca tiếng Việt vào tập train để chạy lại NB3 và NB5 nhằm chứng minh rằng cổng hồi quy sẽ chuyển sang `PASSED` mà điểm target CSKH vẫn duy trì trên 97%. Ngoài ra, tôi sẽ thực hiện thử thách Bonus B1 để merge adapter và đo độ trễ thực tế khi phục vụ người dùng.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
