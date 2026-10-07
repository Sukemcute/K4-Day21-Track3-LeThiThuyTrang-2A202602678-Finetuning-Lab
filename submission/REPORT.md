# Lab 21 — Evaluation Report

**Họ tên**: Lê Thị Thùy Trang  **MSSV**: 2A202602678  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `NVIDIA Tesla T4 16GB (Google Colab)`

> Mọi con số trong báo cáo này đều được em lấy trực tiếp từ các file kết quả thực nghiệm đầy đủ (50 mẫu target + 15 mẫu regression) trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 mẫu (chia tỷ lệ 90/10, cố định seed 42) |
| `max_length` | 1024 — phân phối p95 thực tế là 98 tokens *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*.
Em đã chạy kiểm tra và file log báo: `open_tag_present: true`, `body_present: true`, kết luận `reasoning preserved — safe to train on traces`. Tức là chat template mặc định của Qwen3.5 vẫn giữ nguyên cặp thẻ `<think>...</think>`, không bị lỗi tự động xóa mất phần suy luận khi gọi hàm `apply_chat_template`.

*Về việc chọn `max_length = 1024`:* Dữ liệu ticket trong bài khá ngắn, đo thực tế mức p95 chỉ có 98 tokens (gợi ý ngưỡng 256 tokens là đủ). Tuy nhiên, em giữ nguyên mức 1024 theo mặc định của tier T4 để trừ hao cho các câu prompt dài và không làm lệch cấu hình chuẩn của bài lab.

---

## 2. Mask proof (NB1)

| Tiêu chí | Kết quả kiểm tra |
|---|---|
| `supervised_fraction` | 0.4149 (khoảng 41.5% tổng số token) |
| Câu trả lời nằm trong loss | `true` (được tính loss) |
| Câu hỏi KHÔNG nằm trong loss | `true` (đã bị che đi) |

Đoạn văn bản thực tế mà mô hình được tính loss (trích từ `results/mask_proof.json`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Còn phần câu hỏi của người dùng và system prompt phía trước đã được che hoàn toàn (gán nhãn `-100` để không tính gradient):
```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

**Giải thích dễ hiểu:** Đoạn này chứng minh code của em đã che mask chuẩn: mô hình chỉ học cách "trả lời đúng định dạng JSON", chứ tuyệt đối không học vẹt lại câu hỏi của khách hàng. Tỷ lệ token được tính loss là 41.49%, nằm an toàn dưới ngưỡng 95%.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

> *Kết quả đo đạc trên toàn bộ 50 mẫu test và 15 câu hỏi tổng quát:*

| Run | target (độ chính xác) | regression (năng lực chung) | format (ra đúng JSON) | latency (độ trễ ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3396.4 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1066.7 |
| (c) LoRA fine-tune (`correct`) | 0.975 | 0.678 | 1.000 | 1390.8 |

**(b) có thật sự mạnh hơn (a) không?** Chắc chắn có, và vượt trội rõ rệt:
- Khi dùng prompt thô sơ (a), mô hình sinh chữ lan man, không ra nổi JSON chuẩn (format = 0.000), dẫn đến target accuracy = 0% và độ trễ rất cao (gần 3.4 giây/mẫu).
- Khi đổi sang prompt tối ưu (b) có hướng dẫn chi tiết và ví dụ mẫu: độ chính xác target nhảy vọt lên 76.5%, format đạt 100% JSON chuẩn, và thời gian sinh văn bản giảm còn khoảng 1 giây/mẫu.

Em giữ nguyên chuỗi `OPTIMIZED_PROMPT` gốc của bài lab (sha256 `719e74d3b6232053`), không chỉnh sửa hay cố tình làm yếu prompt đi để tâng bốc bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí gắn LoRA | Rank (r) | Số tham số train | Learning Rate | Train loss (NB4) | **Target accuracy (NB5)** | Thời gian train (s) | VRAM (GB) |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (tất cả các lớp) | 16 | 32,464,896 | 0.0001 | 0.6257 | **0.9750** | 395.7 | 8.78 |
| `attn_only` | chỉ gắn q, v | 283 *(bù tham số)* | 32,456,704 | 0.0001 | 0.5384 | **0.9700** | 270.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | **0.0000** | 402.0 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.9400** | 472.4 | 3.86 |

> **Trả lời 3 câu hỏi phân tích (dễ hiểu, bám sát số liệu):**

**4.1 — `attn_only` có cùng số tham số với `correct`. Trên tập target nó thắng hay thua? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói lên điều gì về rank so với vị trí gắn adapter?**
Để so sánh công bằng, run `attn_only` đã được tăng rank lên tận $r=283$ để có cùng khoảng 32.4 triệu tham số như `correct`. Kết quả: trên tập target thực tế, `attn_only` đạt 97.0%, vẫn **thua** so với mức 97.5% của `correct`. 

Điều thú vị nhất là nếu chỉ nhìn điểm số khi đang train (train loss), `attn_only` lại có loss thấp hơn hẳn (0.5384 so với 0.6257). Tức là train loss thấp hơn nhưng khi đi thi thật thì điểm lại kém hơn! Điều này chứng minh: **Vị trí gắn adapter quan trọng hơn việc cắm đầu tăng rank**. Việc rải đều rank nhỏ ($r=16$) vào tất cả các tầng (cả Attention lẫn MLP) giúp mô hình tiếp thu kiến thức toàn diện, trong khi dồn hết rank cực lớn vào một chỗ hẹp ($q, v$) chỉ làm mô hình học vẹt tập train mà kém linh hoạt khi gặp dữ liệu mới.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Run `wrong_lr` bị chỉnh hạ Learning Rate xuống 10 lần ($10^{-5}$ thay vì $10^{-4}$ - đây là mức LR người ta hay dùng cho Full Fine-tuning chứ không phải LoRA). Hậu quả là đường loss gần như đi ngang, kẹt cứng ở mức cao 1.5702 (trong khi bản chuẩn giảm mượt mà về 0.6257). Đến lúc test, độ chính xác rớt thẳng về 0% và mô hình không thể xuất ra JSON hợp lệ.

Nếu một người mới nhìn vào đường loss đi ngang này mà không biết nguyên nhân do LR quá bé, họ sẽ dễ kết luận sai lầm rằng: "Bài toán này khó quá mô hình không học nổi", hoặc "Dữ liệu bị gắn nhãn sai bét", hoặc "LoRA $r=16$ dung lượng quá nhỏ không tải nổi". Thực tế là LoRA chỉ cập nhật một lượng tham số rất nhỏ, nên mỗi bước nhảy (LR) bắt buộc phải lớn hơn Full Fine-tune từ 5 đến 10 lần thì mới học được.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**
Số liệu thực tế cho thấy QLoRA tiết kiệm VRAM cực kỳ ấn tượng: giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% bộ nhớ đồ họa, máy cấu hình thấp vẫn chạy được). Tuy nhiên, cái giá phải trả thể hiện rất rõ ở 2 điểm:
1. **Train chậm hơn**: Mất 472.4 giây so với 395.7 giây của bản 16-bit, vì GPU phải tốn công nén và giải nén lượng tử 4-bit liên tục trong lúc tính toán.
2. **Độ chính xác bị tụt**: Điểm target giảm từ 97.5% xuống 94.0% (mất 3.5 điểm %).

Kết quả này hoàn toàn ủng hộ khuyến nghị kỹ thuật: Nếu phần cứng của bạn đã đủ VRAM (ví dụ card T4 có sẵn 16GB, thừa sức chứa model 4B ở chuẩn fp16) thì **không nên dùng QLoRA**, vì ta sẽ bị thiệt cả về tốc độ lẫn độ chính xác mà không tận dụng được lợi ích tiết kiệm VRAM.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.210` · `regression Δ = -0.113` · `valid_trace_rate = 0.00`

**Diễn giải phán quyết (Vì sao FAILED và bài học rút ra):**
Cổng kiểm tra tự động trả về kết quả `FAILED` không phải vì mô hình học kém bài toán chính, mà vì mô hình bị hiện tượng **sụt giảm năng lực tổng quát** (điểm `regression` trên 15 câu hỏi kiến thức phổ thông bị tụt mất -0.113, từ 79.1% xuống 67.8%, vượt ngưỡng dung sai cho phép là 0.02). Trong khi đó, ở bài toán phân loại ticket CSKH chính, mô hình thể hiện cực kỳ ấn tượng khi tăng trưởng +21.0% so với prompt tối ưu (từ 76.5% lên 97.5%) và sinh JSON chuẩn 100%.

Hiện tượng này trong ngành gọi là **Quên thảm họa (Catastrophic Forgetting)**, như đã được thầy giảng ở Deck mục §6.3: Khi chúng ta đem một mô hình ngôn ngữ lớn đi fine-tune 100% trên một tập dữ liệu chuyên biệt và nhỏ hẹp (250 mẫu CSKH), các trọng số LoRA bị cuốn hoàn toàn theo lối hành văn và từ vựng CSKH, vô tình làm mòn đi một phần tri thức phổ thông mà mô hình từng học ở giai đoạn pre-train. 

Ngay trong thông báo lỗi của hệ thống cũng đã chỉ ra giải pháp khắc phục rất rõ: *"add 1-5% replay data"*. Tức là trong thực tế, chỉ cần ta trộn thêm một tỷ lệ nhỏ (tầm 1% đến 5%) các câu hỏi đố vui hoặc dữ liệu đàm thoại chung vào tập train, mô hình sẽ giữ được "trí nhớ nền" mà vẫn phân loại ticket chuẩn xác. Việc nhìn thấy và giải thích được kết quả FAILED này giúp em hiểu sâu sắc hơn nhiều về bản chất của fine-tuning.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket của khách (rút gọn) | Nhãn chuẩn | (b) Prompt tối ưu đoán | (c) Fine-tune đoán | Nhận xét chi tiết |
|---|---|---|---|---|---|
| 1 | "Cho mình hỏi... chuột không dây VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt." | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | `doi_tra`, `cao`, `chuột không dây`, **`trung_tinh`** | `doi_tra`, `cao`, `chuột không dây`, **`tich_cuc`** | ✅ **FT thắng**: Fine-tune bắt đúng lời khen cuối câu ("Shop hỗ trợ tốt") nên nhận diện đúng cảm xúc tích cực. |
| 2 | "Xin chào... đèn bàn LED VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều." | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | `hoan_tien`, **`trung_binh`**, `đèn bàn LED`, **`tieu_cuc`** | `hoan_tien`, **`cao`**, `đèn bàn LED`, **`tich_cuc`** | ✅ **FT thắng**: Bắt đúng mức độ khẩn cấp cao do hàng quá hạn và nhận ra khách vẫn nói cảm ơn rất lịch sự. |
| 3 | "Cho mình hỏi... bình giữ nhiệt VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều." | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, **`thap`**, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, **`trung_binh`**, `bình giữ nhiệt`, `tich_cuc` | ❌ **FT thua**: Fine-tune đoán nhầm khẩn cấp `trung_binh` vì thấy chữ "chưa thấy tiền", bỏ qua chữ "khi nào tiện". |
| 4 | "Shop ơi... áo khoác gió VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều." | `san_pham_loi`, `thap`, `áo khoác gió`, `tich_cuc` | `san_pham_loi`, **`thap`**, `áo khoác gió`, `tich_cuc` | `san_pham_loi`, **`trung_binh`**, `áo khoác gió`, `tich_cuc` | ❌ **FT thua**: Thấy chữ "bị lỗi" là FT vội vàng nâng lên mức `trung_binh`, trong khi khách nhắn rất thoải mái "khi nào tiện". |
| 5 | "Alo shop... máy xay sinh tố OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình." | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `doi_tra`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | ⚖️ **Cả hai đều đúng**: Câu cú rõ ràng, cả hai bên đều bắt chuẩn 100% các trường. |

**Mẫu số chung ở các ca mà Fine-tune bị THUA:**
Mô hình fine-tune có xu hướng "học đường tắt" (shortcut learning). Cứ mỗi lần nhìn thấy các cụm từ tiêu cực hoặc khiếu nại (như *"chưa thấy tiền"*, *"bị lỗi"*, *"hoàn tiền"*), mô hình sẽ tự động gán mức độ khẩn cấp là `trung_binh` trở lên. Trong khi đó, mô hình gốc khi kết hợp với Prompt tối ưu (b) lại đọc hiểu câu một cách tổng thể hơn, nhận ra được các cụm từ giảm nhẹ ở cuối câu như *"khi nào tiện"*, *"không vội"* để chọn đúng mức khẩn cấp là `thap`.

---

## 7. Kết luận & điều tôi học được

**Kết luận thực tế:**
Nếu hỏi em có nên mang bản mô hình fine-tune này đi triển khai cho khách hàng ngay ngày mai không, câu trả lời của em là: **Chưa nên triển khai ngay lập tức**. 

Mặc dù mô hình đạt độ chính xác phân loại ticket rất cao (97.5% so với 76.5% của prompt tối ưu, 100% JSON chuẩn), nhưng việc năng lực tổng quát bị sụt giảm -11.3% là một rủi ro lớn. Nếu khách hàng nhắn những câu hỏi nằm ngoài phạm vi hẹp của dữ liệu ticket CSKH, mô hình có thể phản hồi ngớ ngẩn. Để đưa vào production an toàn, bước tiếp theo em cần làm là bổ sung 3-5% dữ liệu đàm thoại chung vào để train lại, giúp khắc phục lỗi quên bài cũ.

Qua toàn bộ bài lab, em thấy rõ đòn bẩy lớn nhất trong fine-tuning không phải là cố tăng rank LoRA thật to, mà nằm ở: (1) **Che loss mask đúng** để mô hình không học vẹt prompt; (2) **Chọn đúng thang Learning Rate** ($10^{-4}$ cho LoRA); và (3) **Gắn adapter phủ đều các tầng (`text-linear`)** thay vì chỉ gắn cục bộ ở attention.

**Ba điều cụ thể em học được:**
1. **Đừng bao giờ tin tưởng tuyệt đối vào Train Loss**: Run `attn_only` có loss thấp hơn hẳn bản `correct`, nhưng lúc đi thi trên tập test thật thì lại đạt điểm thấp hơn. Loss giảm sâu ở một vị trí hẹp chỉ là biểu hiện của overfit.
2. **Luôn đo Baseline trước khi bấm nút Train**: Muốn biết fine-tune có đáng tiền và đáng công sức không, bắt buộc phải viết một cái prompt thật xịn kèm ví dụ mẫu để đo xem base model làm được đến đâu. Nếu prompt xịn đã giải quyết được 80-90% bài toán thì chưa chắc đã cần fine-tune.
3. **Hiện tượng quên tri thức cũ là có thật và diễn ra rất nhanh**: Chỉ mới train 2 epoch trên vỏn vẹn 250 mẫu dữ liệu mà mô hình đã bắt đầu suy giảm các kỹ năng trả lời câu hỏi thông thường.

**Nếu có thêm 2 giờ nữa, em sẽ thử:**
Em sẽ thử nghiệm trộn thêm 5% dữ liệu hội thoại tiếng Việt thông thường vào tập train để chạy lại NB3 và NB5, nhằm chứng minh rằng cổng hồi quy sẽ chuyển từ `FAILED` sang `PASSED` mà độ chính xác ticket vẫn giữ vững trên 97%. Ngoài ra, em sẽ chạy thêm notebook NB6 để thử merge adapter trực tiếp vào model gốc và đo xem tốc độ phản hồi có nhanh hơn không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
