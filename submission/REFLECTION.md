# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng ở run `attn_only`: khi nâng rank lên tận $r=283$ để bù tham số cho việc chỉ gắn vào 2 ma trận $q, v$, train loss giảm xuống rất thấp (0.5373, thấp hơn nhiều so với mức 0.6265 của cấu hình chuẩn `correct`), nhưng khi đo độ chính xác target thực tế thì `attn_only` chỉ đạt 0.9700 (hoà với `correct` chứ không hề vượt trội hơn). Điều này cho thấy train loss thấp chỉ là biểu hiện của việc học vẹt (overfit) cục bộ ở một nhóm tầng, và nó là một thước đo đánh lừa nếu cấu hình vị trí adapter bị lệch.


**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở khâu sinh văn bản autoregressive trong quá trình đánh giá (NB2 và NB5) khi phải decode tuần tự tập test qua nhiều mô hình/prompt đối chứng. Ban đầu tôi dự đoán giai đoạn backward pass huấn luyện LoRA ở NB3 và NB4 sẽ chiếm phần lớn thời gian, nhưng thực tế việc chạy inference nhiều lần để đánh giá 3 baseline tốn thời gian tương đương hoặc lâu hơn.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước đây tôi tin rằng: (1) cứ tăng rank $r$ càng cao thì mô hình càng thông minh; (2) mô hình sau fine-tune mặc định sẽ vượt trội hoàn toàn so với mô hình gốc. Sau lab này, tôi hiểu rằng vị trí gắn adapter (toàn bộ tầng tuyến tính) quan trọng hơn rank rất nhiều, và nếu base model được prompt tối ưu kèm few-shot tử tế thì bản fine-tune phải cấu hình chuẩn xác mới có thể thắng được.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI assistant để phân tích rubric, hỗ trợ theo dõi pipeline, debug lỗi encoding utf-8 trên console Windows (`scripts/verify.py`), và kiểm tra tính nhất quán giữa các artifact JSON. Điểm AI dễ mắc lỗi là thói quen ban đầu hay đề xuất xếp hạng các run dựa trên `final_loss` của NB4 thay vì bám sát điểm số trên tập Target ở NB5 theo đúng tiêu chuẩn rubric.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên là xây dựng một bộ dữ liệu đánh giá chuẩn (Golden Evaluation Set) hoàn toàn độc lập, sau đó thiết kế một bộ prompt tối ưu (Baseline b) để đo đạc trần năng lực của base model trước. Nếu prompt engineering đã đáp ứng đủ KPI và độ trễ thì chưa cần fine-tune; nếu quyết định fine-tune thì việc đầu tiên trong code là kiểm tra giải mã ngược loss mask để đảm bảo không bị rò rỉ prompt vào hàm mục tiêu.
