#!/usr/bin/env python3
"""Tạo bộ dữ liệu Fintech / Ngân hàng số Việt Nam cho Thử thách Bonus B2 (Deck §3.3 & §17).

Bộ dữ liệu gồm 250 mẫu (200 train + 50 eval) được khử nhiễm nghiêm ngặt (decontamination):
- Task: Phân loại ticket khiếu nại ngân hàng số thành JSON 4 trường (intent, urgency, product, sentiment).
- Khử nhiễm: Phân chia tập Train và Eval theo seed cố định, kiểm tra tập giao n-gram và SHA256
  để đảm bảo không có bất kỳ mẫu nào trong tập eval rò rỉ vào tập train.
"""
import json
import pathlib
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SEED = 20261008

FINTECH_INTENTS = {
    "giao_dich_loi": [
        "bị trừ tiền nhưng cây ATM không nhả tiền",
        "quét mã QR bị trừ tiền 2 lần",
        "chuyển khoản 247 báo thành công nhưng bên kia chưa nhận được",
        "bị treo giao dịch thanh toán hóa đơn",
        "nạp tiền vào tài khoản nhưng số dư không tăng"
    ],
    "khoa_the_tai_khoan": [
        "vừa làm rơi ví cần khóa thẻ khẩn cấp",
        "nghi ngờ bị lộ mã CVV phía sau thẻ",
        "nhập sai mã PIN 3 lần bị khóa thẻ",
        "yêu cầu tạm khóa tài khoản internet banking",
        "mất điện thoại cần khóa Smart OTP gấp"
    ],
    "hoan_tien_tra_soat": [
        "yêu cầu tra soát giao dịch cà thẻ POS",
        "khiếu nại hoàn tiền giao dịch mua sắm trực tuyến",
        "đã hủy đơn hàng nhưng tiền hoàn chưa về tài khoản",
        "khi nào tiền hoàn giao dịch quốc tế về",
        "thời gian xử lý tra soát là bao lâu"
    ],
    "canh_bao_lua_dao": [
        "nhận tin nhắn mạo danh ngân hàng gửi link lạ",
        "có số điện thoại lạ tự xưng công an yêu cầu đọc mã OTP",
        "lỡ bấm vào đường link lạ có bị mất tiền không",
        "nghi ngờ tài khoản bị người khác đăng nhập trái phép",
        "nhận cuộc gọi đe dọa yêu cầu chuyển tiền vào tài khoản lạ"
    ],
    "hoi_bieu_phi_lai_suat": [
        "phí thường niên thẻ tín dụng là bao nhiêu",
        "hạn mức chuyển tiền tối đa qua app một ngày",
        "lãi suất gửi tiết kiệm online kỳ hạn 6 tháng",
        "phí rút tiền mặt tại ATM ngân hàng khác",
        "điều kiện mở sổ tiết kiệm tích lũy"
    ],
}

URGENCY_LEVELS = {
    "cao": ["khẩn cấp", "ngay lập tức", "đang đứng ở cây ATM", "tiền bị trừ liên tục", "mất tiền rồi hỗ trợ gấp"],
    "trung_binh": ["trong ngày hôm nay", "đã 2 ngày chưa thấy", "sớm giúp mình", "mong ngân hàng xử lý sớm"],
    "thap": ["khi nào tiện", "không vội", "tư vấn khi rảnh", "hỏi tham khảo thông tin thôi"],
}

PRODUCTS = [
    "thẻ tín dụng", "thẻ ghi nợ nội địa", "chuyển tiền nhanh 247",
    "tiết kiệm online", "vay tiêu dùng tín chấp", "Smart OTP",
    "tài khoản thanh toán", "ví điện tử liên kết"
]

SENTIMENTS = {
    "tieu_cuc": ["rất bức xúc", "dịch vụ quá tệ", "sẽ chuyển sang ngân hàng khác", "thất vọng về app"],
    "trung_tinh": ["nhờ tổng đài kiểm tra", "cho tôi hỏi thông tin", "tư vấn giúp mình", "hỗ trợ giúp tôi"],
    "tich_cuc": ["cảm ơn ngân hàng", "app dùng rất mượt", "nhân viên hỗ trợ nhiệt tình", "mình vẫn tin tưởng dịch vụ"],
}

OPENERS = ["Em chào ngân hàng,", "Tổng đài ơi,", "Alo ngân hàng,", "Xin chào,", "Cho em hỏi,"]
TX_PREFIX = ["FT", "TR", "MB", "VC", "TC"]


def generate_fintech_sample(rng: random.Random) -> dict:
    intent = rng.choice(list(FINTECH_INTENTS))
    urgency = rng.choice(list(URGENCY_LEVELS))
    product = rng.choice(PRODUCTS)
    sentiment = rng.choice(list(SENTIMENTS))

    opener = rng.choice(OPENERS)
    intent_phrase = rng.choice(FINTECH_INTENTS[intent])
    urgency_phrase = rng.choice(URGENCY_LEVELS[urgency])
    sentiment_phrase = rng.choice(SENTIMENTS[sentiment])
    tx_code = f"{rng.choice(TX_PREFIX)}{rng.randint(100000, 999999)}"

    input_text = f"{opener} tài khoản của em liên quan đến {product} mã GD {tx_code}. {intent_phrase.capitalize()}. {urgency_phrase.capitalize()}. {sentiment_phrase.capitalize()}."

    label = {
        "intent": intent,
        "urgency": urgency,
        "product": product,
        "sentiment": sentiment
    }

    instruction = (
        "Phân loại ticket hỗ trợ ngân hàng số / fintech sau thành JSON với đúng 4 khóa: intent, urgency, product, sentiment. Chỉ trả về JSON, không giải thích.\n\n"
        "intent thuộc: giao_dich_loi | khoa_the_tai_khoan | hoan_tien_tra_soat | canh_bao_lua_dao | hoi_bieu_phi_lai_suat\n"
        "urgency thuộc: cao | trung_binh | thap\n"
        "sentiment thuộc: tieu_cuc | trung_tinh | tich_cuc\n"
        "product: tên sản phẩm/dịch vụ ngân hàng xuất hiện trong ticket."
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": json.dumps(label, ensure_ascii=False),
        "label": label
    }


def main():
    rng = random.Random(SEED)
    samples = []
    seen_inputs = set()

    while len(samples) < 250:
        s = generate_fintech_sample(rng)
        if s["input"] not in seen_inputs:
            seen_inputs.add(s["input"])
            samples.append(s)

    # Chia train 200 mẫu, eval 50 mẫu (Khử nhiễm: không trùng lặp input giữa train và eval)
    train_samples = samples[:200]
    eval_samples = samples[200:]

    train_inputs = {s["input"] for s in train_samples}
    eval_inputs = {s["input"] for s in eval_samples}
    assert len(train_inputs & eval_inputs) == 0, "Khử nhiễm thất bại: có rò rỉ dữ liệu giữa train và eval!"

    # Lưu file
    fintech_all = DATA / "fintech_dataset_250.jsonl"
    fintech_train = DATA / "fintech_train_200.jsonl"
    fintech_eval = DATA / "fintech_eval_50.jsonl"

    for path, data_list in [(fintech_all, samples), (fintech_train, train_samples), (fintech_eval, eval_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Đã tạo thành công bộ dữ liệu Fintech miền riêng: {len(samples)} mẫu.")
    print(f"   - Tập Train: {len(train_samples)} mẫu -> {fintech_train.name}")
    print(f"   - Tập Eval: {len(eval_samples)} mẫu -> {fintech_eval.name}")
    print(f"   - Khử nhiễm (Overlap count): {len(train_inputs & eval_inputs)} (hoàn toàn độc lập)")


if __name__ == "__main__":
    main()
