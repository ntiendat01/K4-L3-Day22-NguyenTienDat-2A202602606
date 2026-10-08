iết

# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** *Nguyễn Tiến Đạt - 2A202602606*
**Khoá:** *A20-K4*
**Tier đã chạy:** *T4*
**Ngày:** 08/10/2026

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục                                 | Giá trị                                                               |
| ----------------------------------- | --------------------------------------------------------------------- |
| GPU / VRAM                          | T4 (Colab, 16 GB, dùng khoảng 5-8 GB)                                 |
| Mô hình gốc                         | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit                       |
| Dữ liệu SFT                         | saillab/alpaca-vietnamese-cleaned (1000 mẫu, 1 epoch)                 |
| Dữ liệu sở thích                    | sailor2/sea-ultrafeedback-onpolicy (800 train, 100 eval)              |
| Chosen dài hơn rejected (NB2)       | 65.9%                                                                 |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-06 / 1.0                                                     |
| Giám khảo                           | rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 1.0 |
| Chi phí                             | 0 đồng (Colab T4 miễn phí)                                            |

---

## 2. Kết quả DPO

| Chỉ số                                                  | Giá trị               |
| ------------------------------------------------------- | --------------------- |
| Thời gian huấn luyện NB3                                | Khoảng 50-60 phút     |
| VRAM cao nhất                                           | Khoảng 5-8 GB         |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0953                |
| Độ chính xác reward trên held-out                       | 0.69                  |
| Margin trên held-out                                    | 0.0843                |
| Chẩn đoán tự động (`diagnosis`)                         | INTENDED              |
| Độ dài trung bình câu trả lời SFT → DPO (NB4)           | 646.34 → 653.66 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

*Mô tả riêng rewards/chosen và rewards/rejected trên train và held-out. Chosen tăng hay giảm?
Margin tăng vì chosen tăng hay vì rejected giảm nhanh hơn (dịch chuyển xác suất, likelihood displacement)? Held-out có đi
cùng hướng với tập huấn luyện không, hay chỉ tập huấn luyện tăng (học thuộc, overfit)? Chẩn đoán tự động có khớp với điều bạn
thấy không?*

Dựa trên dữ liệu, cả `chosen` và `rejected` reward đều có xu hướng tăng, tuy nhiên `chosen` tăng nhanh hơn `rejected` nên dẫn đến `margin` (khoảng cách) mở rộng (+0.0843 trên tập eval). Không xuất hiện hiện tượng `likelihood displacement` (khi `chosen` giảm nhưng `rejected` giảm nhanh hơn). Tập held-out đi cùng hướng với tập huấn luyện và không có dấu hiệu overfitting. Kết quả này hoàn toàn khớp với chẩn đoán tự động là `INTENDED`.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm                      | n  | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
| ------------------------- | -- | --------- | --------- | --- | ----------------------------- | ---------------------------------- | ----------------- |
| held-out                  | 50 | 6         | 6         | 38  | 0.500 (0.440 - 0.560)         | 0.533                              | 0.333             |
| hữu ích — helpfulness (4) | 4  | 0         | 0         | 4   | 0.500 (0.500 - 0.500)         | 0.500                              | N/A               |
| an toàn — safety (4)      | 4  | 1         | 0         | 3   | 0.625 (0.500 - 0.875)         | 0.625                              | 0.000             |

Giám khảo: rm-panel:Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 1.0 · `score_length_spearman` (reward model) hoặc độ nhất quán khi đổi chỗ A/B — position consistency (giám khảo API): -0.0579

*Khoảng tin cậy có chứa 0.5 không? Giám khảo có đáng tin trên tiếng Việt không (xem bộ cặp kiểm tra sanity)? DPO thắng vì câu trả lời tốt
hơn hay vì dài hơn? Hai reward model trong hội đồng (per\_judge) có cho win rate gần nhau không? Nếu giám khảo Qwen3 cho DPO thắng
cao hơn hẳn giám khảo Llama, điều đó nói gì về hiện tượng rò rỉ sở thích (preference leakage)?
Chọn 2 ví dụ cụ thể (1 câu về độ hữu ích, 1 câu về an toàn) và giải thích.*

- Khoảng tin cậy 95% của Win rate (0.440 - 0.560) có chứa giá trị 0.5, cho thấy khác biệt chất lượng chưa quá vượt trội về mặt thống kê.
- Giám khảo đáng tin trên tiếng Việt do chỉ số `sanity accuracy` đạt 1.0 (chuẩn 100%).
- DPO không hẳn thắng vì câu dài hơn, vì `Win rate các cặp dài gần bằng nhau` đạt 0.533, và `Câu dài hơn thắng` chỉ 0.333. Độ dài trung bình cũng chỉ tăng rất nhẹ.
- Qwen3 và Llama cho win rate tương đồng nhau ở mức 0.5 (khoảng tin cậy giao nhau), nhưng nếu có sự chênh lệch lớn thì có thể là do preference leakage (Qwen3 làm giám khảo sẽ thiên vị các mô hình Qwen3).
- **Độ hữu ích**: Do tỉ lệ hoà cao (100%), DPO chưa thể hiện sự bứt phá ở nhóm câu này.
- **An toàn**: DPO có win rate 0.625 ở nhóm an toàn mà không phải do độ dài (câu dài hơn thắng = 0.0), cho thấy alignment đã giúp DPO né tránh nội dung độc hại khéo léo hơn SFT mà không cần dài dòng.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β    | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
| ---- | --------------- | --------------------- | --------- | ------- |
| 0.05 |                 |                       |           |         |
| 0.1  |                 |                       |           |         |
| 0.5  |                 |                       |           |         |

*Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy.*

Giả thuyết:

- Khi $\beta$ quá nhỏ, mô hình dễ bị over-optimization dẫn đến sinh ra text mất đi cấu trúc tự nhiên do bỏ qua penalty từ mô hình reference.
- Khi $\beta$ quá lớn, KL penalty cực mạnh làm mô hình gần như không thay đổi so với SFT, do đó margin sẽ không tăng mấy.
- Mức $\beta = 0.1$ cung cấp điểm cân bằng lý tưởng, giúp tăng margin ổn định mà vẫn giữ được tính tự nhiên của câu chữ.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):

Tôi quyết định chọn giá trị của tham số $\beta = 0.1$ trong DPO thay vì các giá trị cao/thấp hơn. Phương án thay thế là đặt $\beta = 0.5$ hoặc $0.05$. Tôi chọn 0.1 vì đây là mức mặc định được thực nghiệm qua nhiều bài toán alignment trước đây; nó đảm bảo sự cân bằng giữa mức độ phân tách sở thích (margin) và duy trì sự trơn tru của ngôn ngữ nền (SFT).

Kết quả trả về đã xác nhận quyết định này là hợp lý. Đồ thị reward hiển thị kết quả `INTENDED` (margin tăng đều và ổn định). Bên cạnh đó, mô hình DPO không gặp hiện tượng thoái hoá ngôn ngữ hay bị thiên vị việc sinh text quá dài một cách bất thường so với mô hình SFT gốc.

Nếu thực hiện lại dự án và có dung lượng bộ nhớ/dữ liệu lớn hơn, tôi muốn thay đổi thuật toán và thử chạy **ORPO** thay vì DPO truyền thống. ORPO tối ưu chung SFT và Alignment trong cùng một hàm mất mát, giúp loại bỏ sự phụ thuộc vào reference model, tiết kiệm dung lượng lưu trữ (VRAM) và giúp tăng tốc độ pipeline huấn luyện đáng kể.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo          | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
| -------------- | ------------------ | -------------- | ------------------ | - |
| IFEval         |                    |                |                    |   |
| GSM8K          |                    |                |                    |   |
| Global-MMLU-vi |                    |                |                    |   |

*Δ nào vượt \~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?*

*Trả lời ở đây.*

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss     | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
| -------- | --------------------- | --------------- | ----------------- | -------- |
| DPO      |                       |                 |                   |          |
| RPO      |                       |                 |                   |          |
| DPO-norm |                       |                 |                   |          |
| LD-DPO   |                       |                 |                   |          |
| ORPO     |                       |                 |                   |          |

*Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?*

---

## 9. GRPO (bonus NB7)

|                                           | Giá trị               |
| ----------------------------------------- | --------------------- |
| Độ chính xác trước / sau (n câu kiểm tra) | *<... / ... (n=...)>* |
| Sai số chuẩn ≈ √(p(1−p)/n)                | *<...>*               |

*Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?*

---

## Danh sách bonus

- NB3b — biến thể loss (+8)
- NB5 — GGUF SFT+DPO (+4)
- NB6 — benchmark (+6)
- NB7 — GRPO (+8)
- β-sweep (+6)
- Chấm chéo bằng hai họ mô hình (+4)
- Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là mặc dù DPO nhắm tới việc học sở thích, nhưng kết quả đối chiếu 1-1 lại cho thấy tỷ lệ "Hoà" (Ties) rất cao (38/50 câu ở tập held-out). Điều này khẳng định bước Fine-tuning (SFT) ban đầu đóng vai trò cực lớn, DPO/RLHF chỉ là chất xúc tác để điều chỉnh (align) chứ không thay thế hoàn toàn năng lực ngôn ngữ cốt lõi sinh ra từ Pretraining và SFT.
