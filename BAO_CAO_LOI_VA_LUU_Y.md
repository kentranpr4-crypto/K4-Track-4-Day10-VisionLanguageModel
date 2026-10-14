# Báo cáo lỗi và lưu ý kỹ thuật

## Môi trường và dữ liệu

Thí nghiệm chạy trên Vast.ai với RTX 6000 Ada 48 GB và PyTorch 2.6.0+cu124. Dataset Kaggle được chia theo phiên quay, phiên bản `d10-21e4070301`, gồm 1.422 ảnh train, 276 val và 311 test. Không commit ảnh, prediction thô hay adapter; chỉ commit evidence chấm điểm và ảnh minh họa.

## Các lỗi đã gặp

| Hiện tượng | Nguyên nhân quan sát được | Cách xử lý |
|---|---|---|
| `torchaudio` không import được | Image VM có `torchaudio` không khớp `torch` | Gỡ `torchaudio` vì lab không dùng audio; dựng venv PyTorch 2.6 cho lượt train cuối |
| PEFT không tìm thấy target LoRA | Regex `target_modules` không khớp cách PEFT 0.21 tra tên module | Dùng danh sách suffix module của LLM/ViT |
| `KeyError: 151643` | `mm_token_type_ids` bị pad bằng token ID | Pad modality và `attention_mask` bằng `0` |
| `IndexError` gần bước 175 | Batch index không quay vòng tại cuối tập train | Lấy index modulo số ảnh train |
| `GradScaler` từ chối gradient FP16 | PEFT/optimizer có gradient nhiều dtype | Dùng BF16 cho E2/E3, không dùng `GradScaler` trên stack này |
| E2/E3 bị NaN hoặc mismatch dtype trong smoke test | Vision/DeepStack và projector không cùng chế độ dtype khi train/suy luận | Dùng BF16 autocast nhất quán và chạy smoke cả train lẫn generate trước lượt đầy đủ |

## Kết quả cần diễn giải thận trọng

E1 có F1 0.273; E2 và E3 đều có F1 0.000 dù loss train giảm và output JSON hợp lệ. E2 trả đúng một chuỗi box manhole cho toàn bộ 311 ảnh test; E3 chỉ có 7 chuỗi khác nhau, vẫn tập trung vào box manhole tương tự. Đây là mode collapse của lần chạy này, không phải bằng chứng rằng việc mở projector hoặc ViT luôn bất lợi.

E4 đổi duy nhất rank LoRA trên nền E1. Rank 4 đạt F1 0.286 (CI95 0.244-0.328); rank 64 đạt 0.194 (CI95 0.159-0.231). CI95 riêng từng setup không thay thế bootstrap ghép cặp của hiệu số F1; kết quả nên được lặp với seed khác nếu cần kết luận tổng quát.

`scripts/score.py` dùng tọa độ 0-1000. Chấm lại cùng output E1 với `--coord pixel` làm F1 từ 0.273 về 0.000. Luôn dùng cùng quy ước trong prompt, nhãn và lệnh chấm.

Một test của bản gốc (`test_report_template_has_student_markers_and_ten_questions`) kiểm tra `REPORT.md` vẫn còn chuỗi `<điền>` của template. Điều này mâu thuẫn với `scripts/check_submission.py` và quy định nộp bài yêu cầu không còn placeholder. Khi báo cáo được điền thật, test đó thất bại đúng như điều kiện viết sẵn; các test còn lại đạt. Không sửa `tests/` hoặc `scripts/` để che xung đột.
