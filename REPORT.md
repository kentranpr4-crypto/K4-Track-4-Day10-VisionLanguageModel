Họ tên: Trần Cao Quốc Định  
MSSV: 2A202602939  
`dataset_version`: `d10-21e4070301`

Môi trường: Vast.ai, NVIDIA RTX 6000 Ada 48 GB, PyTorch 2.6.0+cu124. Các số dưới đây lấy từ `report/evidence/*.json` trên split test 311 ảnh; các thí nghiệm còn chạy được ghi rõ trạng thái, không thay bằng số ước lượng.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | chưa đo | chưa đo | 0.079 | 0.054–0.109 | 1.010 | 0.170 | 0.021 | 0.162 |
| E1 | LoRA LLM | 33,030,144 | 6.218 | 33.14 | 0.273 | 0.233–0.314 | 0.643 | 0.174 | 0.199 | 0.462 |
| E2 | + projector | 87,609,344 | 6.442 | 35.53 | 0.000 | 0.000–0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| E3 | + LoRA ViT | 94,892,032 | 6.764 | 37.51 | 0.000 | 0.000–0.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| E4 | rank 4 và 64 trên E1 | đang chạy | đang chạy | đang chạy | đang chạy | đang chạy | đang chạy | đang chạy | đang chạy | đang chạy |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Theo tài liệu model trong đề, tổng khoảng 4,44 tỉ tham số gồm LLM khoảng 4,0 tỉ và ViT khoảng 415 triệu; merger khoảng 27 triệu và ba deepstack merger khoảng 82 triệu. Đây là quy mô tham khảo, không phải số đo trực tiếp từ checkpoint VM. E1 cập nhật LoRA của attention và MLP trong LLM: runner ghi 33,030,144 tham số trainable. E2 mở thêm các tham số `visual.merger` và `visual.deepstack_merger_list`, runner ghi 87,609,344 tham số trainable, tăng 54,579,200 so với E1. E3 thêm LoRA ViT, đạt 94,892,032 tham số trainable, tăng 7,282,688 so với E2. Chênh lệch E2 thực đo không bằng 109 triệu mà hướng dẫn ước tính; đây là điểm cần kiểm tra lại mapping module và điều kiện `requires_grad` trước khi kết luận về toàn bộ projector.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate = 0.913` (khoảng 284/311 ảnh), F1 0.079. Hai lỗi quan sát được: box crack quá rộng/sai vị trí và JSON sai cấu trúc. Ví dụ ảnh `vlcsnap-00201`: `[{"label":"crack", "bbox_2d":[0,457,998,999]}]`, gần như phủ hết mặt đường; ảnh `vlcsnap-00210` có hai khóa `bbox_2d` trong cùng một object, nên parser JSON giữ khóa cuối và làm mất box đầu. E0 có 254 false positive của lớp crack, nhiều hơn các lớp khác.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Tôi chấm cùng file `work/e1.jsonl` theo hai quy ước: `--coord norm1000` cho F1 = 0.273, còn `--coord pixel` cho F1 = 0.000 (CI95 0–0). Đây là kiểm chứng trực tiếp rằng output 0–1000 không được diễn giải như pixel 640×360: tọa độ bị nhân sai tỉ lệ theo từng trục, khiến IoU với ground truth dưới 0.5 dù JSON vẫn hợp lệ.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Loss chỉ tính trên token câu trả lời JSON và token kết thúc, còn toàn bộ prefix của user, prompt và token ảnh được đặt `labels=-100`. Mục tiêu là học cách ánh xạ ảnh/prompt thành box. Nếu tính loss trên prompt và token ảnh, gradient bị chi phối bởi việc tái tạo phần đầu vào vốn đã biết, làm sai mục tiêu và tốn khả năng cập nhật cho output grounding. Runner kiểm tra prompt token là prefix của chuỗi đầy đủ trước khi mask.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Projector cần trọng số có thể cập nhật trực tiếp; nếu khối `visual` bị lượng tử hóa 4-bit, các lớp linear trong merger không còn là tham số floating point thông thường để optimizer huấn luyện đầy đủ. Runner dùng `llm_int8_skip_modules=["visual","lm_head"]`, giữ vision ngoài lượng tử hóa và chuyển `visual` sang float32 trong E2/E3. Float32 giúp LayerNorm và gradient của merger ổn định hơn khi có nhiều nhánh DeepStack; trên VM đã gặp NaN với FP16. Dù vậy kết quả E2 cuối cùng vẫn bị mode collapse, nên việc tránh NaN không đồng nghĩa mô hình học tốt.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Frame liền nhau từ cùng phiên quay gần như trùng cảnh. Chia ngẫu nhiên từng ảnh sẽ đưa cảnh giống nhau vào cả train và test, tạo F1 cao giả do rò rỉ. Split theo `session` cố định phiên vào đúng một tập. Số liệu thật: train 1,422 ảnh, pothole 1,074, crack 1,750, manhole 691; val 276 ảnh, 66/470/103; test 311 ảnh, 121/299/163 theo thứ tự ba lớp trên. `dataset_version = d10-21e4070301`.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 đạt F1 0.273, CI95 0.233–0.314, VRAM 6.218 GB và 33.14 phút. E2 thêm projector có F1 0.000 dù VRAM 6.442 GB và 35.53 phút. Kiểm tra prediction E2 cho thấy cả 311 ảnh đều trả `[{"label":"manhole","bbox_2d":[402,583,484,633]}]`: đây là mode collapse, không phải lỗi parser (`parse_valid_rate=1.000`). E3 thêm LoRA ViT vẫn F1 0.000, VRAM 6.764 GB, 37.51 phút. Vì vậy E1 tốt nhất trong ba setup của lần chạy này; chi phí mở projector/ViT không đem lại lợi ích với cách huấn luyện hiện tại. CI95 E1 là 0.233–0.314, trong khi E2/E3 đều 0–0.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Bootstrap 95% trên 311 ảnh cho E0 là 0.054–0.109 (rộng 0.055) và E1 là 0.233–0.314 (rộng 0.081). Hai khoảng không giao nhau, nên cải thiện E1 so với E0 khá rõ trên tập test này. Tuy nhiên CI từng setup không phải CI trực tiếp của *hiệu số* F1; để so sánh sát hơn cần bootstrap ghép cặp theo ảnh. Với lớp pothole chỉ 121 box test, kết luận riêng lớp còn bất định hơn micro F1.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E1, pothole kém nhất với F1 0.174, recall 0.124 (15 TP, 106 FN), sát với crack F1 0.199 nhưng thấp hơn manhole F1 0.462. Pothole có ít box test hơn crack (121 so với 299); vùng hư hại nhỏ và ranh giới không rõ cũng làm khó định vị IoU≥0.5. Crack có nhiều false positive nhất (117) do vệt đường dễ bị nhầm với vết nứt. Sáu ảnh minh họa E1 bên dưới: viền xanh là ground truth, viền đỏ là prediction. `hit` nghĩa là ảnh chứa ít nhất một TP của lớp, `miss` nghĩa là ảnh chứa ít nhất một FN; ảnh có thể đồng thời chứa box đúng và sai.

| Lớp | Có dự đoán đúng | Có bỏ sót |
|---|---|---|
| pothole | ![Pothole hit](report/figures/pothole_hit.jpg) | ![Pothole miss](report/figures/pothole_miss.jpg) |
| crack | ![Crack hit](report/figures/crack_hit.jpg) | ![Crack miss](report/figures/crack_miss.jpg) |
| manhole | ![Manhole hit](report/figures/manhole_hit.jpg) | ![Manhole miss](report/figures/manhole_miss.jpg) |

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
E4 thay đúng một biến là rank LoRA trên E1, giữ nguyên dữ liệu, prompt và các siêu tham số khác. Hai giá trị cần chạy là `r=4` và `r=64`, với alpha bằng `2*r`. Kết quả và CI95 sẽ được bổ sung từ hai file `e4_rank_4.json` và `e4_rank_64.json` sau khi job hoàn tất; hiện chưa có cơ sở để kết luận rank nào tốt hơn.

## Nhìn lại

Lab cho thấy loss giảm không đảm bảo F1 tăng: E2 giảm loss từ khoảng 0.98 ở bước 25 xuống 0.55 ở bước 300, nhưng trả cùng một box manhole cho mọi ảnh test. Công cụ chấm phát hiện F1=0, song không giải thích được nguyên nhân mode collapse; cần xem output thô và kiểm tra riêng độ đa dạng dự đoán. Trong quá trình chạy, tôi cũng phải sửa padding `mm_token_type_ids`, vòng lặp batch và dtype/autocast của Qwen3-VL để có training ổn định trên VM.
