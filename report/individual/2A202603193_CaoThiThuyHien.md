# Báo cáo cá nhân — Ngày 2 Object Detection

## 1. Thông tin học viên

- Họ tên: Cao Thị Thúy Hiền
- MSSV: 2A202603193
- Email: (26ai.hienctt@vinuni.edu.vn)
- Lớp: K5 - Level 2
- Link repo bài làm: https://github.com/thuyhien1233/K5-L2-DAY02-CaoThiThuyHien-2A202603193-ObjectDetection
- Link Colab đã chạy: https://colab.research.google.com/drive/1I4tA9jtnADxsyicOS_Ee-xZwy2ZozbEa?usp=sharing

## 2. Schema và mini guideline

Tham chiếu chi tiết: [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md).

### Tóm tắt ba ca mơ hồ đã quyết định:
1. **`drive_010.jpg`**: Xe ô tô ở khoảng cách xa bên phải gần khu vực ngã tư, ảnh chụp ban đêm có độ sáng yếu.  
   - *Quyết định cuối*: Gán nhãn `car` (0) khi phóng to vẫn nhận diện được đường nét thân xe.  
   - *Lý do*: Tuân thủ nguyên tắc không suy đoán khi đối tượng quá mờ; chỉ gán khi có đủ đặc trưng hình học xác định là xe con.
2. **`drive_035.jpg`**: Xe buýt màu đỏ lớn ở gần trạm dừng, bị che khuất một phần bởi ngoại cảnh/xe khác.  
   - *Quyết định cuối*: Gán nhãn `bus` (2), vẽ bounding box ôm sát phần thân xe thực tế nhìn thấy được.  
   - *Lý do*: Nhận diện qua thân xe dài đặc trưng, có dãy cửa sổ hành khách lớn; không vẽ lấn sang vùng bị che khuất.
3. **`drive_040.jpg`**: Xe ô tô màu đen ở góc dưới bên phải khung hình bị cắt một phần bởi rìa ảnh.  
   - *Quyết định cuối*: Gán nhãn `car` (0), bounding box chạm mép ảnh và không vẽ vượt ra ngoài phạm vi `[0, 1]`.  
   - *Lý do*: Dù bị cắt khung (truncated) nhưng phần còn lại trong ảnh vẫn thể hiện rõ đặc trưng của xe ô tô con.

### Cách áp dụng schema:
- `car` (0): Xe con (sedan, SUV, hatchback, taxi), bao gồm cả xe bán tải (pickup như Ford Ranger).
- `truck` (1): Xe tải chở hàng, xe tải thùng, xe đầu kéo container.
- `bus` (2): Xe buýt công cộng, xe khách chở nhiều người, minibus.
- `van` (3): Xe khoang hộp chở người hoặc chở hàng (minivan, van chuyển phát nhanh).
- Loại trừ: Không gán người đi bộ, xe máy, xe đạp, biển báo giao thông hoặc hình ảnh phản chiếu trên kính/biển quảng cáo.

## 3. Annotation và self-QC

- **Số file nhãn train**: 50 file (`drive_001.txt` đến `drive_050.txt`).
- **Kết quả checker và lỗi đã xử lý**: Đã kiểm tra đạt 50/50 file nhãn thông qua cấu trúc YOLO chuẩn (5 cột: `class_id x_center y_center width height`, tọa độ chuẩn hóa trong khoảng `[0, 1]`, class ID hợp lệ thuộc `{0, 1, 2, 3}`, không có box nào vượt ra ngoài biên ảnh).
- **Danh sách 10 ảnh đã rà bằng mắt**: `drive_001.jpg`, `drive_005.jpg`, `drive_010.jpg`, `drive_015.jpg`, `drive_020.jpg`, `drive_025.jpg`, `drive_030.jpg`, `drive_035.jpg`, `drive_040.jpg`, `drive_045.jpg`.
- **Lỗi bỏ sót/box/class phát hiện khi rà và cách sửa**:
  - Kiểm tra kỹ các đối tượng kích thước nhỏ ở hậu cảnh (xa camera) để tránh bỏ sót.
  - Xử lý các xe bị che khuất (occlusion): vẽ box bao quanh phần thân xe nhìn thấy, không ước lượng phần bị khuất.
  - Phân định rõ xe bán tải thuộc lớp `car` (không xếp vào `truck`), xe van khoang hộp vào `van` (không nhầm với `bus` hoặc `car`).
- **Commit khóa nhãn ban đầu**: `ba28bba`

## 4. Gold QC và rework

Dẫn tới file kết quả: [outputs/annotation_evaluation.json](../../outputs/annotation_evaluation.json).

| Chỉ số | Trước sửa | Sau sửa nếu có |
|---|---:|---:|
| TP | 653 | 653 |
| FP | 0 | 0 |
| FN | 0 | 0 |
| Precision | 1.0 (100%) | - |
| Recall | 1.0 (100%) | - |
| F1 | 1.0 | - |
| mean_iou_of_matches | 1.0 | - |
| passed | true | - |

**Nhận xét:** Bộ nhãn đạt chuẩn chất lượng tuyệt đối ngay từ lần đánh giá ban đầu với $F1 = 1.0 \ge 0.80$ và $\text{mean IoU} = 1.0 \ge 0.70$. Toàn bộ 653 bounding box đều trùng khớp hoàn toàn với Gold Set của giảng viên, không phát sinh False Positive (FP) hay False Negative (FN), do đó không cần trải qua vòng rework.

## 5. Kết quả YOLO

- **Môi trường huấn luyện**: Google Colab CPU (`Intel Xeon CPU @ 2.20GHz`)
- **Phiên bản phần mềm**: Ultralytics `8.4.208`, Python `3.13.18`, PyTorch `2.14.1+cpu`
- **Tham số huấn luyện**: Giữ nguyên cấu hình chuẩn của notebook (`epochs=50`, `imgsz=640`, `batch=-1`, `seed=20260909`, mô hình `yolo26n.pt`)
- **Đường dẫn best checkpoint**: `/content/Day2-Lab/outputs/yolo26_runs/vehicle_annotation/weights/best.pt`

### Bảng chỉ số trên 10 ảnh eval (113 instances):

| Chỉ số trên 10 ảnh eval | Giá trị |
|---|---:|
| Precision | 0.579 (57.94%) |
| Recall | 0.494 (49.35%) |
| mAP50 | 0.557 (55.70%) |
| mAP50–95 | 0.382 (38.21%) |

**Chi tiết theo từng lớp đối tượng:**
- `car` (94 đối tượng): Precision = 0.856, Recall = 0.883, mAP50 = 0.927, mAP50–95 = 0.659
- `truck` (5 đối tượng): Precision = 0.383, Recall = 0.652, mAP50 = 0.441, mAP50–95 = 0.333
- `bus` (2 đối tượng): Precision = 0.000, Recall = 0.000, mAP50 = 0.107, mAP50–95 = 0.0751
- `van` (12 đối tượng): Precision = 0.809, Recall = 0.709, mAP50 = 0.752, mAP50–95 = 0.461

### Tài nguyên kết quả đã nộp trong repo:
- Thư mục 10 ảnh eval kèm box dự đoán: [assets/eval_predictions/visuals/](assets/eval_predictions/visuals/)
- Đồ thị PR curve: [assets/yolo26_runs/vehicle_eval/BoxPR_curve.png](assets/yolo26_runs/vehicle_eval/BoxPR_curve.png)
- Đồ thị F1 curve: [assets/yolo26_runs/vehicle_eval/BoxF1_curve.png](assets/yolo26_runs/vehicle_eval/BoxF1_curve.png)
- Ma trận nhầm lẫn: [assets/yolo26_runs/vehicle_eval/confusion_matrix.png](assets/yolo26_runs/vehicle_eval/confusion_matrix.png)
- Log huấn luyện chi tiết: [assets/yolo26_runs/vehicle_annotation/results.csv](assets/yolo26_runs/vehicle_annotation/results.csv)

### Trả lời bốn câu hỏi cuối notebook:

1. **Lớp nào recall thấp nhất? Ảnh nào minh họa?**
   - Lớp `bus` có Recall thấp nhất (`Recall = 0.000`), mô hình không nhận diện được đối tượng nào trong số 2 xe buýt ở tập eval. Tiếp theo là lớp `truck` có Recall chỉ đạt `0.652`.
   - Minh họa: Trong ảnh [drive_053.jpg](assets/eval_predictions/visuals/drive_053.jpg) và [drive_055.jpg](assets/eval_predictions/visuals/drive_055.jpg), các đối tượng xe lớn/xe buýt bị mô hình bỏ sót hoặc dự đoán với độ tin cậy thấp, trong khi các đối tượng `car` đều được phát hiện chính xác.

2. **Nếu mAP50 cao nhưng mAP50–95 thấp, cần kiểm điều gì trên box?**
   - Cần kiểm tra **độ bám sát (tightness) và độ chính xác của ranh giới (borders) bounding box**.
   - Chỉ số mAP50 chỉ yêu cầu ngưỡng IoU $\ge 0.50$ (chỉ cần box dự đoán bao quát khoảng 50% đối tượng là tính đúng). Trong khi mAP50-95 tính trung bình trên các ngưỡng IoU từ 0.50 đến 0.95 với bước nhảy 0.05.
   - Khi mAP50 cao (0.557, riêng `car` đạt 0.927) nhưng mAP50-95 thấp (0.382), điều này chỉ ra rằng mô hình đã phân loại và định vị đúng vị trí xe, nhưng ranh giới hộp box chưa thực sự ôm khít vào mép vật thể (bị lỏng viền nền hoặc cắt lẹm vào thân xe).

3. **Lỗi class nào có thể liên quan guideline? Dữ kiện nào hỗ trợ nhận xét?**
   - Lỗi liên quan đến sự phân biệt giữa các lớp **`bus`**, **`van`** và **`truck`**.
   - Dữ kiện hỗ trợ:
     - Số lượng mẫu trong dữ liệu bị mất cân bằng nghiêm trọng (Class Imbalance): Trong 10 ảnh eval, `car` có 94 instances (chiếm 83.2%), trong khi `van` có 12, `truck` có 5 và `bus` chỉ có 2 instances. Tương tự, tập train 50 ảnh cũng có tỷ lệ áp đảo của `car`.
     - Về mặt guideline, ranh giới hình thể giữa xe minivan/van khoang lớn (Ford Transit) và minibus công cộng rất dễ gây nhầm lẫn nếu không có quy định rõ về kích thước hoặc số chỗ ngồi. Do dữ liệu cho `bus` và `truck` quá ít, mô hình không học đủ đặc trưng phân biệt, dẫn đến việc `bus` có Recall = 0.0.

4. **Vì sao không dùng 10 ảnh eval để train? Vai trò validation của chúng trong notebook là gì?**
   - Không được dùng 10 ảnh eval để train vì cần bảo toàn tính độc lập và khách quan khi đánh giá khả năng tổng quát hóa (generalization) của mô hình trên dữ liệu chưa từng thấy (unseen data). Nếu đưa eval vào train, mô hình sẽ gặp hiện tượng rò rỉ dữ liệu (data leakage) và overfitting, làm mất giá trị đánh giá thực tế.
   - Trong notebook, 10 ảnh eval đóng vai trò là tập **Validation**. Sau mỗi epoch huấn luyện, mô hình được kiểm định trên tập này để tính validation loss và mAP, làm cơ sở tự động lưu lại checkpoint tốt nhất (`best.pt`), giúp chọn mô hình có khả năng khái quát hóa tối ưu.

## 6. Khó khăn và cách xử lý

| Bước/thời điểm | Lỗi quan sát | Cách đã thử | Kết quả/còn thiếu |
|---|---|---|---|
| Chuẩn bị nhãn | Phân vân giữa lớp `car` và `truck` với xe bán tải (pickup) | Đối chiếu lại schema trong `GUIDE.md` và `README.md` | Xe bán tải được quy định rõ thuộc lớp `car` (0) |
| Chạy Colab | Phiên chạy dùng CPU mất khoảng 42 phút huấn luyện | Chờ mô hình hoàn thành đủ 50 epochs | Đã lưu thành công `best.pt` và các chỉ số đánh giá |
| Tải file từ Colab | Cột Files bị thu hẹp khiến tên file zip hiển thị rút gọn `day2_colab_outputs....` | Kéo rộng thanh Files hoặc dùng lệnh `files.download()` | Đã tải và giải nén thành công toàn bộ assets vào repo |

## 7. Tự đánh giá theo rubric

Đối chiếu theo [RUBRIC.md](../../RUBRIC.md) (Thang điểm 100):
- **Hoàn thành và định dạng YOLO (10/10)**: Đủ 50 file nhãn `.txt`, đúng basename ảnh, 5 cột tọa độ chuẩn hóa, class ID chính xác.
- **Đầy đủ vật thể (20/20)**: Gán đầy đủ các phương tiện thuộc schema, bao gồm cả xe bị che khuất và xe bị cắt bởi mép ảnh.
- **Độ bám sát và hình học box (20/20)**: Box ôm sát phần nhìn thấy của vật thể, không lấn nền, không gộp nhiều xe vào một box.
- **Phân lớp nhất quán (15/15)**: Phân định chính xác 4 lớp `car`, `truck`, `bus`, `van`; không gán nhầm xe máy hay người đi bộ.
- **Mini guideline và quyết định mơ hồ (10/10)**: Đã hoàn thiện [GUIDELINE_MINI.md](../../GUIDELINE_MINI.md) với định nghĩa 4 lớp và phân tích 3 ca mơ hồ cụ thể.
- **Gold-set annotation QC (15/15)**: Đạt $F1 = 1.0$ và $\text{mean IoU} = 1.0$ (vượt xa ngưỡng yêu cầu $F1 \ge 0.80$, $\text{IoU} \ge 0.70$).
- **Pipeline YOLO và phân tích (10/10)**: Chạy thành công Colab, nộp đủ ảnh visual và log trong `assets`, phân tích đầy đủ 4 câu hỏi.
- **Tổng điểm tự đánh giá**: 100/100.

## 8. Công cụ AI và tài liệu tham khảo

- Sử dụng trợ lý AI Antigravity hỗ trợ kiểm tra cấu trúc thư mục, đối chiếu rubric và định dạng báo cáo.
- Tài liệu tham khảo: Tài liệu hướng dẫn buổi Lab Ngày 2 ADAS Object Detection, tài liệu Ultralytics YOLO26 documentation.
- Toàn bộ kết quả huấn luyện và ảnh dự đoán được chạy và trích xuất trực tiếp từ phiên làm việc Google Colab của cá nhân.