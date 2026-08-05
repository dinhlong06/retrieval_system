# AIC 2026 – Bối cảnh đầy đủ cho AI Agent vòng Sơ tuyển

> Tài liệu này chuyển đổi và tái cấu trúc nội dung từ **“Thông tin vòng Sơ tuyển AIC 2026”** thành định dạng Markdown, nhằm giúp AI agent hiểu rõ bài toán, dữ liệu đầu vào, định dạng đầu ra, cách chấm điểm và các ràng buộc quan trọng.

---

## 1. Tổng quan cuộc thi

- **Tên cuộc thi:** Hội thi Thử thách Trí tuệ Nhân tạo Thành phố Hồ Chí Minh năm 2026.
- **Giai đoạn:** Vòng Sơ tuyển.
- **Bản chất bài toán:** Truy xuất video và định vị chính xác các khoảnh khắc hoặc sự kiện trong video.
- **Đơn vị dự đoán chính:**
  - `video_id`
  - `frame_id`
  - Câu trả lời văn bản, nếu là truy vấn Q&A
  - Một chuỗi nhiều `frame_id`, nếu là truy vấn TRAKE

Hệ thống không chỉ cần xác định video phù hợp mà còn phải xác định đúng khung hình hoặc đúng chuỗi khung hình ngữ nghĩa liên quan tới truy vấn.

---

# 2. Ba dạng truy vấn chính

## 2.1. Dạng 1 – Textual Known Item Search

### Tên đầy đủ

**Textual Known Item Search – Textual KIS**

### Mục tiêu

Tìm chính xác đoạn video chứa một sự kiện được mô tả bằng ngôn ngữ tự nhiên.

### Đầu vào

Một mô tả văn bản đầy đủ về sự kiện cần tìm.

Ví dụ:

> Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh.

### Đầu ra bắt buộc

```text
<video_id>, <frame_id>
```

Ví dụ:

```text
video_abc.mp4, 1500
```

### Ý nghĩa đầu ra

- `video_id`: video chứa sự kiện.
- `frame_id`: một khung hình bất kỳ nằm trong đoạn video chứa sự kiện đúng.

### Điều kiện đúng

Một câu trả lời chỉ được xem là đúng khi đồng thời:

1. `video_id` khớp video đáp án.
2. `frame_id` nằm trong khoảng khung hình đúng `[s, e]`.

### Hàm điểm R-Score

```math
R-Score(r_i) = I(v_i = GT_v \land id_i \in [s,e])
```

Trong đó:

- `r_i`: câu trả lời thứ `i`.
- `v_i`: video mà hệ thống dự đoán.
- `GT_v`: video ground truth.
- `id_i`: frame được dự đoán.
- `[s,e]`: khoảng frame hợp lệ.
- `I(.)`: hàm chỉ thị, trả về `1` nếu điều kiện đúng và `0` nếu sai.

### Ví dụ chấm điểm

Ground truth:

```text
video_id = L01_V001
frame hợp lệ = [500, 510]
```

| Câu trả lời | Kết quả | R-Score |
|---|---:|---:|
| `L01_V001, 505` | Đúng video, đúng khoảng frame | `1` |
| `L01_V001, 600` | Đúng video, sai khoảng frame | `0` |
| `L02_V003, 505` | Sai video | `0` |

### Hàm ý cho AI agent

Agent phải giải quyết hai tầng:

1. **Video retrieval:** tìm video phù hợp nhất với mô tả.
2. **Temporal localization:** tìm chính xác đoạn frame chứa sự kiện.

Chỉ đúng nội dung video nhưng sai thời điểm vẫn nhận `0` điểm.

---

## 2.2. Dạng 2 – Q&A

### Tên đầy đủ

**Visual Question Answering – Q&A**

### Mục tiêu

Tìm đúng sự kiện trong video và trích xuất một thông tin cụ thể từ sự kiện đó.

### Đầu vào

Bao gồm:

1. Mô tả bằng ngôn ngữ tự nhiên về sự kiện.
2. Một câu hỏi liên quan đến thông tin xuất hiện trong sự kiện.

Ví dụ:

> Trong video về lễ trao giải thưởng âm nhạc, có bao nhiêu người lên sân khấu để nhận giải thưởng lớn nhất?

### Đầu ra bắt buộc

```text
<video_id>, <frame_id>, <answer>
```

Ví dụ:

```text
video_xyz.mp4, 3450, 5
```

hoặc:

```text
video_xyz.mp4, 3450, Năm
```

### Ngôn ngữ câu trả lời

Câu trả lời có thể bằng:

- Tiếng Việt
- Tiếng Anh

### Điều kiện đúng

Một câu trả lời chỉ được xem là đúng khi đồng thời:

1. `video_id` đúng.
2. `frame_id` nằm trong khoảng frame hợp lệ.
3. `answer` khớp đáp án về mặt ngữ nghĩa.

### Hàm điểm R-Score

```math
R-Score(r_i) = I(v_i = GT_v \land id_i \in [s,e] \land a_i = GT_a)
```

Trong đó:

- `a_i`: câu trả lời văn bản của hệ thống.
- `GT_a`: đáp án ground truth.

### Ví dụ chấm điểm

Ground truth:

```text
video_id = L05_V005
frame hợp lệ = [800, 900]
answer = "màu xanh"
```

| Câu trả lời | Kết quả | R-Score |
|---|---:|---:|
| `L05_V005, 888, màu xanh` | Đúng hoàn toàn | `1` |
| `L05_V005, 888, màu trắng` | Sai answer | `0` |
| `L06_V007, 888, màu xanh` | Sai video | `0` |

### Hàm ý cho AI agent

Q&A là bài toán kết hợp ba năng lực:

1. **Truy xuất video**
2. **Định vị thời gian**
3. **Hiểu nội dung video để trả lời**

Agent không thể chỉ dùng mô hình ngôn ngữ trên metadata. Câu trả lời cuối cùng phải được kiểm chứng bằng nội dung hình ảnh hoặc chuỗi khung hình liên quan.

---

## 2.3. Dạng 3 – TRAKE

### Tên đầy đủ

**Temporal Retrieval and Alignment of Key Events – TRAKE**

### Mục tiêu

Từ một truy vấn mô tả chuỗi sự kiện:

1. Tìm đúng video chứa toàn bộ chuỗi sự kiện.
2. Căn chỉnh chính xác từng giai đoạn của chuỗi sự kiện với một khung hình ngữ nghĩa tương ứng.

### Hai giai đoạn của nhiệm vụ

#### Giai đoạn 1 – Retrieval

Từ một thư viện video lớn, tìm ra **một video duy nhất** chứa chuỗi sự kiện khớp nhất với truy vấn.

#### Giai đoạn 2 – Alignment

Trong video đã tìm được, xác định **một semantic keyframe duy nhất** cho mỗi giai đoạn của chuỗi sự kiện.

### Semantic keyframe là gì?

`Semantic keyframe` là khung hình đại diện đúng cho một khoảnh khắc có ý nghĩa về nội dung.

Nó **không phải** `I-Frame` trong kỹ thuật nén video.

- **Semantic keyframe:** được xác định theo ý nghĩa sự kiện.
- **I-Frame:** được xác định theo cấu trúc mã hóa video.

### Đầu vào

Một mô tả có cấu trúc gồm nhiều giai đoạn hoặc nhiều sự kiện con theo thứ tự thời gian.

### Đầu ra bắt buộc

```text
<video_id>, <frame_id_1>, ..., <frame_id_N>
```

Trong đó:

- `N`: số khoảnh khắc hoặc sự kiện con trong truy vấn.
- Thứ tự các `frame_id` phải tương ứng với thứ tự các sự kiện trong truy vấn.

### Điều kiện tiên quyết

Nếu `video_id` sai:

```text
R-Score = 0
```

Nếu `video_id` đúng, điểm được tính theo tỷ lệ số khoảnh khắc được căn chỉnh đúng.

### Hàm điểm R-Score

Nếu video đúng:

```math
R-Score(r_i) =
\frac{1}{N}
\sum_{j=1}^{N}
I(id_{i,j} \in [s_j,e_j])
```

Nếu video sai:

```math
R-Score(r_i) = 0
```

Trong đó:

- `N`: tổng số khoảnh khắc trong truy vấn.
- `id_{i,j}`: frame dự đoán cho khoảnh khắc thứ `j`.
- `[s_j,e_j]`: khoảng frame ground truth của khoảnh khắc thứ `j`.

### Đặc điểm khoảng ground truth

Khoảng `[s_j,e_j]` của semantic keyframe thường rất ngắn, thông thường dưới 10 frame.

Vì vậy, sai lệch thời gian nhỏ vẫn có thể làm mất điểm ở một sự kiện con.

---

## 2.4. Ví dụ TRAKE – Hành động nhảy cao

Chuỗi sự kiện gồm bốn khoảnh khắc:

1. **Chạy đà – Approach**
   - Khoảnh khắc bàn chân đầu tiên chạm đất và bước qua khỏi vạch xuất phát.

2. **Giậm nhảy – Take-off**
   - Khoảnh khắc đầu tiên bàn chân của chân giậm nhảy rời hoàn toàn khỏi mặt đất.

3. **Bay qua xà – Clearance**
   - Khoảnh khắc phần hông của vận động viên ở vị trí cao nhất so với xà ngang.

4. **Tiếp đất – Landing**
   - Khoảnh khắc đầu tiên một bộ phận của lưng, từ vai đến hông, bắt đầu chạm vào đệm.

Ví dụ ground truth khác:

```text
video_id = L10_V010

Khoảnh khắc 1: [95, 105]
Khoảnh khắc 2: [145, 155]
Khoảnh khắc 3: [195, 205]
Khoảnh khắc 4: [245, 255]
```

Câu trả lời:

```text
L10_V010, 101, 156, 203, 251
```

Đánh giá:

| Khoảnh khắc | Frame dự đoán | Khoảng đúng | Kết quả |
|---|---:|---:|---:|
| 1 | `101` | `[95,105]` | Đúng |
| 2 | `156` | `[145,155]` | Sai |
| 3 | `203` | `[195,205]` | Đúng |
| 4 | `251` | `[245,255]` | Đúng |

Kết quả:

```text
R-Score = 3/4 = 0.75
```

### Hàm ý cho AI agent

TRAKE không chỉ là video retrieval.

Agent phải:

1. Hiểu cấu trúc nhiều bước của truy vấn.
2. Phân tách truy vấn thành các sự kiện con.
3. Giữ đúng thứ tự thời gian.
4. Tìm video chứa toàn bộ chuỗi.
5. Căn chỉnh mỗi sự kiện con với frame chính xác.
6. Tránh chọn các frame chỉ “gần giống” nhưng chưa đúng thời điểm định nghĩa.

---

# 3. Cơ chế nộp nhiều câu trả lời

Với mỗi truy vấn, đội thi được gửi tối đa:

```text
100 câu trả lời
```

Mỗi câu trả lời có một `R-Score` riêng.

Điểm cuối cùng không chỉ dựa trên câu trả lời tốt nhất trong toàn bộ danh sách, mà còn phụ thuộc việc câu trả lời tốt được xếp ở vị trí nào.

---

# 4. Điểm cuối cùng

## 4.1. Top-k R-Score

Với mỗi:

```text
k ∈ {1, 5, 20, 50, 100}
```

hệ thống tính:

```math
R@k = \max_{1 \leq i \leq k} R-Score(r_i)
```

Nghĩa là:

- `R@1`: điểm câu trả lời đầu tiên.
- `R@5`: điểm cao nhất trong năm câu đầu.
- `R@20`: điểm cao nhất trong hai mươi câu đầu.
- `R@50`: điểm cao nhất trong năm mươi câu đầu.
- `R@100`: điểm cao nhất trong một trăm câu.

## 4.2. Final Score

```math
FinalScore =
\frac{1}{5}
\sum_{k \in \{1,5,20,50,100\}} R@k
```

### Ví dụ

Giả sử:

- Câu số 1 có `R-Score = 0.5`
- Câu số 3 có `R-Score = 0.8`
- Câu số 15 có `R-Score = 0.6`
- Các câu còn lại thấp hơn

Khi đó:

```text
R@1   = 0.5
R@5   = 0.8
R@20  = 0.8
R@50  = 0.8
R@100 = 0.8
```

```math
FinalScore =
\frac{0.5 + 0.8 + 0.8 + 0.8 + 0.8}{5}
= 0.74
```

### Ý nghĩa chiến lược

Hệ thống cần làm tốt cả hai việc:

1. Tìm được câu trả lời có độ chính xác cao.
2. Xếp câu trả lời tốt lên đầu danh sách.

Nếu đáp án đúng chỉ nằm ở vị trí thấp, hệ thống vẫn bị giảm điểm ở các mốc `R@1`, `R@5` hoặc `R@20`.

---

# 5. Dữ liệu vòng Sơ tuyển – Đợt 1

Dữ liệu cung cấp gồm các thành phần sau.

## 5.1. Videos

Chứa các video được cung cấp cho đội thi.

Đây là dữ liệu thi chính thức.

---

## 5.2. Keyframes

Chứa các keyframe được trích xuất từ video.

### Quy ước thư mục

Keyframe của mỗi video được lưu trong thư mục có tên tương ứng với `video_id`.

Ví dụ:

```text
Video:
L01_V001.mp4

Thư mục keyframe:
L01_V001/
```

### Quy ước tên file

Các file keyframe được đặt tên theo thứ tự tăng dần.

Ví dụ:

```text
L01_V001/
├── 0000.jpg
├── 0001.jpg
├── 0002.jpg
└── ...
```

### Ánh xạ sang frame gốc

Chỉ số frame gốc tương ứng của mỗi keyframe được ghi trong metadata.

Không được mặc định:

```text
0000.jpg == frame_id 0
```

Agent phải đọc metadata để ánh xạ từ chỉ số keyframe sang `frame_id` thực tế.

---

## 5.3. Objects

Chứa các file JSON liệt kê vật thể được phát hiện trong từng keyframe.

### Mô hình sử dụng

```text
Faster R-CNN pretrained trên OpenImages V4
```

### Quy ước tên file

File object JSON có tên tương ứng với file keyframe.

Ví dụ:

```text
Keyframe:
L01_V001/0000.jpg

Object JSON:
L01_V001/0000.json
```

### Vai trò

Dữ liệu object có thể hỗ trợ:

- Tìm người
- Tìm phương tiện
- Tìm đồ vật
- Lọc video hoặc frame theo thực thể xuất hiện

### Giới hạn

Kết quả object detection chỉ là dữ liệu hỗ trợ.

Nó có thể:

- Bỏ sót vật thể
- Nhận nhầm vật thể
- Không biểu diễn hành động
- Không mô tả quan hệ phức tạp giữa các đối tượng

---

## 5.4. CLIP features

Chứa đặc trưng CLIP của các keyframe.

### Mô hình sử dụng

```text
clip-ViT-B-32
```

### Định dạng

Toàn bộ vector của các keyframe được lưu trong một file:

```text
.npy
```

Thứ tự vector tăng dần tương ứng với thứ tự chỉ số keyframe.

### Vai trò

CLIP features có thể được dùng cho:

- Text-to-image retrieval
- Truy xuất keyframe theo mô tả văn bản
- Tạo chỉ mục vector
- Tìm candidate video hoặc candidate frame

### Ràng buộc ánh xạ

Agent phải đảm bảo:

```text
vector index
→ keyframe index
→ keyframe file
→ frame_id gốc
→ video_id
```

Nếu ánh xạ sai, hệ thống có thể tìm đúng hình ảnh nhưng nộp sai `frame_id`.

---

## 5.5. Metadata

Metadata được lấy từ YouTube của kênh cung cấp dữ liệu.

### Quy ước tên file

Ví dụ:

```text
Video:
L01_V001.mp4

Metadata:
L01_V001.json
```

### Lưu ý

Một số video có thể không có metadata tương ứng.

Vì vậy, pipeline phải xử lý trường hợp:

```text
metadata file không tồn tại
```

Agent không được giả định metadata luôn đầy đủ.

---

## 5.6. Liên kết dữ liệu

```text
https://docs.google.com/spreadsheets/d/1rfn1fieTThS_Ki3SIoJ6uXOx2AhMq7wGCak6W4jZyZM/edit?usp=sharing
```

---

# 6. Lưu ý chính thức về dữ liệu

## 6.1. Dữ liệu chính thức

Dữ liệu thi chính thức là:

```text
Videos
```

Các thành phần sau chỉ là dữ liệu bổ trợ:

- Keyframes
- Objects
- CLIP features
- Metadata

Do đó, hệ thống có thể tự trích xuất lại:

- Keyframes
- Embeddings
- Object detections
- Captions
- OCR
- Audio transcript
- Temporal features

Tuy nhiên, mọi kết quả cuối cùng vẫn phải ánh xạ về video và frame của dữ liệu video chính thức.

## 6.2. Phạm vi dữ liệu đợt 1

Dữ liệu hiện tại là:

```text
Batch 1 của AIC 2025
```

Dữ liệu đầy đủ của vòng Sơ tuyển AIC 2026 dự kiến có thêm:

```text
Batch 2
```

Batch 2 sẽ được thông báo sau.

---

# 7. Mô hình hóa bài toán cho AI agent

## 7.1. Input chuẩn hóa

AI agent nên chuyển mỗi truy vấn thành một cấu trúc nội bộ.

### Textual KIS

```json
{
  "task_type": "textual_kis",
  "event_description": "mô tả sự kiện"
}
```

### Q&A

```json
{
  "task_type": "qa",
  "event_description": "mô tả sự kiện",
  "question": "câu hỏi cần trả lời"
}
```

### TRAKE

```json
{
  "task_type": "trake",
  "global_description": "mô tả tổng thể",
  "events": [
    {
      "order": 1,
      "name": "sự kiện 1",
      "definition": "điều kiện xác định semantic keyframe"
    },
    {
      "order": 2,
      "name": "sự kiện 2",
      "definition": "điều kiện xác định semantic keyframe"
    }
  ]
}
```

---

## 7.2. Output chuẩn hóa

### Textual KIS

```json
{
  "video_id": "L01_V001",
  "frame_id": 505
}
```

### Q&A

```json
{
  "video_id": "L05_V005",
  "frame_id": 888,
  "answer": "màu xanh"
}
```

### TRAKE

```json
{
  "video_id": "L10_V010",
  "frame_ids": [101, 150, 203, 251]
}
```

---

# 8. Pipeline xử lý đề xuất cho AI agent

Phần này là cách tổ chức hệ thống để agent hiểu rõ luồng xử lý. Đây là diễn giải kỹ thuật từ yêu cầu bài toán, không phải kiến trúc bắt buộc của ban tổ chức.

## Bước 1 – Phân loại truy vấn

Xác định truy vấn thuộc:

- `Textual KIS`
- `Q&A`
- `TRAKE`

## Bước 2 – Phân tích ngữ nghĩa

Trích xuất:

- Chủ thể
- Hành động
- Vật thể
- Bối cảnh
- Địa điểm
- Màu sắc
- Số lượng
- Văn bản xuất hiện
- Thứ tự sự kiện
- Điều kiện xác định khoảnh khắc

## Bước 3 – Sinh nhiều truy vấn tìm kiếm

Tạo các biến thể:

- Mô tả đầy đủ
- Từ khóa chính
- Mô tả ngắn
- Bản dịch Việt–Anh hoặc Anh–Việt
- Truy vấn theo object
- Truy vấn theo hành động
- Truy vấn theo bối cảnh

## Bước 4 – Truy xuất candidate

Có thể kết hợp:

- CLIP similarity
- Caption retrieval
- OCR retrieval
- Object filtering
- Metadata retrieval
- Audio transcript retrieval
- Visual-language embeddings

## Bước 5 – Gom điểm theo video

Không chỉ xếp hạng từng keyframe riêng lẻ.

Cần tổng hợp tín hiệu từ nhiều keyframe để xếp hạng video:

```text
frame score
→ segment score
→ video score
```

## Bước 6 – Temporal localization

Sau khi có candidate video:

- Tìm đoạn thời gian phù hợp
- Mở rộng vùng lân cận quanh keyframe
- Đọc chuỗi frame liên tiếp
- Xác định khoảng `[s,e]`
- Chọn frame đại diện tốt nhất

## Bước 7 – Xử lý theo từng task

### Với Textual KIS

Chọn một frame nằm trong đoạn sự kiện.

### Với Q&A

- Định vị sự kiện
- Quan sát các frame liên quan
- Trả lời câu hỏi
- Chuẩn hóa answer

### Với TRAKE

- Phân tách sự kiện
- Tìm từng sự kiện con
- Kiểm tra thứ tự thời gian
- Chọn một semantic keyframe cho mỗi sự kiện

## Bước 8 – Sinh tối đa 100 câu trả lời

Nên duy trì tính đa dạng giữa các candidate:

- Nhiều candidate video
- Nhiều frame gần nhau
- Nhiều giả thuyết temporal
- Nhiều answer candidate cho Q&A

## Bước 9 – Re-ranking

Do `Final Score` phụ thuộc mạnh vào thứ hạng, cần đưa các dự đoán đáng tin cậy nhất lên đầu.

Các yếu tố có thể dùng:

- Similarity score
- Cross-modal reranker
- Temporal consistency
- Object agreement
- OCR agreement
- Answer confidence
- Event-order consistency
- Video-level evidence

## Bước 10 – Kiểm tra định dạng

Trước khi nộp phải xác minh:

- `video_id` tồn tại
- `frame_id` hợp lệ
- `frame_id` là frame gốc, không phải chỉ số keyframe
- Đủ số lượng frame đối với TRAKE
- Đúng thứ tự sự kiện
- Answer không rỗng đối với Q&A
- Không vượt quá 100 câu trả lời

---

# 9. Các lỗi hệ thống cần tránh

## 9.1. Nhầm keyframe index với frame_id

Sai:

```text
0005.jpg → frame_id = 5
```

Đúng:

```text
0005.jpg
→ tra metadata
→ frame_id gốc
```

## 9.2. Chỉ tìm đúng video nhưng không định vị đúng thời gian

Textual KIS và Q&A chấm `0` nếu frame nằm ngoài khoảng ground truth.

## 9.3. Q&A đúng câu trả lời nhưng sai video hoặc sai frame

Ba điều kiện phải cùng đúng.

## 9.4. TRAKE sai video

Sai video làm toàn bộ truy vấn nhận `0`, bất kể các frame dự đoán có giống sự kiện đến đâu.

## 9.5. TRAKE chọn frame đại diện chung chung

Semantic keyframe phải đúng theo định nghĩa khoảnh khắc, ví dụ:

- Không chọn lúc vận động viên chuẩn bị giậm nhảy.
- Phải chọn thời điểm chân giậm nhảy rời hoàn toàn khỏi mặt đất.

## 9.6. Không giữ thứ tự sự kiện

Mảng `frame_ids` phải theo đúng thứ tự các sự kiện trong truy vấn.

## 9.7. Xếp candidate tốt ở vị trí thấp

Dù candidate đúng nằm trong top 100, điểm vẫn giảm nếu top 1, top 5 hoặc top 20 yếu.

## 9.8. Phụ thuộc hoàn toàn vào dữ liệu hỗ trợ

Keyframes, Objects, CLIP features và Metadata không phải ground truth chính thức.

## 9.9. Giả định metadata luôn tồn tại

Một số video không có metadata.

---

# 10. Tiêu chí tối ưu hệ thống

## 10.1. Recall ở tầng retrieval

Candidate đúng phải xuất hiện trong top 100.

## 10.2. Precision ở top đầu

Candidate tốt nhất phải được đẩy lên top 1 hoặc top 5.

## 10.3. Temporal precision

Frame dự đoán phải nằm trong khoảng ground truth ngắn.

## 10.4. Semantic correctness

Với Q&A và TRAKE, hệ thống phải hiểu đúng ý nghĩa của sự kiện.

## 10.5. Mapping correctness

Mọi candidate phải ánh xạ chính xác giữa:

```text
video
↔ keyframe
↔ frame_id
↔ feature vector
↔ object JSON
↔ metadata
```

---

# 11. Checklist triển khai

## Dữ liệu

- [ ] Đọc được danh sách video.
- [ ] Đọc được keyframe.
- [ ] Đọc được CLIP `.npy`.
- [ ] Đọc được object JSON.
- [ ] Đọc được metadata nếu tồn tại.
- [ ] Có mapping keyframe index sang frame gốc.
- [ ] Có mapping vector index sang keyframe.
- [ ] Có xử lý video không có metadata.

## Retrieval

- [ ] Tạo index tìm kiếm.
- [ ] Hỗ trợ truy vấn tiếng Việt.
- [ ] Hỗ trợ truy vấn tiếng Anh.
- [ ] Truy xuất theo hình ảnh.
- [ ] Truy xuất theo object.
- [ ] Có reranking.
- [ ] Có gom điểm theo video.

## Temporal localization

- [ ] Có thể mở video quanh candidate frame.
- [ ] Có thể kiểm tra frame trước và sau.
- [ ] Có thể trả về frame gốc chính xác.
- [ ] Có cơ chế tìm semantic keyframe.

## Q&A

- [ ] Có bộ phận sinh answer.
- [ ] Có chuẩn hóa số, màu sắc, tên riêng.
- [ ] Có kiểm tra answer dựa trên frame liên quan.
- [ ] Có thể trả lời bằng tiếng Việt hoặc tiếng Anh.

## TRAKE

- [ ] Phân tách truy vấn thành nhiều sự kiện.
- [ ] Dự đoán đủ số frame.
- [ ] Giữ đúng thứ tự.
- [ ] Kiểm tra tính nhất quán trong cùng một video.
- [ ] Không dùng một frame cho nhiều sự kiện nếu không hợp lý.

## Submission

- [ ] Đúng định dạng từng task.
- [ ] Không quá 100 kết quả.
- [ ] Candidate tốt nhất ở đầu.
- [ ] Không trùng lặp vô ích.
- [ ] Không nộp keyframe index thay cho frame_id.

---

# 12. Tóm tắt ngắn cho AI agent

```text
Bạn đang giải bài toán truy xuất video và định vị thời gian cho AIC 2026.

Có ba loại truy vấn:

1. Textual KIS:
   Input là mô tả sự kiện.
   Output là video_id và một frame_id nằm trong đoạn sự kiện.

2. Q&A:
   Input là mô tả sự kiện và một câu hỏi.
   Output là video_id, frame_id và answer.
   Video, frame và answer đều phải đúng.

3. TRAKE:
   Input là chuỗi nhiều sự kiện theo thời gian.
   Output là một video_id và một frame_id cho từng sự kiện.
   Sai video thì toàn bộ truy vấn bằng 0.
   Đúng video thì điểm bằng tỷ lệ số sự kiện có frame nằm trong khoảng ground truth.

Mỗi truy vấn được gửi tối đa 100 câu trả lời.
Điểm cuối cùng là trung bình của R@1, R@5, R@20, R@50 và R@100.
Vì vậy, hệ thống phải vừa có recall tốt trong top 100, vừa xếp candidate tốt lên đầu.

Dữ liệu gồm Videos, Keyframes, Objects, CLIP features và Metadata.
Videos là dữ liệu chính thức.
Các thành phần còn lại chỉ hỗ trợ.
Phải ánh xạ đúng keyframe sang frame_id gốc.
Semantic keyframe là khung hình đúng về ý nghĩa sự kiện, không phải I-Frame kỹ thuật.
```

---

# 13. Nguồn

- Tài liệu: **Thông tin vòng Sơ tuyển AIC 2026**
- Phạm vi: 6 trang
- Nội dung gồm:
  - Ba dạng truy vấn
  - Công thức R-Score
  - Công thức Final Score
  - Mô tả dữ liệu vòng Sơ tuyển – Đợt 1
