## **Phần 1 — Giới thiệu về Docker**

# 1. Giới thiệu về Docker

Docker là một nền tảng giúp người dùng đóng gói và chạy chương trình của mình
trên các môi trường khác nhau một cách nhanh nhất dựa trên các container.

**Docker Image** là một dạng tập hợp các tệp của ứng dụng, được tạo ra bởi Docker
engine. Nội dung của các Docker image sẽ không bị thay đổi khi di chuyển.
Docker image được dùng để chạy các Docker container.

**Docker Container** là một dạng runtime của các Docker image, dùng để làm môi
trường chạy ứng dụng.

Hướng dẫn chi tiết tham khảo tại:  
https://docs.docker.com/get-started/

Dưới đây là **Phần 2 — Cài đặt Docker trên Ubuntu**, được đặt trong **Markdown code block**.

---

## **Phần 2 — Cài đặt Docker trên Ubuntu**

# 2. Cài đặt Docker trên Ubuntu

Đối với các hệ điều hành khác, tham khảo cách cài đặt tại:  
https://docs.docker.com/install/overview/

## 1. Cài đặt Docker

```bash
sudo apt-get install docker.io
```

## 2. Kiểm tra phiên bản Docker

```bash
sudo docker --version
# ⚠️ Docker version 24.0.5, build ced0996
```

## 3. Chạy thử Docker Hello World

```bash
sudo docker run hello-world
```

## 4. Một số câu lệnh phổ biến

**Liệt kê các images hiện có**

```bash
sudo docker images
```

**Liệt kê các container hiện có**

```bash
sudo docker ps -a
```

---

## **Phần 3 — Cài đặt Nvidia Docker / Driver**

# 3. Cài đặt Nvidia Docker / Driver

Để sử dụng được GPU trong Docker, bạn cần cài đặt **Nvidia Docker**.

Hướng dẫn cài đặt Nvidia Docker tại:  
https://github.com/NVIDIA/nvidia-docker

## Lưu ý quan trọng

Để tránh lỗi khi chấm, các đội cần cài đặt các thư viện đồng bộ môi trường
Docker với server của BTC.

Docker image yêu cầu:

```
docker image: pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
```

## **Phần 4 — Nộp bài cho Zalo AI Challenge**

# 4. Nộp bài cho Zalo AI Challenge

## Các bước cần thiết để tạo một Docker Image

1. **Tạo một docker container mới** (hoặc sử dụng container có sẵn).

2. **Đưa model và source code** vào trong container.

3. **Cài đặt các packages và libraries** cần thiết mà solution của bạn sử dụng để chạy.

4. **Viết file script `predict.sh`**  
   File này chứa command các bước để chạy test.  
   Nhận input từ `/data` và output ra `/result/submission.json` (tuỳ theo format của đề bài).  
   Ngoài ra, phải in ra các thông số **thời gian load model** và **thời gian predict** ở stdout.

5. **Nộp file `predict_notebook.ipynb` để đo thời gian inference.**

6. **Commit các thay đổi** trong Docker container.

7. **Save** Docker container thành **file image** và nộp lên website cuộc thi.

Dưới đây là **Phần 5 — Cấu trúc thư mục code**, trong **Markdown code block**.

---

## **Phần 5 — Cấu trúc thư mục code**

```markdown
# 5. Cấu trúc thư mục code

Ví dụ source code của bạn ở folder `/home/zdeploy/zac2025` với cấu trúc như sau:
```

📂 |---- predict.py
|---- preprocessing.py
|---- saved_models
| |---- models.safetensors # model cần được copy vào trong Docker
|---- train.py
|---- requirements.txt
|---- predict.sh
|---- start_jupyter.sh
|---- predict_notebook.ipynb # dùng để đánh giá thời gian inference
|---- training_code # chứa toàn bộ mã nguồn training và README
|---- README.md

````

## 1. Khởi động Docker container và đặt tên là `zac2025`

```bash
docker run --gpus '"device=0"' --network host -it --name zac2025 \
pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel /bin/bash
````

Cờ `--network host` là bắt buộc để chạy Jupyter đo thời gian inference.

Lúc này trong container đang ở vị trí `/`

```bash
root@zac2025:/# pwd
/
```

## 2. Kiểm tra container có sử dụng được GPU hay không

```bash
nvidia-smi
```

## 3. Mở terminal mới và copy source code vào container

Cú pháp chung:

```bash
sudo docker cp [source_path] [container_name]:[destination_path]
```

Ví dụ:

```bash
sudo docker cp /home/zdeploy/AILab/zac2025/ zac2025:/code/
```

Toàn bộ source code từ bên ngoài được copy vào container ở thư mục:

```
/code
```

## 3. Cài Python và các package cần thiết (trong container)

```bash
apt update
apt-get -y install libgl1-mesa-glx libglib2.0-0 vim
apt -y install python3-pip
```

(Chấp nhận các câu hỏi nếu được hỏi)

## 3. Cài đặt các thư viện từ `requirements.txt` hoặc thủ công

Chuyển sang thư mục code:

```bash
cd /code
```

Cài đặt:

```bash
pip install jupyterlab
pip install -r requirements.txt
pip install numpy
```

**Lưu ý:** Bắt buộc container phải cài đặt `jupyterlab` để chấm thời gian inference.

## 4. Chỉnh sửa file `/code/predict.sh`

Ví dụ nội dung:

```bash
python3 /code/preprocessing.py
python3 /code/predict.py
```

### ⚠️ Lưu ý quan trọng

- Trong `predict.py` phải define class model có hàm:

```python
def predict_streaming(self, frame_rgb_np, frame_idx)
```

Trả về `[x1, y1, x2, y2]` nếu phát hiện object, hoặc `None` nếu không có.

- Khi chấm, BTC sẽ liên tục gọi:

```python
model.predict_streaming(frame_rgb_np, frame_idx)
```

- Được phép cache frame cũ (t−1, t−2, …) để dự đoán frame t.
  **Không được** dùng thông tin frame t để update frame t−1.

- Trọng số load trong `predict.py` phải **trùng** với trọng số deploy lên drone ở Final Round.

- File `predict.py` phải ghi kết quả vào:

  ```
  /result/submission.json
  ```

  (Tự tạo `/result` nếu chưa tồn tại)

Chạy thử:

```bash
sh /code/predict.sh
```

## 5. Chỉnh sửa file `/code/start_jupyter.sh`

```bash
jupyter lab --port 9777 --ip 0.0.0.0 \
--NotebookApp.password='zac2025' \
--NotebookApp.token='zac2025' \
--allow-root --no-browser
```

## 6. Chỉnh sửa file `predict_notebook.ipynb`

Nội dung theo phụ lục.

## 7. Lưu lại các thay đổi trong container

```bash
sudo docker commit zac2025 zac2025:v1
```

Ví dụ output:

```
docker commit zac2025 zac2025:v1
sha256:...
```

## 10. Kiểm tra Docker lần cuối

### 10.1 Kiểm tra chạy `predict.sh`

Cấu trúc thư mục `/data`:

```
data
└── samples/   # chứa video để test
```

Chạy:

```bash
sudo docker run --gpus '"device=0"' \
 -v /data:/data \
 -v /home/zdeploy/zac2025/:/result \
 zac2025:v1 /bin/bash /code/predict.sh
```

Kiểm tra kết quả:

```bash
$ pwd && ls
/home/zdeploy/zac2025
predict.py predict.sh preprocessing.py requirements.txt submission.csv
```

File kết quả:

```
/home/zdeploy/zac2025/submission.json
```

### 10.2 Kiểm tra chạy Jupyter

```bash
sudo docker run -it --gpus '"device=0"' -p 9777:9777 \
 -v /data:/data -v /home/zdeploy/zac2025/:/result \
 zac2025:v1 /bin/bash /code/start_jupyter.sh
```

Mở trình duyệt:
`localhost:9777`
Mật khẩu: `zac2025`

Trong thư mục `/code` phải có file:

```
predict_notebook.ipynb
```

---

---

## Phần 6 — Bổ sung bắt buộc về Training Code & Tài liệu mô tả

# 6. Bổ sung bắt buộc về Training Code & Tài liệu mô tả

## 1. README.md mô tả ý tưởng

Các đội phải cung cấp file `README.md` mô tả ngắn gọn về:

- Ý tưởng tổng quan.
- Quy trình training.
- Quy trình inference.
- Các thành phần chính của code.

## 2. Cung cấp đầy đủ Training Code & Data

BTC sẽ sử dụng code và data trong thư mục `training_code/` để **reproduce**
lại quá trình training nhằm kiểm tra tính nhất quán của kết quả.

Vì BTC sẽ chạy reproduce _có internet_, các đội có thể:

- Upload training data và models lên **HuggingFace** hoặc dịch vụ tương tự.
- Trong thư mục `training_code/`, phải ghi rõ **URL tải xuống**.

⚠️ Lưu ý:

- Dữ liệu và mô hình ở những nền tảng này **không được thay đổi** sau deadline nộp Docker.
- Base model dùng trong quá trình training **không được** đưa vào bên trong Docker tránh làm Docker quá lớn.

## 3. Cố định seed

Để đảm bảo reproducibility, các đội phải:

- Set seed cố định trong cả training và inference.
- Đảm bảo mô hình reproduce lại từ code/data cho ra kết quả **giống** mô hình dùng trong inference.

Dưới đây là **Phần 7 — Phụ lục: Cấu trúc Jupyter notebook để đo thời gian**, trong **Markdown code block**.

---

## **Phần 7 — Phụ lục: Cấu trúc Jupyter notebook để đo thời gian**

# 7. Phụ lục: Cấu trúc Jupyter Notebook để đo thời gian

BTC yêu cầu mỗi đội chuẩn bị một file `predict_notebook.ipynb` để đánh giá
**thời gian chạy**. Nội dung notebook phải mô phỏng lại toàn bộ quá trình
dự đoán nhưng được chia thành các bước rõ ràng.

Notebook phải có **ít nhất 3 ô (cell)**:

---

## **Bước 1 — Set seed cố định**

BTC sẽ chỉ mở notebook và `Run All`.  
Các bạn phải đảm bảo không lỗi và seed được đặt cố định.

```python
import os
import torch
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)  # ví dụ seed 42
```

---

## **Bước 2 — Nạp mô hình và tài nguyên**

Không được dùng API ngoài (inference không có Internet).

```python
# load model, weights, preprocessors, configs,...
model = ...
```

---

## **Bước 3 — Đọc nội dung các test case**

```python
# read all test cases from /data
test_cases = ...
```

---

## **Bước 4 — Thực hiện dự đoán và log thời gian**

```python
from time import time

all_predicted_time = []
all_result = []

for file_name in test_cases:

    t1 = time()
    input_ = preprocess(file_name)
    forward = model.predict(input_)       # forward pass
    result = postprocess(forward)         # format output
    t2 = time()

    predicted_time = int((t2 - t1) * 1000)  # millisecond
    all_predicted_time.append((file_name, predicted_time))
    all_result.append(result)

write_predict_file(all_result)           # jupyter_submission.json
write_time_file(all_predicted_time)      # time_submission.csv
```

---

## **Yêu cầu bắt buộc của BTC**

1. Notebook khi chạy **không được lỗi**.
2. Output sau khi `Run All` phải gồm:

   - `time_submission.csv` (gồm 3 cột: id, answer, time_ms)
   - `jupyter_submission.json` (tương tự submission.json)

3. Các file xuất bởi Jupyter phải có prefix **`jupyter_`**
   để tránh bị overwrite với file từ `predict.py`.

BTC chỉ bắt đầu chấm phần **time_submission.csv** nếu `jupyter_submission.json`
cho kết quả **giống hệt** `submission.json` của predict.py.

---

Dưới đây là **Phần 8 — Upload Docker**, trong **Markdown code block**.

---

## **Phần 8 — Upload Docker**

# 8. Upload Docker

## 1. Lấy checksum MD5 của file docker

File docker sau khi đóng gói:  
`zac2025_TeamName.tar.gz`

Tham khảo cách kiểm tra MD5:

- Windows / macOS:  
  https://portal.nutanix.com/page/documents/kbs/details?targetId=kA07V000000LWYqS

- Linux:  
  https://www.geeksforgeeks.org/md5sum-linux-command/

Ví dụ:

```bash
md5sum zac2025_TeamName.tar.gz
```

---

## 2. Upload docker lên Google Drive

BTC sẽ tải Docker của bạn về máy chủ để kiểm tra kết quả cuối cùng.

Yêu cầu:

- Upload file `.tar.gz` lên Google Drive
- Chỉnh quyền share thành **“Anyone with the link”**

---

## 3. Nộp link Google Drive và checksum theo thông báo của BTC

BTC sẽ sử dụng link và checksum bạn gửi để xác minh file không bị thay đổi trong quá trình upload.

---

**Chúc các bạn thành công!**

---
