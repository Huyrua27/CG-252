# BTL2 System - Synthetic Road Scene Dataset Generator

Pipeline này dùng framework OpenGL có sẵn trong thư mục `Sample/libs` để sinh dữ liệu cho Bài tập lớn 2:
- Ảnh RGB
- Bản đồ depth (npy + ảnh hiển thị)
- Instance mask theo pixel
- Nhãn detection/segmentation theo YOLO + COCO
- Metadata scene cho từng frame

## 1) Cấu trúc đầu ra

Mặc định dữ liệu được sinh ở `btl2_sys/output`:

```
output/
  images/rgb/*.png
  depth/raw_npy/*.npy
  depth/vis/*.png
  masks/instance/*.png
  masks/instance_vis/*.png
  labels/yolo/*.txt
  labels/coco/instances_train.json
  metadata/scenes/*.json
  metadata/asset_catalog.json
  references/*.png|*.jpg
```

## 2) Chạy nhanh

Từ thư mục `Sample`:

```bash
python btl2_sys/run.py --frames 40 --width 1280 --height 720 --extract
```

Tùy chọn chính:
- `--extract`: giải nén archive trong `object/` vào `object/extracted`
- `--frames`: số frame sinh ra
- `--seed`: seed để tái lập
- `--output`: thư mục output
- `--width`, `--height`: độ phân giải render
- `--max-faces`: giới hạn số tam giác cho mỗi mesh (để tăng tốc)
  - `0` (mặc định): giữ full mesh để tránh object bị vỡ
- `--camera-view`: hướng camera (`forward/front`, `rear/back`, `left`, `right`, `bird`)
- `--time-of-day`: preset ánh sáng (`day`, `sunset`, `night`)
- `--sun-azimuth`, `--sun-elevation`: chỉnh hướng nắng theo góc (độ)
- `--sun-intensity`, `--fill-intensity`: chỉnh cường độ đèn nắng/đèn phụ
- `--shadow-strength`: độ đậm bóng đổ trong khoảng `[0, 1]`
- `--disable-shadows`: tắt shadow map
- `--street-light-intensity`: cường độ đèn đường ảo (hữu ích ở `night`)
- `--street-light-count`: số đèn đường ảo dọc theo tuyến đường

Ví dụ:

```bash
python btl2_sys/run.py --extract --frames 120 --seed 252 --max-faces 120000 --output btl2_sys/output_120
```

Ví dụ sinh dữ liệu với góc camera sau + ánh nắng thấp + bóng đổ rõ:

```bash
python btl2_sys/run.py \
  --frames 80 --camera-view back --time-of-day sunset \
  --sun-azimuth 32 --sun-elevation 10 \
  --sun-intensity 1.3 --fill-intensity 0.85 --shadow-strength 0.82
```

Ví dụ cảnh đêm có đèn đường rõ nguồn sáng:

```bash
python btl2_sys/run.py \
  --frames 80 --camera-view forward --time-of-day night \
  --sun-intensity 0.15 --fill-intensity 0.75 --shadow-strength 0.55 \
  --street-light-intensity 2.8 --street-light-count 10
```

## 3) Danh mục lớp (class)

Class ID mặc định:
1. `car`
2. `truck`
3. `bike`
4. `moto`
5. `person`
6. `tree`
7. `road`
8. `building`

## 4) Ghi chú kỹ thuật

- Renderer chạy offscreen bằng GLFW hidden window (OpenGL 3.3 core).
- Object được lấy từ `object/extracted` (OBJ/MTL/texture) và chuẩn hóa scale theo kích thước mục tiêu.
- Với texture path bị sai (đường dẫn tuyệt đối từ máy nguồn), hệ thống tự dò texture theo tên file/nhãn vật liệu.
- `person` dùng billboard mesh với ảnh nhân vật trong bộ archive `rp_nathan_*` vì không có OBJ người đi bộ trong tập object hiện tại.

## 5) Gaussian Hybrid RGB (giảm domain gap)

Pipeline đã hỗ trợ chế độ hybrid:
- RGB: lấy từ Gaussian render (nếu có frame tương ứng)
- Ground truth (depth / mask / COCO / YOLO): vẫn từ mesh rasterization

### Sinh dataset hybrid

```bash
python btl2_sys/run.py \
  --frames 150 --width 640 --height 360 --seed 252 \
  --rgb-source gaussian_hybrid \
  --gaussian-rgb-dir btl2_sys/gaussian_rgb_frames \
  --gaussian-blend 1.0 \
  --output btl2_sys/output_gaussian_hybrid
```

Yêu cầu thư mục `--gaussian-rgb-dir` chứa ảnh theo tên frame `000001.png`, `000002.png`, ...
Nếu frame Gaussian thiếu thì hệ thống tự fallback RGB từ mesh cho frame đó.

## 6) Script thí nghiệm đầy đủ (mesh-only vs Gaussian-hybrid)

Script `btl2_sys/experiment_gaussian_vs_mesh.py` tự động:
1. Chọn asset có nhiều reference ảnh nhất (hoặc dùng `--asset-key`)
2. Chuẩn bị reference để train Gaussian
3. Chạy baseline mesh-only
4. Chạy dataset Gaussian-hybrid
5. (Tùy chọn) train YOLO và so sánh mAP

Ví dụ chạy cơ bản:

```bash
python btl2_sys/experiment_gaussian_vs_mesh.py --frames 150
```

Ví dụ có lệnh train/render Gaussian ngoài (template placeholder):

```bash
python btl2_sys/experiment_gaussian_vs_mesh.py \
  --frames 150 \
  --gaussian-train-cmd "python tools/train_gaussian.py --images {images_dir} --out {model_dir}" \
  --gaussian-render-cmd "python tools/render_gaussian.py --model {model_dir} --out {frames_dir} --cams {camera_meta_dir} --w {width} --h {height}" \
  --compare-map --epochs 20
```

Các placeholder hỗ trợ trong command template:
- Train: `{images_dir}`, `{workspace_dir}`, `{model_dir}`, `{asset_key}`, `{sample_root}`
- Render: `{model_dir}`, `{frames_dir}`, `{camera_meta_dir}`, `{width}`, `{height}`, `{frames}`, `{sample_root}`

Lưu ý:
- `--compare-map` yêu cầu cài `ultralytics`.
- Nếu không truyền `--val-data-yaml`, script sẽ val trên chính synthetic set (chỉ để so sánh nhanh, không phản ánh tốt khả năng tổng quát).

## 7) Advanced: Sinh dữ liệu 3D từ frame 2D

Pipeline mới `run_2d_to_3d.py` cho phép:
- Đọc chuỗi frame 2D từ thư mục ảnh
- Ước lượng depth (hoặc dùng depth `.npy` có sẵn)
- Temporal smoothing depth bằng optical flow
- Back-project thành point cloud 3D theo intrinsics camera
- Xuất point cloud từng frame + point cloud fused toàn sequence

### Chạy nhanh

```bash
python btl2_sys/run_2d_to_3d.py \
  --frames-dir btl2_sys/output/images/rgb \
  --output btl2_sys/output_2d_to_3d
```

### Dùng metadata camera + depth có sẵn

```bash
python btl2_sys/run_2d_to_3d.py \
  --frames-dir btl2_sys/output/images/rgb \
  --camera-meta-dir btl2_sys/output/metadata/scenes \
  --depth-npy-dir btl2_sys/output/depth/raw_npy \
  --sample-stride 2 \
  --max-points-per-frame 180000 \
  --output btl2_sys/output_2d_to_3d
```

### Cấu trúc output

```text
output_2d_to_3d/
  depth_est/
    raw_npy/*.npy
    vis/*.png
  pointcloud/
    frame_npy/*.npy
    frame_ply/*.ply
    fused/fused_pointcloud.ply
  metadata/reconstruction_summary.json
```

Ghi chú:
- Nếu có `--depth-npy-dir` và tồn tại file cùng stem (`000001.npy`...), hệ thống dùng depth đó.
- Nếu không có depth sẵn, hệ thống ước lượng depth heuristic từ ảnh RGB.
