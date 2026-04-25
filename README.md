# CARLA + BEVFusion Stage 1

Workspace này bọc ngoài `D:\Bevfusion-Z` để làm Phase 1 perception mà sửa repo BEVFusion gốc ở mức tối thiểu.

## Trạng thái hiện tại

- Phase 1A:
  - rig 6 camera + 1 lidar + 5 radar đã dump ổn
  - sync theo frame đã pass
  - sample layout `images/`, `lidar/`, `radar/`, `meta.json` đã pass
- Phase 1B:
  - geometry sanity offline đã có script riêng
  - adapter active đã bỏ path synthetic nuScenes radar 45D
  - radar bridge active là minimal `6D = [x, y, z, vx_comp, vy_comp, time_diff]`
  - checkpoint runtime được patch đúng vào radar input layer đầu tiên bằng semantic slice
- Phase 1C:
  - live inference script đã reuse đúng đường `dump -> adapter -> infer`
  - chỉ infer khi đủ history LiDAR/Radar
  - output live tách theo session để không lẫn history giữa các lần chạy
  - hỗ trợ `zero_bev`, `minimal radar`, và compare live

## Repo facts đã khóa

- Config runtime:
  - `D:\Bevfusion-Z\configs\nuscenes\det\transfusion\secfpn\camera+lidar+radar\swint_v0p075\sefuser_radarbev.yaml`
- Camera order đúng theo repo:
  - `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_LEFT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`
- Test image pipeline:
  - final size `[256, 704]`
  - resize `0.48`
  - normalize mean `[0.485, 0.456, 0.406]`
  - normalize std `[0.229, 0.224, 0.225]`
- LiDAR input branch:
  - adapter xuất `float32[N,5] = [x, y, z, intensity, time_lag]`
- Radar repo trace:
  - repo train dùng selected radar schema `45D`
  - checkpoint xác nhận radar input layer đầu là `[128, 47]`
  - `47 = 45 selected dims + f_center_x + f_center_y`

## Radar mismatch đã xử lý

CARLA radar native chỉ có:

- `velocity`
- `altitude`
- `azimuth`
- `depth`

Checkpoint nuScenes radar branch mong các field mà CARLA không có trực tiếp:

- `rcs`
- `dyn_prop_*`
- `ambig_state_*`
- `invalid_state_*`
- `pdh0_*`

Bridge active của workspace này không bịa các field đó nữa.

Thay vào đó:

- adapter dựng `x, y, z` từ spherical CARLA radar
- adapter ước lượng `vx_comp, vy_comp` bằng line-of-sight compensation trong current lidar frame
- adapter thêm `time_diff`
- runtime patch giảm radar input layer từ `45 -> 6`
- runtime slice checkpoint columns:
  - `x, y, z, vx_comp, vy_comp, time_diff, f_center_x, f_center_y`

Đây là bridge tối thiểu để infer/debug được, nhưng vẫn còn domain gap thật giữa CARLA radar và nuScenes radar.

## Cây file chính

```text
d:\Agent-AI\
  README.md
  carla_bevfusion_stage1\
    adapter.py
    bevfusion_runtime.py
    collector.py
    config_loader.py
    constants.py
    coordinate_utils.py
    dumper.py
    rig.py
    visualization.py
  scripts\
    geometry_sanity_check.py
    live_infer.py
    offline_infer.py
    print_blueprint_attributes.py
    run_carla_dump.py
```

## Environment

Cần Python env có:

- `carla`
- `torch`
- `mmcv`
- `mmdet`
- `mmdet3d`
- `matplotlib`
- `Pillow`

Shell dùng khi implement đã có `carla`, `torch`, `matplotlib`, `Pillow`; chưa có `mmcv/mmdet/mmdet3d`, nên offline model inference không verify end-to-end được trong shell này.

## Commands

### 1. In blueprint attributes của CARLA

```powershell
python d:\Agent-AI\scripts\print_blueprint_attributes.py `
  --host 127.0.0.1 `
  --port 2000 `
  --blueprint sensor.other.radar
```

### 2. Dump sample từ CARLA

```powershell
python d:\Agent-AI\scripts\run_carla_dump.py `
  --host 127.0.0.1 `
  --port 2000 `
  --output-root d:\Agent-AI\outputs\samples `
  --num-samples 30 `
  --fixed-delta-seconds 0.05 `
  --image-width 1600 `
  --image-height 900 `
  --camera-fov 70 `
  --autopilot
```

### 3. Geometry sanity trên sample dump

```powershell
python d:\Agent-AI\scripts\geometry_sanity_check.py `
  --sample-dir d:\Agent-AI\outputs\samples\sample_000000 `
  --output-dir d:\Agent-AI\outputs\geometry\sample_000000
```

Output:

- `radar_lidar_topdown.png`
- `geometry_report.json`

### 4. Offline inference baseline `camera + lidar` theo nghĩa zero radar BEV

```powershell
python d:\Agent-AI\scripts\offline_infer.py `
  --repo-root D:\Bevfusion-Z `
  --config D:\Bevfusion-Z\configs\nuscenes\det\transfusion\secfpn\camera+lidar+radar\swint_v0p075\sefuser_radarbev.yaml `
  --checkpoint D:\Data-Train\epoch_8_3sensor.pth `
  --sample-dir d:\Agent-AI\outputs\samples\sample_000010 `
  --output-dir d:\Agent-AI\outputs\offline\sample_000010_cam_lidar `
  --device cuda `
  --radar-bridge minimal `
  --radar-ablation zero_bev
```

### 5. Offline inference bridge `camera + lidar + minimal radar`

Lệnh này mặc định sẽ chạy:

- baseline `zero_bev`
- bridge `minimal`
- ghi thêm `comparison.json` và `bev_comparison.png`

```powershell
python d:\Agent-AI\scripts\offline_infer.py `
  --repo-root D:\Bevfusion-Z `
  --config D:\Bevfusion-Z\configs\nuscenes\det\transfusion\secfpn\camera+lidar+radar\swint_v0p075\sefuser_radarbev.yaml `
  --checkpoint D:\Data-Train\epoch_8_3sensor.pth `
  --sample-dir d:\Agent-AI\outputs\samples\sample_000010 `
  --output-dir d:\Agent-AI\outputs\offline\sample_000010_trimodal `
  --device cuda `
  --radar-bridge minimal `
  --radar-ablation none
```

### 6. Live inference baseline `camera + lidar` theo nghĩa zero radar BEV

Script live mới sẽ:

- tạo session output riêng trong `--output-root`
- dump mọi sample ra `samples/`
- chỉ bắt đầu infer khi đủ history
- ghi `live_summary.jsonl` để debug theo frame

```powershell
python d:\Agent-AI\scripts\live_infer.py `
  --host 127.0.0.1 `
  --port 2000 `
  --tm-port 8000 `
  --repo-root D:\Bevfusion-Z `
  --config D:\Bevfusion-Z\configs\nuscenes\det\transfusion\secfpn\camera+lidar+radar\swint_v0p075\sefuser_radarbev.yaml `
  --checkpoint D:\Data-Train\epoch_8_3sensor.pth `
  --output-root d:\Agent-AI\outputs\live `
  --num-samples 30 `
  --device cuda `
  --radar-bridge minimal `
  --radar-ablation zero_bev
```

### 7. Live inference trimodal `camera + lidar + minimal radar`

```powershell
python d:\Agent-AI\scripts\live_infer.py `
  --host 127.0.0.1 `
  --port 2000 `
  --tm-port 8000 `
  --repo-root D:\Bevfusion-Z `
  --config D:\Bevfusion-Z\configs\nuscenes\det\transfusion\secfpn\camera+lidar+radar\swint_v0p075\sefuser_radarbev.yaml `
  --checkpoint D:\Data-Train\epoch_8_3sensor.pth `
  --output-root d:\Agent-AI\outputs\live `
  --num-samples 30 `
  --device cuda `
  --radar-bridge minimal `
  --radar-ablation none
```

### 8. Live compare `zero_bev` vs `minimal radar`

Mỗi frame infer-ready sẽ ghi:

- `baseline_zero_bev\...`
- `bridge_minimal\...`
- `comparison.json`
- `bev_comparison.png`

```powershell
python d:\Agent-AI\scripts\live_infer.py `
  --host 127.0.0.1 `
  --port 2000 `
  --tm-port 8000 `
  --repo-root D:\Bevfusion-Z `
  --config D:\Bevfusion-Z\configs\nuscenes\det\transfusion\secfpn\camera+lidar+radar\swint_v0p075\sefuser_radarbev.yaml `
  --checkpoint D:\Data-Train\epoch_8_3sensor.pth `
  --output-root d:\Agent-AI\outputs\live `
  --num-samples 30 `
  --device cuda `
  --radar-bridge minimal `
  --radar-ablation none `
  --compare-zero-radar
```

### 9. Hybrid live workflow khi container va CARLA server khac version API

Case da gap tren may nay:

- container Python API `carla == 0.9.16`
- simulator host `CARLA == 0.9.15`

Direct capture mode trong container co the crash native. Khi gap case nay, dung workflow:

- host/Windows dump live samples bang `run_carla_dump.py`
- container chi watch `samples-root` va chay infer

Host dump:

```powershell
python d:\Agent-AI\scripts\run_carla_dump.py `
  --host 127.0.0.1 `
  --port 2000 `
  --output-root d:\Agent-AI\outputs\live_bridge_samples `
  --num-samples 200 `
  --fixed-delta-seconds 0.05 `
  --image-width 1600 `
  --image-height 900 `
  --camera-fov 70 `
  --autopilot
```

Container watch infer baseline:

```bash
python /workspace/Agent-AI/scripts/live_infer.py \
  --samples-root /workspace/Agent-AI/outputs/live_bridge_samples \
  --repo-root /workspace/Bevfusion-Z \
  --config /workspace/Bevfusion-Z/configs/nuscenes/det/transfusion/secfpn/camera+lidar+radar/swint_v0p075/sefuser_radarbev.yaml \
  --checkpoint /workspace/Data-Train/epoch_8_3sensor.pth \
  --output-root /workspace/Agent-AI/outputs/live_watch \
  --num-samples 30 \
  --device cuda \
  --radar-bridge minimal \
  --radar-ablation zero_bev
```

Container watch infer compare:

```bash
python /workspace/Agent-AI/scripts/live_infer.py \
  --samples-root /workspace/Agent-AI/outputs/live_bridge_samples \
  --repo-root /workspace/Bevfusion-Z \
  --config /workspace/Bevfusion-Z/configs/nuscenes/det/transfusion/secfpn/camera+lidar+radar/swint_v0p075/sefuser_radarbev.yaml \
  --checkpoint /workspace/Data-Train/epoch_8_3sensor.pth \
  --output-root /workspace/Agent-AI/outputs/live_watch_compare \
  --num-samples 30 \
  --device cuda \
  --radar-bridge minimal \
  --radar-ablation none \
  --compare-zero-radar
```

Watch mode cua `live_infer.py`:

- khong import `carla`
- poll thu muc `--samples-root`
- chi infer khi sample dump da day du file
- van giu history gate nhu offline/live capture mode
- ghi `session_manifest.json`, `session_summary.json`, `live_summary.jsonl`

## Output debug

Geometry:

- `radar_lidar_topdown.png`
- `geometry_report.json`

Offline inference single-run:

- `bev_debug.png`
- `camera_overlays\*.png`
- `predictions.json`
- `adapter_report.json`

Offline inference compare mode:

- `baseline_zero_bev\...`
- `bridge_minimal\...`
- `comparison.json`
- `bev_comparison.png`

Live inference session:

- `session_manifest.json`
- `session_summary.json`
- `live_summary.jsonl`
- `samples\sample_xxxxxx\...`
- `inference\sample_xxxxxx\...`

## Debug checklist

### Frame mismatch

- check log từ `run_carla_dump.py`
- tất cả 12 sensor phải cùng `frame`
- giữ `synchronous mode`
- giữ `sensor_tick == fixed_delta_seconds`

### Camera rỗng hoặc đen

- check `meta.json -> frame_log -> CAM_* -> mean_rgb`
- không dùng `no_rendering_mode`
- nếu headless thì phải dùng off-screen rendering, không tắt render camera

### LiDAR parse sai

- `lidar/LIDAR_TOP.npy` phải là `N x 4`
- raw layout là `[x, y, z, intensity]`
- adapter mới thêm `time_lag` ở bước build model input

### Radar schema sai

- `radar/*.npy` phải là `N x 4`
- raw layout là `[velocity, altitude, azimuth, depth]`
- `adapter_report.json` phải cho thấy active bridge là:
  - `minimal_6d_runtime_patch`
  - `input_columns = [x, y, z, vx_comp, vy_comp, time_diff]`

### Extrinsic sai

- check `meta.json -> sensors -> <name> -> matrix_*`
- check `geometry_report.json`
- front/back/left/right phải nằm đúng quadrant trong `radar_lidar_topdown.png`

### Coordinate frame sai

- CARLA: `x-forward, y-right, z-up`
- BEVFusion LiDAR frame: `x-forward, y-left, z-up`
- camera projection dùng OpenCV-style camera frame

### Checkpoint load được nhưng infer vô lý

- check `adapter_report.json -> runtime_info -> radar_bridge_details`
- check `radar_debug -> z_out_of_range_total`
- check `bev_debug.png` và `bev_comparison.png` trước
  - chấp nhận trước rằng radar branch vẫn còn domain shift lớn vì CARLA radar không phải nuScenes radar

### Live inference không chạy ngay từ frame đầu

- đây là hành vi đúng
- script chỉ infer khi đủ history:
  - LiDAR: `lidar_sweeps_test`
  - Radar: `radar_sweeps - 1`
- xem `live_summary.jsonl` để biết frame nào bị skip vì chưa đủ history

## Giới hạn hiện còn

- `time_diff` là chiều radar mạnh trong checkpoint, nên cadence sweep khác domain train sẽ ảnh hưởng rõ
- `vx_comp, vy_comp` hiện là line-of-sight compensated estimate, không phải full nuScenes compensated velocity
- một phần radar point có thể nằm ngoài `z` range của voxelizer; workspace đã log rõ nhưng không âm thầm “sửa đẹp” dữ liệu
- live inference hiện đã dùng đúng bridge của Phase 1B, nhưng vẫn phải so với output offline trước khi kết luận hành vi model

## Stage 2

Stage 2 doc code nam trong thu muc `stage2/` va chay tren artifact da co cua Stage 1.

Output chinh cua Stage 2:

- `normalized_prediction.json`
- `tracked_objects.json`
- `scene_summary.json`
- `risk_summary.json`
- `world_state.json`
- `decision_intent.json`
- `planner_interface_payload.json`
- `decision_timeline.jsonl`
- `world_state_timeline.jsonl`
- `evaluation_summary.json`

### Replay Stage 2 tren watch compare session voi bridge minimal

```powershell
python d:\Agent-AI\scripts\run_stage2_replay.py `
  --stage1-session d:\Agent-AI\outputs\live_watch_compare\live_20260401_095649 `
  --output-dir d:\Agent-AI\outputs\stage2\live_watch_compare_bridge `
  --prediction-variant bridge_minimal `
  --min-score 0.2
```

### Replay Stage 2 voi nguong score thap hon de debug scene richness

```powershell
python d:\Agent-AI\scripts\run_stage2_replay.py `
  --stage1-session d:\Agent-AI\outputs\live_watch_compare\live_20260401_095649 `
  --output-dir d:\Agent-AI\outputs\stage2\live_watch_compare_bridge_thr010 `
  --prediction-variant bridge_minimal `
  --min-score 0.1
```

### Replay Stage 2 tren baseline zero radar BEV

```powershell
python d:\Agent-AI\scripts\run_stage2_replay.py `
  --stage1-session d:\Agent-AI\outputs\live_watch_compare\live_20260401_095649 `
  --output-dir d:\Agent-AI\outputs\stage2\live_watch_compare_baseline `
  --prediction-variant baseline_zero_bev `
  --min-score 0.2
```

## Stage 3A

Stage 3A doc code nam trong thu muc `stage3/` va nang he thong tu:

- `world_state + tactical_intent`

len:

- `lane/route-aware behavior-planner-ready scene layer`

Input cua Stage 3A:

- `world_state.json`
- `decision_intent.json`
- `planner_interface_payload.json`
- CARLA map semantics thong qua waypoint API that

Output chinh cua Stage 3A moi frame:

- `lane_context.json`
- `lane_relative_objects.json`
- `maneuver_validation.json`
- `behavior_request.json`
- `lane_aware_world_state.json`
- `stage2_stage3_comparison.json`

Output session:

- `behavior_timeline.jsonl`
- `evaluation_summary.json`
- `visualization\behavior_timeline.png`
- `visualization\lane_context_timeline.png`

### Replay Stage 3A tren session Stage 2 da giau object hon

Khuyen dung `bridge_thr010` vi Stage 2 da xac nhan day la mode phu hop cho Stage 3 prototyping.

```powershell
python d:\Agent-AI\scripts\run_stage3_replay.py `
  --stage2-output-dir d:\Agent-AI\outputs\stage2\live_watch_compare_bridge_thr010 `
  --output-dir d:\Agent-AI\outputs\stage3\live_watch_compare_bridge_thr010_stage3a `
  --carla-host 127.0.0.1 `
  --carla-port 2000
```

### Replay Stage 3A chi tren mot doan ngan de probe nhanh

```powershell
python d:\Agent-AI\scripts\run_stage3_replay.py `
  --stage2-output-dir d:\Agent-AI\outputs\stage2\live_watch_compare_bridge_thr010 `
  --output-dir d:\Agent-AI\outputs\stage3\live_watch_compare_bridge_thr010_stage3a_probe `
  --carla-host 127.0.0.1 `
  --carla-port 2000 `
  --max-frames 5
```

### Ghi chu quan trong cho Stage 3A

- Stage 3A can query CARLA map API that, nen phai chay tren host/env co `carla` tuong thich voi simulator.
- Neu CARLA server dang o sai town so voi artifact, script se fail som truoc khi replay.
- Co the dung `--load-town-if-needed` neu muon script chu dong switch simulator world sang town dung cua artifact.
- Route context hien tai la `minimal and honest`:
  - current lane
  - left/right candidate lane
  - forward waypoint corridor
  - junction proximity
  - branch count / turn-like options neu map tra ve du branch
- Stage 3A chua sinh trajectory, chua la full behavior planner, va chua dong vao controller/MPC.
