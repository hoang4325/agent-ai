# Agent-AI

Agent-AI là workspace nghiên cứu cho hệ thống autonomous driving nhiều tầng. Repo này gom các artifact, mô-đun replay, benchmark, runtime online, và lớp takeover/safety theo từng stage.

## Tổng quan

Repository được tổ chức theo pipeline từ perception đến control:

1. **Stage 1** — CARLA + BEVFusion perception bridge
2. **Stage 2** — world-state và tactical context
3. **Stage 3 / 3B / 3C** — behavior planning, route awareness, replay và coverage

4. **Stage 4** — online runtime, monitoring, shadow execution
5. **Benchmark** — metric registry, replay corpus, gate, report
6. **Stage 9** — authority takeover, TOR, human override, safety veto, MRM

Repo này là workspace nghiên cứu, không phải một ứng dụng đơn lẻ có thể chạy end-to-end ngay từ root.

## Cấu trúc thư mục

- [carla_bevfusion_stage1/](carla_bevfusion_stage1) — bridge giữa CARLA sensor dump và BEVFusion runtime
- [benchmark/](benchmark) — benchmark scaffold, frozen corpus, metrics, gate scripts, reports
- [stage2/](stage2) — world-state, intent, scene abstraction

- [stage3/](stage3) — behavior planning và route-aware logic
- [stage3b/](stage3b) — builder cho behavior request và route context
- [stage3c/](stage3c) — replay runner và artifact-based stage 3c flows

- [stage3c_coverage/](stage3c_coverage) — coverage, planner quality, stop tuning
- [stage4/](stage4) — online orchestrator, runtime, evaluation, monitoring
- [stage9/](stage9) — authority arbitration, safety, TOR, MRM, evaluation

- [scripts/](scripts) — CLI utilities cho dump, replay, benchmark, runtime
- [outputs/](outputs) — artifact sinh ra khi chạy thử
- [tmp/](tmp) — file tạm và helper scripts

## Stage 1 ngắn gọn

Stage 1 là lớp perception: dump sensor từ CARLA, chuẩn hoá dữ liệu, rồi đưa vào BEVFusion.

Các nhóm file chính:

- `collector.py`, `dumper.py`, `rig.py` — lấy mẫu sensor và đồng bộ frame
- `adapter.py`, `bevfusion_runtime.py`, `config_loader.py` — bridge dữ liệu vào model
- `coordinate_utils.py`, `visualization.py` — geometry và debug

## Stage 9 ngắn gọn

Stage 9 là phần điều phối quyền lái rõ nhất trong repo. Nó dùng state machine để chuyển giữa baseline, agent, human và MRM.

Các file lõi:

- [stage9/schemas.py](stage9/schemas.py) — dataclass và enum dùng chung
- [stage9/authority_state_machine.py](stage9/authority_state_machine.py) — FSM, cooldown, hysteresis
- [stage9/authority_arbiter.py](stage9/authority_arbiter.py) — vòng lặp điều phối chính

- [stage9/safety_supervisor.py](stage9/safety_supervisor.py) — hard gate và veto
- [stage9/contract_resolver.py](stage9/contract_resolver.py) — chuyển contract sang trajectory request
- [stage9/handoff_planner.py](stage9/handoff_planner.py) — blend ở mức reference/objective

- [stage9/baseline_detector.py](stage9/baseline_detector.py) — phát hiện baseline stuck/oscillation/degen
- [stage9/tor_manager.py](stage9/tor_manager.py) — takeover request escalation
- [stage9/minimal_risk_maneuver.py](stage9/minimal_risk_maneuver.py) — fallback an toàn

- [stage9/human_override_monitor.py](stage9/human_override_monitor.py) — detect can thiệp người lái
- [stage9/authority_logger.py](stage9/authority_logger.py) — audit log JSONL
- [stage9/stage9_evaluator.py](stage9/stage9_evaluator.py) — metric evaluation

- [stage9/scenario_runner.py](stage9/scenario_runner.py) — 12 scenario giả lập T9-001 → T9-012

## Luồng dữ liệu chính

Luồng tổng quát trong Stage 9 là:

`WorldState` → `BaselineDetector` → `SafetySupervisor` → `AuthorityStateMachine` → `ContractResolver` / `HandoffPlanner` → MPC → `ActuatorCommand`

Nguyên tắc quan trọng:

- agent chỉ được quyền tạo maneuver contract bounded
- blend chỉ thực hiện ở tầng reference/objective, không blend raw actuator
- human override luôn có ưu tiên cao nhất
- safety supervisor có quyền veto trước khi grant và trong lúc active

## Dữ liệu và artifact

Các kiểu dữ liệu chính nằm trong [stage9/schemas.py](stage9/schemas.py):

- `WorldState` — snapshot thế giới từ perception/behavior layer
- `ManeuverContract` — hợp đồng maneuver bounded cho agent
- `TrajectoryRequest` — đầu vào cho planner/MPC

- `MRCPlan` — minimal risk maneuver plan
- `ActuatorCommand` — output cuối cùng cho CARLA/MPC

Benchmark và replay dùng nhiều artifact JSON, JSONL, PNG, và frozen corpus để đánh giá ổn định.

## Yêu cầu môi trường

Tùy stage mà cần các dependency khác nhau. Với Stage 1 / Stage 4 / Stage 9, thường cần:

- Python 3.10+ hoặc theo môi trường hiện có của workspace
- `carla`
- `torch`

- `matplotlib`
- `Pillow`
- các package cho benchmark/runtime nếu chạy sâu hơn như `mmcv`, `mmdet`, `mmdet3d`

Không phải stage nào cũng có thể chạy đầy đủ chỉ từ root repo; nhiều phần phụ thuộc vào CARLA server, artifact session, hoặc repo ngoại vi như BEVFusion.

## Cách dùng repo

- Muốn hiểu perception: bắt đầu từ [carla_bevfusion_stage1/](carla_bevfusion_stage1)
- Muốn hiểu benchmark: bắt đầu từ [benchmark/README.md](benchmark/README.md)
- Muốn hiểu behavior/planning: xem [stage3/](stage3) và [stage3b/](stage3b)

- Muốn hiểu runtime online: xem [stage4/](stage4)
- Muốn hiểu takeover và safety: xem [stage9/authority_arbiter.py](stage9/authority_arbiter.py)

## Gợi ý đọc source theo thứ tự

1. [README.md](README.md) — tổng quan repo
2. [stage9/schemas.py](stage9/schemas.py) — hiểu data model trước
3. [stage9/authority_state_machine.py](stage9/authority_state_machine.py) — hiểu state transition

4. [stage9/safety_supervisor.py](stage9/safety_supervisor.py) — hiểu gate an toàn
5. [stage9/authority_arbiter.py](stage9/authority_arbiter.py) — hiểu luồng điều phối chính
6. [stage9/scenario_runner.py](stage9/scenario_runner.py) — xem các scenario test

## Ghi chú

- `outputs/` và `tmp/` chủ yếu là artifact sinh ra khi chạy thử
- benchmark và replay là phần rất quan trọng của repo, không chỉ là mã phụ trợ
- Stage 9 hiện là phần code có cấu trúc rõ nhất để nắm logic takeover/safety

## Nếu bạn muốn chạy tiếp

Mình có thể làm tiếp một trong các việc sau:

- viết thêm `README` chi tiết cho từng stage
- thêm mục `Setup` và `Run` theo đúng lệnh của repo
- rút gọn README thành bản ngắn hơn, chuyên nghiệp hơn cho GitHub

# Agent-AI

Repository này là workspace nghiên cứu cho chuỗi autonomous driving nhiều tầng: perception từ CARLA/BEVFusion, benchmark đánh giá, các lớp behavior/planning, runtime online, và Stage 9 takeover/authority protocol.

## Mục tiêu của repo

- tạo và kiểm tra artifact cho pipeline lái xe tự hành theo từng stage
- giữ các mô-đun mô phỏng, replay, benchmark và online runtime tách rõ nhau
- mô tả quyền lực theo tầng: baseline, agent, safety, human override, và minimal risk maneuver

## Cấu trúc chính

- [carla_bevfusion_stage1/](carla_bevfusion_stage1) — Stage 1 perception bridge cho CARLA + BEVFusion
  - `collector.py`, `dumper.py`, `rig.py` xử lý lấy mẫu sensor
  - `adapter.py`, `bevfusion_runtime.py`, `config_loader.py` xử lý bridge vào model
  - `coordinate_utils.py`, `visualization.py` hỗ trợ geometry và debug
- [benchmark/](benchmark) — scaffold benchmark, metric registry, frozen corpus, gate scripts, reports
- [stage2/](stage2) — lớp world-state / intent / scene abstraction
- [stage3/](stage3) — behavior planning và route-aware logic
- [stage3b/](stage3b) — builder cho behavior request và route context
- [stage3c/](stage3c) và [stage3c_coverage/](stage3c_coverage) — replay, coverage và kiểm thử theo artifact
- [stage4/](stage4) — online runtime, shadow runtime, evaluation và monitoring
- [stage9/](stage9) — authority takeover, TOR, safety veto, MRM và evaluator
- [scripts/](scripts) — CLI cho dump, replay, benchmark và runtime utilities

## Stage 9 là gì

Stage 9 là phần source code rõ nhất trong repo này: một state machine 9 trạng thái để điều phối quyền lái giữa baseline, agent, human và MRM.

Các file lõi:

- [stage9/schemas.py](stage9/schemas.py) — dataclass và enum dùng chung
- [stage9/authority_state_machine.py](stage9/authority_state_machine.py) — FSM và cooldown/hysteresis
- [stage9/authority_arbiter.py](stage9/authority_arbiter.py) — vòng lặp điều phối chính
- [stage9/safety_supervisor.py](stage9/safety_supervisor.py) — hard gate và veto
- [stage9/contract_resolver.py](stage9/contract_resolver.py) — đổi ManeuverContract sang TrajectoryRequest
- [stage9/handoff_planner.py](stage9/handoff_planner.py) — blend ở mức reference/objective
- [stage9/baseline_detector.py](stage9/baseline_detector.py) — phát hiện baseline bị kẹt / oscillation / degeneracy
- [stage9/tor_manager.py](stage9/tor_manager.py) — Takeover Request escalation
- [stage9/minimal_risk_maneuver.py](stage9/minimal_risk_maneuver.py) — fallback an toàn
- [stage9/human_override_monitor.py](stage9/human_override_monitor.py) — phát hiện can thiệp từ người lái
- [stage9/authority_logger.py](stage9/authority_logger.py) — audit log JSONL
- [stage9/stage9_evaluator.py](stage9/stage9_evaluator.py) — tính TSR, SRR, GRR, AOI, SGC, OGV, MRM
- [stage9/scenario_runner.py](stage9/scenario_runner.py) — 12 scenario giả lập T9-001 → T9-012

## Luồng chạy Stage 9

1. `WorldState` đi vào [authority_arbiter.py](stage9/authority_arbiter.py)
2. [baseline_detector.py](stage9/baseline_detector.py) quyết định baseline có bị kẹt không
3. [safety_supervisor.py](stage9/safety_supervisor.py) kiểm tra gate, freshness, ODD, TTC và confidence
4. [authority_state_machine.py](stage9/authority_state_machine.py) giữ state và cooldown
5. [contract_resolver.py](stage9/contract_resolver.py) và [handoff_planner.py](stage9/handoff_planner.py) chỉ blend ở mức trajectory reference
6. MPC xuất actuator command; agent không bao giờ tạo L3 command trực tiếp
7. [authority_logger.py](stage9/authority_logger.py) ghi audit JSONL để [stage9_evaluator.py](stage9/stage9_evaluator.py) chấm điểm

## Dạng dữ liệu chính

- `WorldState` — snapshot thế giới từ perception/behavior layer
- `ManeuverContract` — hợp đồng maneuver bounded cho agent
- `TrajectoryRequest` — đầu vào tầng trajectory/MPC
- `MRCPlan` — kế hoạch minimal risk maneuver
- `ActuatorCommand` — output cuối cùng cho CARLA/MPC

## Benchmark

Thư mục [benchmark/](benchmark) không phải code runtime lái xe, mà là scaffold đánh giá:
# Agent-AI

Agent-AI là hệ thống nghiên cứu autonomous driving nhiều tầng mà tôi đã xây dựng để nối perception, benchmark, planning, runtime online, và takeover/safety thành một pipeline thống nhất.

## Hệ thống tôi đã xây dựng

Hệ thống này không phải một ứng dụng đơn lẻ, mà là một kiến trúc theo stage:

1. **Perception layer** — lấy dữ liệu từ CARLA, chuẩn hoá sensor dump, và đẩy vào BEVFusion.
2. **World-model layer** — chuyển output perception thành world state và tactical context.
3. **Behavior layer** — sinh behavior request, route-aware logic, và replay/coverage artifact.
4. **Runtime layer** — chạy online orchestration, monitoring, shadow execution, và evaluation.
5. **Benchmark layer** — materialize frozen corpus, chạy metric, và so sánh artifact giữa các lần replay.
6. **Authority layer** — điều phối quyền lái giữa baseline, agent, human override, TOR, và minimal risk maneuver.

## Mục tiêu thiết kế

- giữ các stage tách biệt nhưng nối với nhau bằng artifact và schema rõ ràng
- cho phép replay, benchmark, và online runtime dùng cùng một ngôn ngữ dữ liệu
- đảm bảo agent chỉ được quyền ở mức maneuver bounded, không chạm trực tiếp vào actuator layer
- có safety veto, takeover request, human override, và fallback an toàn khi hệ thống không còn đáng tin

## Các phần chính của hệ thống

### Stage 1 — Perception bridge

Phần này xử lý sensor từ CARLA và map chúng vào pipeline BEVFusion. Nó gồm các thành phần lấy mẫu, đồng bộ frame, chuyển đổi toạ độ, dựng input cho model, và xuất artifact để debug geometry.

### Stage 2–4 — World model, behavior, runtime

Các stage này biến perception output thành world state, tactical context, behavior request, rồi chạy replay hoặc runtime online để theo dõi chất lượng và độ ổn định của hệ thống.

### Benchmark

Benchmark đóng vai trò chuẩn hoá đánh giá bằng frozen corpus, metric registry, scenario replay, và gate report. Đây là lớp xác nhận hệ thống hoạt động đúng theo artifact chứ không chỉ theo cảm tính.

### Stage 9 — Authority and safety

Stage 9 là lớp điều phối quyền lực trong hệ thống. Nó dùng state machine để chuyển giữa:

- baseline control
- agent requesting authority
- supervised execution
- agent active bounded
- authority revoke pending
- TOR active
- human control active
- minimal risk maneuver
- safe stop

Stage này có các thành phần chính:

- state machine để giữ trạng thái và cooldown
- safety supervisor để kiểm tra freshness, ODD, TTC, confidence, và preview feasibility
- contract resolver để chuyển maneuver contract sang trajectory request
- handoff planner để blend ở mức reference/objective, không blend actuator
- TOR manager để xử lý takeover request và timeout
- MRM executor để đưa xe về trạng thái an toàn
- logger và evaluator để audit và chấm metric

## Luồng hệ thống

Luồng tổng quát của hệ thống là:

`sensor dump` → `perception` → `world state` → `behavior request` → `authority/safety decision` → `trajectory request` → `MPC` → `actuator command`

Điểm quan trọng nhất là agent không được đi thẳng từ perception sang actuator. Mọi quyền của agent đều bị chặn ở tầng hợp đồng bounded và tầng an toàn.

## Dữ liệu trung tâm

Hệ thống này xoay quanh một bộ schema rõ ràng:

- `WorldState` — trạng thái thế giới từ perception và runtime
- `ManeuverContract` — hợp đồng maneuver bounded của agent
- `TrajectoryRequest` — đầu vào cho planner/MPC
- `MRCPlan` — kế hoạch minimal risk maneuver
- `ActuatorCommand` — lệnh cuối cùng cho xe

## Điểm mạnh của hệ thống

- pipeline theo stage rõ ràng, dễ replay và debug
- dùng artifact thay vì chỉ dùng trạng thái trong bộ nhớ
- có benchmark và evaluator riêng cho từng lớp
- có cơ chế safety và fallback để không giao toàn quyền cho agent

## Cách hiểu nhanh

Nếu muốn hiểu hệ thống theo thứ tự hợp lý, hãy đi theo:

1. schema dữ liệu
2. state machine authority
3. safety supervisor
4. arbiter điều phối
5. scenario runner và evaluator

Đây là mô tả đúng phần hệ thống tôi đã xây dựng trong repo này: một kiến trúc autonomous driving theo stage, có perception, benchmark, behavior, runtime, và authority/safety được tách rõ.
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
