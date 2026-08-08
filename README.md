# Agent-AI: Bounded Agentic AI Framework for Autonomous Driving

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-88%20passing-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-Authority%20%26%20Safety%20Layer-orange.svg)]()

**Agent-AI** là workspace nghiên cứu và thử nghiệm ứng dụng **Agentic AI** vào hệ thống xe tự lái (Autonomous Driving). Thay vì giao phó toàn bộ quyền điều khiển xe cho một mô hình end-to-end hoặc một planner tĩnh, Agent-AI áp dụng kiến trúc **Bounded Agentic Decision**: Agent đóng vai trò lớp suy luận chiến thuật (Tactical Reasoning Layer), đề xuất hành vi thông qua hợp đồng có giới hạn (**Bounded Maneuver Contract**), và luôn đặt dưới sự giám sát nghiêm ngặt của **Safety Supervisor**, **Authority State Machine**, và **MPC Controller**.

Workspace tích hợp thử nghiệm trên giả lập **CARLA**, kết nối cảm biến qua **BEVFusion**, xử lý runtime online/shadow execution, cùng hệ thống **Benchmark & Replay Corpus** phục vụ đánh giá an toàn.

---

## 1. Triết lý Thiết kế & 3 Tầng Quyền Lực (Authority Tiers)

Để đảm bảo xe không bao giờ bị mất kiểm soát bởi các quyết định không xác định từ Agent, hệ thống phân chia quyền lực thành 3 tầng phân cấp tuyệt đối:

| Tầng | Tên Quyền Hạn | Chủ Thể Nắm Giữ | Phạm Vi & Trách Nhiệm |
| :--- | :--- | :--- | :--- |
| **L1** | **Maneuver Authority** | Agentic AI | Đề xuất ý định hành vi chiến thuật (vd: `keep_lane`, `change_lane_left`, `decelerate_stop`, `yield`) qua hợp đồng **ManeuverContract**. |
| **L2** | **Trajectory Authority** | Trajectory Planner / MPC | Chuyển đổi hợp đồng được duyệt thành quỹ đạo hình học ($x, y, v, \theta$) tối ưu toán học. |
| **L3** | **Actuator Authority** | Controller $\rightarrow$ CARLA | Xuất tín hiệu điều khiển phần cứng trực tiếp (`steer`, `throttle`, `brake`). |

> ⚠️ **Nguyên tắc cốt lõi:** Agent **chỉ sở hữu L1**, tuyệt đối **không được phép can thiệp trực tiếp vào L2 hoặc L3**. Mọi đề xuất L1 của Agent bắt buộc phải đi qua **Safety Supervisor** kiểm duyệt trước khi được thực thi.

---

## 2. Kiến Trúc Tổng Thể & Luồng Dữ Liệu

Pipeline hoạt động của hệ thống được tổ chức thành các khối chức năng chính:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CARLA Sensor Rig / Bridge                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Sensor Dump (LiDAR, Camera, Radar)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Perception Bridge (BEVFusion Runtime & Coordinate Transformer)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 3D Bounding Boxes & Map Features
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 2: World State Engine (Object Tracking, Scene & Risk Summary)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ WorldState Payload
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Tactical Behavior Layer (Lane Topology, Route Context)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Tactical Context
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Agentic Reasoning & Online Shadow Execution Runtime           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Proposed ManeuverContract
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 9: Safety Supervisor & Authority State Machine (ASM)             │
│   - Check Veto: Freshness, ODD, TTC, Geometry & Confidence Gates       │
│   - Decision: Baseline Active / Agent Granted / TOR Issued / MRM       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Approved Maneuver Contract
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Contract Resolver & MPC Controller ──► Output Actuators (Steer/Throttle)│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Cấu Trúc Package chuẩn (`agent_ai/`)

Mã nguồn được tái cấu trúc hoàn toàn theo domain chức năng, loại bỏ các package prefix `stage*` cũ:

| Package / Module | Trách Nhiệm Kỹ Thuật |
| :--- | :--- |
| **`agent_ai.perception`** | Sensor collection, rig calibration, BEVFusion inference bridge & geometry utilities. |
| **`agent_ai.world_state`** | Object tracking, scene representation, spatial-temporal risk modeling. |
| **`agent_ai.behavior`** | Phân mảnh thành các subpackage:<br>• `.lane`: Ngữ cảnh làn đường & lane topology.<br>• `.route`: Lập kế hoạch hành vi theo lộ trình.<br>• `.execution`: Local planner & execution helpers.<br>• `.coverage`: Tooling kiểm thử độ bao phủ kịch bản. |
| **`agent_ai.runtime`** | Vòng lặp điều phối online 10Hz, shadow execution, monitoring & metrics logger. |
| **`agent_ai.authority`** | Lớp an toàn trung tâm:<br>• `authority_state_machine.py`: Trạng thái ASM & chuyển vùng quyền lực.<br>• `safety_supervisor.py`: Động cơ Veto với các hard safety gates.<br>• `authority_arbiter.py`: Điều phối luồng 10Hz.<br>• `baseline_detector.py`: Phát hiện baseline bị stuck/dao động.<br>• `contract_resolver.py`: Chuyển contract thành Trajectory Request.<br>• `tor_manager.py` & `minimal_risk_maneuver.py`: Xử lý Takeover Request & MRM fallback. |
| **`agent_ai.benchmark`** | Frozen corpus, metric registry, gate evaluations (`gates/`, `shadow/`, `takeover/`, `assist/`). |
| **`agent_ai.shared`** | I/O artifacts, numeric math, common data schemas, logging. |
| **`agent_ai.cli`** | Endpoint CLI thống nhất cho toàn bộ hệ thống. |

---

## 4. Hướng Dẫn Sử Dụng CLI (`agent_ai.cli`)

Hệ thống cung cấp giao diện dòng lệnh thống nhất qua module `agent_ai.cli`:

```bash
# 1. Liệt kê toàn bộ CLI commands hỗ trợ
python -m agent_ai.cli list

# 2. Xem trợ giúp chi tiết từng command
python -m agent_ai.cli world_replay --help

# 3. Chạy Replay Stage 2 dựa trên kết quả Stage 1 session
python -m agent_ai.cli world_replay \
  --stage1-session /path/to/stage1_session \
  --output-dir /tmp/stage2_output

# 4. Chạy Online Runtime Orchestrator
python -m agent_ai.cli online_runtime \
  --output-dir /tmp/online_runtime_out

# 5. Đánh giá chiến dịch Authority & Safety (Stage 9 Campaign)
python -m agent_ai.cli authority_campaign

# 6. Chạy Gate kiểm định Shadow Execution
python -m agent_ai.cli shadow_gate
```

---

## 5. Cơ Chế An Toàn & Quản Lý Trạng Thái (Authority State Machine - ASM)

Trạng thái điều phối quyền kiểm soát xe được quản lý bởi **Authority State Machine (ASM)** gồm các trạng thái chính:

1. **`BASELINE_ACTIVE`**: Hệ thống baseline mặc định nắm quyền lái xe.
2. **`AGENT_REQUESTED`**: Agent phát hiện cơ hội/sự cố và xin cấp quyền **ManeuverContract**.
3. **`AGENT_ACTIVE`**: Safety Supervisor duyệt contract; Agent tạm thời giữ quyền L1.
4. **`HUMAN_OVERRIDE`**: Người lái can thiệp tay; vĩnh viễn tước quyền Agent & Baseline.
5. **`TOR_ISSUED`**: Cảnh báo **Takeover Request** gửi tới tài xế khi gặp tình huống nguy hiểm ngoài ODD.
6. **`MRM_EXECUTING`**: Kích hoạt **Minimal Risk Maneuver** (tấp lề an toàn/dừng khẩn cấp) nếu tài xế không phản hồi TOR.

### Các Safety Gates Độc Lập
- **Freshness Gate**: Hủy bỏ quyết định nếu dữ liệu perception quá $N$ frames.
- **TTC (Time-to-Collision) Gate**: Phủ quyết hành vi nếu TTC rơi vào vùng nguy hiểm ($< 2.5s$).
- **Geometry & ODD Gate**: Kiểm tra giới hạn làn, chướng ngại vật & vùng hoạt động an toàn.
- **Confidence Gate**: Đảm bảo độ tin cậy nhận dạng cảm biến đạt ngưỡng quy định.

---

## 6. Đánh Giá & Benchmark (`agent_ai.benchmark`)

Hệ thống đánh giá sự ổn định và an toàn thông qua:
- **Corpus Replay**: Chạy lại các kịch bản giao thông nguy hiểm đã đóng băng.
- **Metrics Registry**: Theo dõi tỉ lệ Grant/Revoke contract, tần suất TOR, độ lệch quỹ đạo, điểm vi phạm hình học, và mức độ ổn định hành vi.
- **Shadow Gate Execution**: So sánh song song quyết định của Agent với Baseline mà không ảnh hưởng tới luồng điều khiển thực tế.

---

## 7. Kiểm Thử Hệ Thống (Testing)

Bộ test toàn diện bảo đảm tính đúng đắn cho các module thuật toán, MPC bounds, CLI và package layout:

```bash
# Chạy toàn bộ unit test suite
PYTHONPATH=. python3 -m unittest discover tests
```

---

## 8. Lộ Trình Đọc Mã Nguồn Cho Developer

Khuyến nghị thứ tự nghiên cứu mã nguồn repo:

1. [README.md](file:///home/hoangnh/Documents/agent-ai/README.md): Nắm tổng quan kiến trúc và triết lý an toàn.
2. [agent_ai/authority/schemas.py](file:///home/hoangnh/Documents/agent-ai/agent_ai/authority/schemas.py): Định nghĩa data structure (`ManeuverContract`, `WorldState`, `SafetyVerdict`).
3. [agent_ai/authority/authority_state_machine.py](file:///home/hoangnh/Documents/agent-ai/agent_ai/authority/authority_state_machine.py): Logic chuyển trạng thái ASM.
4. [agent_ai/authority/safety_supervisor.py](file:///home/hoangnh/Documents/agent-ai/agent_ai/authority/safety_supervisor.py): Bộ kiểm duyệt an toàn và quy tắc phủ quyết.
5. [agent_ai/authority/authority_arbiter.py](file:///home/hoangnh/Documents/agent-ai/agent_ai/authority/authority_arbiter.py): Vòng lặp điều phối chính (10Hz Orchestration).
6. [agent_ai/benchmark/](file:///home/hoangnh/Documents/agent-ai/agent_ai/benchmark/): Quy trình replay và đánh giá bằng metrics.
7. [agent_ai/perception/](file:///home/hoangnh/Documents/agent-ai/agent_ai/perception/) & [agent_ai/runtime/](file:///home/hoangnh/Documents/agent-ai/agent_ai/runtime/): Cầu nối cảm biến và runtime online.

---

## 9. Tóm Tắt

**Agent-AI** chứng minh phương pháp đưa **Agentic AI** vào hệ thống tự lái một cách an toàn và có thể giải thích được. Thay vì một "hộp đen" end-to-end, Agent hoạt động như một lớp lập luận chiến thuật có giới hạn, đảm bảo luôn đáp ứng các tiêu chuẩn an toàn kỹ thuật khắt khe.
