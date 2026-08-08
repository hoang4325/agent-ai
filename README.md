# Agent-AI — Agentic AI cho xe tự lái

Agent-AI là workspace nghiên cứu cách ứng dụng **Agentic AI** vào hệ thống xe tự lái. Thay vì chỉ dùng một mô hình dự đoán hoặc một planner cố định, repo này mô tả một kiến trúc nhiều tầng, trong đó agent có thể quan sát môi trường, hiểu trạng thái thế giới, đề xuất hành vi lái, kiểm tra rủi ro, chạy replay/benchmark và phối hợp với lớp takeover/safety.

Mục tiêu chính của repo là xây dựng một pipeline thử nghiệm cho autonomous driving trên CARLA, BEVFusion, runtime online, benchmark artifact và cơ chế điều phối quyền lái an toàn.

## Agentic AI trong xe tự lái là gì?

Trong repo này, Agentic AI không được hiểu là một AI điều khiển trực tiếp vô-lăng, ga hoặc phanh. Agent chỉ hoạt động ở tầng quyết định có ràng buộc. Nó có thể đề xuất một hành động lái dưới dạng **bounded maneuver contract**, ví dụ như tiếp tục đi thẳng, giảm tốc, đổi làn, nhường đường hoặc chuẩn bị dừng an toàn.

Mọi đề xuất của agent đều phải đi qua safety supervisor, authority state machine, trajectory planner và MPC trước khi trở thành actuator command. Điều này giúp hệ thống tận dụng khả năng lập luận của agent nhưng vẫn giữ lớp an toàn tách biệt và có quyền phủ quyết.

## Kiến trúc tổng thể

Pipeline của hệ thống được chia thành nhiều stage rõ ràng:

1. **Perception layer** — lấy dữ liệu từ CARLA, chuẩn hóa sensor dump và đưa vào BEVFusion.
2. **World-state layer** — chuyển output perception thành trạng thái thế giới, object tracking, scene summary và risk summary.
3. **Behavior layer** — tạo behavior request, hiểu route/lane context và chuẩn bị dữ liệu cho planning.
4. **Agentic decision layer** — agent phân tích tình huống, đề xuất maneuver contract và lý do hành động.
5. **Runtime layer** — chạy online orchestration, shadow execution, monitoring và evaluation.
6. **Benchmark layer** — dùng frozen corpus, replay, metric registry và gate report để kiểm tra chất lượng.
7. **Authority/safety layer** — điều phối quyền lái giữa baseline, agent, human override, TOR và minimal risk maneuver.

Luồng dữ liệu tổng quát:

```text
sensor dump
→ perception
→ world state
→ tactical context
→ agentic behavior decision
→ bounded maneuver contract
→ safety supervisor
→ trajectory request
→ MPC
→ actuator command
```

Điểm quan trọng là agent không được đi thẳng từ perception sang actuator. Agent chỉ được đề xuất hành vi ở tầng contract, còn quyền thực thi cuối cùng thuộc về safety, planner và controller.

## Cấu trúc package chuẩn (`agent_ai/`)

Source chính nằm trong package `agent_ai` với tên semantic. Tên stage cũ vẫn còn dưới dạng **shim** để script/import legacy không gãy.

| Canonical package | Legacy name | Vai trò |
|-------------------|-------------|---------|
| `agent_ai.perception` | `carla_bevfusion_stage1` | Sensor bridge + BEVFusion |
| `agent_ai.world_state` | `stage2` | Tracking, risk, world state |
| `agent_ai.behavior.lane` | `stage3` | Lane context + behavior v1 |
| `agent_ai.behavior.route` | `stage3b` | Route-aware behavior v2 |
| `agent_ai.behavior.execution` | `stage3c` | Execution / local planner |
| `agent_ai.behavior.coverage` | `stage3c_coverage` | Planner coverage tooling |
| `agent_ai.runtime` | `stage4` | Online orchestration + shadow |
| `agent_ai.authority` | `stage9` | Authority / safety / TOR / MRM |
| `agent_ai.benchmark` | `benchmark` | Cases, metrics, gates, corpus |
| `agent_ai.shared` | `common` | I/O, numeric, ports, logging |

Ví dụ import mới:

```python
from agent_ai.runtime.online_orchestrator import Stage4OnlineOrchestrator
from agent_ai.authority import AuthorityArbiter, ManeuverContract
from agent_ai.shared.artifact_io import write_json
```

## Các stage chính

### Stage 1 — Perception bridge (`agent_ai.perception`)

Stage 1 kết nối CARLA sensor dump với BEVFusion runtime. Thành phần này xử lý việc lấy mẫu sensor, đồng bộ frame, chuyển đổi tọa độ, tạo input cho model và sinh artifact phục vụ debug geometry.

Các nhóm file tiêu biểu:

- `collector.py`, `dumper.py`, `rig.py` — lấy mẫu và đồng bộ sensor
- `adapter.py`, `bevfusion_runtime.py`, `config_loader.py` — bridge dữ liệu vào model
- `coordinate_utils.py`, `visualization.py` — xử lý tọa độ và trực quan hóa

### Stage 2 — World state (`agent_ai.world_state`)

Stage 2 biến output perception thành dữ liệu có thể dùng cho reasoning và planning, gồm normalized prediction, tracked objects, scene summary, risk summary, world state và planner interface payload.

Đây là lớp giúp agent không phải xử lý trực tiếp raw sensor, mà làm việc trên biểu diễn thế giới đã được chuẩn hóa.

### Stage 3 / 3B / 3C — Behavior (`agent_ai.behavior.*`)

Các stage này bổ sung ngữ cảnh về lane, route, maneuver validation, behavior request và replay artifact. Mục tiêu là giúp agent hiểu tình huống lái theo bối cảnh giao thông thay vì chỉ nhìn object rời rạc.

Stage 3C và coverage được dùng để kiểm tra chất lượng planner, độ bao phủ scenario và độ ổn định của behavior decision.

### Stage 4 — Online runtime (`agent_ai.runtime`)

Stage 4 tập trung vào runtime online, shadow execution, monitoring và evaluation. Đây là lớp cho phép so sánh quyết định của agent trong môi trường chạy thật hoặc replay mà chưa cần giao toàn quyền điều khiển.

Shadow execution đặc biệt quan trọng vì nó cho phép đánh giá agent trong nhiều tình huống trước khi cho agent tham gia vào luồng điều khiển chính.

### Stage 9 — Authority and safety (`agent_ai.authority`)

Stage 9 là lớp điều phối quyền lái. Nó quyết định khi nào baseline tiếp tục điều khiển, khi nào agent được đề xuất maneuver, khi nào cần yêu cầu người lái takeover, và khi nào phải kích hoạt minimal risk maneuver.

Các thành phần chính:

- `authority_state_machine.py` — quản lý trạng thái quyền lái, cooldown và hysteresis
- `authority_arbiter.py` — vòng lặp điều phối chính
- `safety_supervisor.py` — kiểm tra hard gate, freshness, ODD, TTC, confidence và feasibility
- `baseline_detector.py` — phát hiện baseline bị kẹt, dao động hoặc suy giảm
- `contract_resolver.py` — chuyển maneuver contract thành trajectory request
- `handoff_planner.py` — blend ở mức reference/objective, không blend raw actuator
- `tor_manager.py` — quản lý takeover request và escalation
- `minimal_risk_maneuver.py` — fallback an toàn khi hệ thống không còn đáng tin
- `human_override_monitor.py` — phát hiện can thiệp của người lái
- `authority_logger.py` — ghi audit log JSONL
- `stage9_evaluator.py` — đánh giá bằng metric

## Vai trò của Agentic AI

Agentic AI trong hệ thống này có thể đảm nhận các vai trò sau:

- phân tích world state và tactical context để hiểu tình huống lái
- phát hiện khi baseline đang bị kẹt hoặc đưa ra hành vi không ổn định
- đề xuất maneuver contract có giới hạn rõ ràng
- giải thích lý do chọn hành vi dựa trên risk, lane, route và object context
- chạy ở chế độ shadow để so sánh với baseline mà chưa can thiệp thật
- tạo artifact phục vụ replay, benchmark và evaluation
- phối hợp với authority layer để xin quyền, nhường quyền hoặc kích hoạt takeover

Tuy nhiên, agent không được phép bypass safety supervisor, không được xuất actuator command trực tiếp và không được giữ quyền điều khiển nếu dữ liệu perception, ODD, TTC hoặc confidence không đạt ngưỡng an toàn.

## Nguyên tắc an toàn

Hệ thống được thiết kế theo hướng agent có năng lực lập luận nhưng bị kiểm soát bởi các ràng buộc an toàn:

- agent chỉ đề xuất maneuver bounded, không điều khiển actuator trực tiếp
- safety supervisor có quyền veto trước và trong lúc agent active
- human override luôn có ưu tiên cao nhất
- nếu takeover timeout hoặc rủi ro tăng cao, hệ thống chuyển sang minimal risk maneuver
- mọi quyết định quan trọng đều được ghi log để audit và replay
- benchmark và frozen corpus được dùng để kiểm tra hồi quy trước khi chạy online

## Benchmark và đánh giá

Benchmark không chỉ là phần phụ trợ, mà là lớp trung tâm để kiểm tra chất lượng hệ thống. Repo sử dụng replay corpus, metric registry, gate scripts và report để đánh giá các thay đổi giữa nhiều lần chạy.

Các metric có thể dùng để đánh giá gồm tỷ lệ grant/revoke, takeover success, safety gate consistency, obstacle geometry violation, minimal risk maneuver activation và độ ổn định của behavior decision.

## Giới hạn hiện tại

Repo này là workspace nghiên cứu, không phải hệ thống autonomous driving production. Một số giới hạn cần lưu ý:

- chưa phải ứng dụng end-to-end chạy hoàn chỉnh từ root repo
- phụ thuộc vào CARLA server, BEVFusion và các artifact session
- dữ liệu CARLA có thể lệch domain so với dữ liệu huấn luyện thực tế
- radar, coordinate frame và sweep timing cần được kiểm tra kỹ khi replay
- agentic layer chỉ nên dùng cho nghiên cứu decision/planning, không dùng để điều khiển xe thật

## Cách đọc repo

Thứ tự đọc hợp lý:

1. `README.md` để nắm tổng quan + bảng package map
2. `agent_ai/authority/schemas.py` để hiểu data model
3. `agent_ai/authority/authority_state_machine.py` để hiểu state transition
4. `agent_ai/authority/safety_supervisor.py` để hiểu safety gate
5. `agent_ai/authority/authority_arbiter.py` để hiểu vòng điều phối quyền lái
6. `agent_ai/authority/scenario_runner.py` để xem các scenario test
7. `agent_ai/benchmark/` để hiểu cách replay và đánh giá
8. `agent_ai/perception/` để hiểu perception bridge
9. `agent_ai/runtime/` để hiểu online orchestration

## Tóm tắt

Agent-AI là một kiến trúc nghiên cứu ứng dụng Agentic AI vào xe tự lái theo hướng an toàn và có kiểm soát. Agent không thay thế toàn bộ stack lái xe, mà đóng vai trò như một lớp reasoning và behavior decision nằm giữa world state và authority/safety layer.

Cách thiết kế này giúp kết hợp khả năng lập luận của agent với các cơ chế kiểm soát truyền thống như state machine, safety veto, trajectory planner, MPC, replay benchmark và human override.
