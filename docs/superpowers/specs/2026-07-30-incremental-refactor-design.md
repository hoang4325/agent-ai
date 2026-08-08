# Refactor Design: Agent-AI Codebase — Incremental by Stage

**Date:** 2026-07-30
**Approach:** Hướng A — Incremental Refactor theo Stage
**Scope:** Toàn bộ codebase (stage9 → stage4 → stage3b/3c → stage2 → stage1 → benchmark → scripts)
**Goals:** Cấu trúc & tổ chức code + Hiệu năng & tối ưu + Testability & maintainability

## 1. Hiện trạng

- Codebase nghiên cứu Agentic AI cho xe tự lái trên CARLA/BEVFusion
- ~3000 LOC stage9, tổng cộng ~50+ file Python across 7 stages + benchmark
- Chưa có test suite hoặc CI pipeline
- Kiến trúc phân tầng rõ ràng về mặt concept nhưng chưa được tổ chức thành package có cấu trúc
- Một số file lớn (>300 LOC): scenario_runner.py (583), authority_arbiter.py (337), schemas.py (300)
- Interface giữa các stage chủ yếu qua dict, thiếu type safety

## 2. Thứ tự Refactor

Refactor từ dưới lên theo mức độ ảnh hưởng và độ ổn định của interface:

1. **stage9 (authority/safety)** — trung tâm điều phối, schemas rõ ràng nhất
2. **stage4 (runtime)** — online orchestration, shadow execution
3. **stage3b/stage3c (behavior/route/replay)** — behavior decision, route awareness
4. **stage2 (world_state)** — perception output → world state
5. **carla_bevfusion_stage1 (perception)** — sensor bridge, BEVFusion adapter
6. **benchmark** — replay, metrics, gate scripts
7. **scripts** — utilities, helpers

## 3. Quy trình cho mỗi Stage

Mỗi stage lặp lại chu kỳ 4 bước nghiêm ngặt:

### Bước 1: Type & Schema
- Thêm type hints cho tất cả function signatures và biến quan trọng
- Chuyển dict-based data structures sang dataclass hoặc pydantic BaseModel
- Chuẩn hóa interface giữa các module trong stage
- Định nghĩa Protocol/ABC cho cross-stage contracts nếu cần

### Bước 2: Tách Module
- Chia file >300 LOC thành các module nhỏ hơn theo single responsibility
- Extract shared utilities vào `shared/` package
- Tách logic thuần (pure functions) khỏi I/O-dependent code
- Giảm coupling giữa các component trong stage

### Bước 3: Test Song Song
- Viết unit test cho pure logic (không phụ thuộc CARLA/BEVFusion)
- Viết integration test nhẹ cho interface giữa các module
- Mock external dependencies (CARLA server, sensor data) khi cần
- Target coverage: logic cốt lõi trước, edge cases sau

### Bước 4: Verify & Commit
- Chạy toàn bộ test hiện có + test mới
- Chạy lint (ruff) + type check (mypy)
- Đảm bảo không regression so với trước refactor
- Commit riêng biệt cho mỗi bước trong chu kỳ

## 4. Cấu trúc Package Mục Tiêu

```
agent_ai/
├── __init__.py
├── authority/          # stage9
│   ├── schemas.py
│   ├── state_machine.py
│   ├── arbiter.py
│   ├── safety_supervisor.py
│   ├── baseline_detector.py
│   ├── contract_resolver.py
│   ├── handoff_planner.py
│   ├── tor_manager.py
│   ├── minimal_risk_maneuver.py
│   ├── human_override_monitor.py
│   ├── evaluator.py
│   └── logger.py
├── runtime/            # stage4
├── behavior/           # stage3b/3c
├── world_state/        # stage2
├── perception/         # carla_bevfusion_stage1
├── benchmark/          # benchmark + eval
├── shared/             # utils, types, config chung
│   ├── types.py
│   ├── config.py
│   └── coordinate_utils.py
└── scripts/
```

**Lưu ý:** Giữ nguyên tên file hiện tại trong giai đoạn đầu. Chỉ di chuyển vào package mới khi interface đã ổn định và test pass.

## 5. Ràng buộc An Toàn

- **Không đổi logic hành vi** — chỉ đổi cấu trúc, giữ nguyên semantics
- **Backward compatibility** — giữ import paths cũ hoạt động trong quá trình chuyển đổi (dùng re-export)
- **Mỗi commit phải pass test + lint** — không bao giờ commit code broken
- **Không refactor cross-stage cùng lúc** — hoàn thành một stage trước khi sang stage tiếp theo
- **Giữ khả năng chạy thử nghiệm** — hệ thống phải luôn ở trạng thái có thể chạy được sau mỗi bước

## 6. Công cụ & Dependencies

- **Testing:** pytest (chọn vì phổ biến trong Python ecosystem, hỗ trợ fixtures/mock tốt)
- **Linting:** ruff (nhanh, thay thế flake8/isort/black)
- **Type checking:** mypy (strict mode cho code mới, gradual cho code cũ)
- **Data validation:** pydantic v2 (cho schemas phức tạp) hoặc dataclasses (cho cấu trúc đơn giản)
- Kiểm tra xem codebase hiện tại đã dùng thư viện nào trước khi thêm dependency mới

## 7. Milestones

| Milestone | Nội dung | Điều kiện hoàn thành |
|-----------|----------|---------------------|
| M1 | Stage9 refactor xong | Tất cả stage9 files typed, modularized, tested |
| M2 | Stage4 + Stage3b/3c refactor xong | Interface stage9↔stage4↔stage3 ổn định |
| M3 | Stage2 + Stage1 refactor xong | End-to-end pipeline chạy được với cấu trúc mới |
| M4 | Benchmark + Scripts refactor xong | Toàn bộ codebase migrated, test green |
| M5 | Cleanup & documentation | Xóa deprecated imports, cập nhật README |

## 8. Rủi ro & Mitigation

| Rủi ro | Mitigation |
|--------|-----------|
| Regression do thiếu test | Viết test TRƯỚC khi refactor mỗi module |
| Circular imports khi tái cấu trúc package | Dùng lazy imports hoặc extract shared types |
| CARLA/BEVFusion không available để test | Mock layer cho external deps, test offline |
| Scope creep (refactor thêm tính năng mới) | Strict rule: chỉ refactor cấu trúc, không thêm feature |

</parameter>