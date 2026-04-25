# 📖 Giải Thích Chi Tiết Từng Thông Số — Agent-AI Research Paper

---

## NHÓM A — WORLD MODEL (Nhận Thức Thế Giới)

---

### [critical_actor_detected_rate](file:///d:/Agent-AI/benchmark/metrics.py#342-357)
**Tỷ lệ phát hiện actor nguy hiểm**

- **Công thức**: `số frame phát hiện được actor nguy hiểm / tổng số frame cần phát hiện`
- **Ý nghĩa**: Đo khả năng của hệ thống trong việc "nhìn thấy" các phương tiện quan trọng (xe đang chặn đường, xe phía trước cần theo dõi). Nếu hệ thống không phát hiện được actor nguy hiểm, toàn bộ pipeline phía sau (lên kế hoạch, điều khiển) sẽ đưa ra quyết định sai.
- **Ví dụ dễ hiểu**: Giống như hỏi "Trong 100 giây lái xe, bạn nhận ra xe phía trước bao nhiêu giây?" — nếu chỉ 60 giây thì nguy hiểm vì có 40 giây "mù".
- **Giá trị mong muốn**: Càng gần 1.0 (100%) càng tốt. Đây là **hard gate** — nếu không đạt thì toàn bộ benchmark fail.
- **Cách viết trong bài báo**: *"The system achieved a critical_actor_detected_rate of X across all benchmark scenarios, demonstrating robust perception coverage for safety-critical actors."*

---

### [blocker_binding_accuracy](file:///d:/Agent-AI/benchmark/metrics.py#359-381)
**Độ chính xác gán nhãn phương tiện chặn đường**

- **Công thức**: [(số frame gán đúng lane cho blocker) / (tổng frame eligible)](file:///d:/Agent-AI/benchmark/metrics.py#19-23), với "đúng" = bound+đúng lane (1.0), ambiguous+đúng lane (0.5), stale+đúng lane (0.5)
- **Ý nghĩa**: Không chỉ "có thấy" xe phía trước, mà còn phải biết xe đó đang ở **làn đường nào**. Nếu xe blocker đang ở làn trái mà hệ thống nghĩ là làn phải → lên kế hoạch sai hoàn toàn.
- **Ví dụ dễ hiểu**: Camera nhìn thấy xe tải, nhưng nghĩ xe tải đang chặn làn phải trong khi thực ra nó chặn làn trái → xe tự lái chuyển sang làn trái → va chạm.
- **Giá trị mong muốn**: ↑ cao hơn tốt hơn (soft guardrail).

---

### [critical_actor_track_continuity_rate](file:///d:/Agent-AI/benchmark/metrics.py#403-424)
**Tỷ lệ theo dõi liên tục actor nguy hiểm**

- **Công thức**: `số lần chuyển frame mà giữ nguyên track ID / tổng số lần chuyển frame`
- **Ý nghĩa**: Khi hệ thống đang theo dõi một chiếc xe (track ID = 5), sang frame tiếp theo nó vẫn phải nhận ra đó là cùng một chiếc xe (track ID = 5). Nếu cứ thay đổi ID liên tục → hệ thống "quên" xe cũ và "phát hiện lại" như xe mới → ảnh hưởng đến dự đoán tốc độ, quỹ đạo.
- **Ví dụ dễ hiểu**: Như một cầu thủ bị mất dấu giữa chừng — huấn luyện viên không biết cầu thủ đó đang ở đâu dù vẫn thấy có người trên sân.
- **Giá trị mong muốn**: ↑ cao hơn tốt hơn (soft guardrail).

---

### [lane_assignment_consistency](file:///d:/Agent-AI/benchmark/metrics.py#383-401)
**Tính nhất quán của gán nhãn làn đường**

- **Công thức**: `số frame gán đúng lane relation / tổng frame bound`
- **Ý nghĩa**: Qua nhiều frame, hệ thống phải nhất quán trong việc xác định actor đang ở làn nào so với xe chủ. Nếu frame này nói "xe đó ở làn trái" nhưng frame sau lại "ở làn phải" dù xe không đổi làn → thông tin không đáng tin cậy.
- **Giá trị mong muốn**: ↑ cao hơn tốt hơn.

---

## NHÓM B — BEHAVIOR PLANNING (Lên Kế Hoạch Hành Vi)

---

### [route_option_match](file:///d:/Agent-AI/benchmark/metrics.py#102-108)
**Tỷ lệ chọn đúng lộ trình**

- **Công thức**: `1.0 nếu route được chọn nằm trong expected routes, 0.0 nếu không`
- **Ý nghĩa**: Ở các điểm phân nhánh (ngã tư, nút giao), hệ thống phải chọn đúng hướng theo lộ trình đã định. Đây là tín hiệu quan trọng nhất để đánh giá khả năng "biết mình đang đi đâu" của agent.
- **Ví dụ dễ hiểu**: GPS nói rẽ trái ở ngã tư, nhưng xe lại đi thẳng — dù có đi an toàn thì vẫn sai mục tiêu.
- **Giá trị mong muốn**: = 1.0 (hard gate).

---

### [expected_behavior_presence](file:///d:/Agent-AI/benchmark/metrics.py#110-116)
**Số lần thực hiện behavior mong đợi**

- **Công thức**: Đếm tổng số lần các behavior trong danh sách `expected_behaviors` được thực hiện.
- **Ý nghĩa**: Mỗi test case định nghĩa behavior nào là "đúng" (ví dụ: trong scenario cần vượt xe thì phải có `commit_lane_change_left`). Metric này đếm xem agent có thực sự làm những việc cần làm hay không.
- **Ví dụ dễ hiểu**: Trong bài thi lái xe thực hành, giám khảo chấm điểm từng thao tác bắt buộc (nhìn gương, xi nhan, bẻ lái). Metric này đếm bao nhiêu thao tác bắt buộc đã làm.
- **Giá trị mong muốn**: ↑ số lượng càng nhiều càng tốt (hard gate).

---

### [forbidden_behavior_count](file:///d:/Agent-AI/benchmark/metrics.py#118-124)
**Số lần thực hiện behavior bị cấm**

- **Công thức**: Đếm tổng số lần các behavior trong danh sách `forbidden_behaviors` xuất hiện.
- **Ý nghĩa**: Đây là danh sách "tuyệt đối không được làm". Ví dụ: trong scenario đang dừng tại đèn đỏ, không được phép có `commit_lane_change_left`. Bất kỳ vi phạm nào đều là lỗi nghiêm trọng.
- **Giá trị mong muốn**: = 0 (hard gate — nếu > 0 là fail ngay).

---

### [unsafe_lane_change_intent_count](file:///d:/Agent-AI/benchmark/metrics.py#126-140)
**Số lần chuyển làn khi không được phép**

- **Công thức**: Đếm số frame có `prepare/commit lane change left/right` trong khi `lane_change_permission.left/right == false`
- **Ý nghĩa**: Hệ thống kiểm tra an toàn (lane permission checker) xác định làn đó có xe không, có an toàn không. Nếu chưa được phép mà agent vẫn cố chuyển làn → nguy hiểm thực sự.
- **Giá trị mong muốn**: = 0 (hard gate về an toàn).

---

### [route_conditioned_override_rate](file:///d:/Agent-AI/benchmark/metrics.py#159-163)
**Tỷ lệ Stage 3B override Stage 3A vì route**

- **Công thức**: `số frame Stage 3B thay đổi quyết định của Stage 3A vì route / tổng frame`
- **Ý nghĩa**: Stage 3A tạo behavior request thuần túy dựa trên scene. Stage 3B "lọc" lại dựa trên route. Nếu override quá nhiều → Stage 3A đang không route-aware. Nếu override quá ít → Stage 3B có thể không phát huy tác dụng.
- **Giá trị mong muốn**: Phụ thuộc test case (case-defined).

---

### [route_context_consistency](file:///d:/Agent-AI/benchmark/metrics.py#473-486)
**Tính nhất quán của route binding**

- **Công thức**: `1.0 nếu route option + preferred lane + không có conflict flags + có source, 0.0 nếu không`
- **Ý nghĩa**: Kiểm tra hệ thống có "nhất quán" về định nghĩa route không — route từ đâu ra, ưu tiên làn nào, có mâu thuẫn không.
- **Giá trị mong muốn**: = 1.0 (hard gate cho các case route-conditioned).

---

## NHÓM C — EXECUTION (Thực Thi Điều Khiển)

---

### [lane_change_success_rate](file:///d:/Agent-AI/benchmark/metrics.py#172-189)
**Tỷ lệ chuyển làn thành công**

- **Công thức**: `số lần chuyển làn thành công / tổng số lần thử chuyển làn`
- **Ý nghĩa**: Sau khi quyết định chuyển làn, hệ thống có thực sự hoàn thành được hay không. "Thành công" = xe đã ở trong làn mới, heading ổn định, không abort.
- **Ví dụ dễ hiểu**: Trong 10 lần định vượt xe, có 8 lần vượt thành công → success rate = 80%.
- **Giá trị mong muốn**: ↑ cao hơn tốt hơn (hard gate).

---

### [lane_change_completion_validity](file:///d:/Agent-AI/benchmark/metrics.py#276-278)
**Tỷ lệ chuyển làn hợp lệ**

- **Công thức**: `số lane change thành công VÀ vào đúng làn mục tiêu VÀ shift hợp lệ / tổng lane change thành công`
- **Ý nghĩa**: Không chỉ "chuyển làn xong" mà còn phải "chuyển đúng làn cần chuyển". Có thể xe đã sang làn khác nhưng không phải làn đúng.
- **Giá trị mong muốn**: ↑ cao hơn tốt hơn (hard gate).

---

### [lane_change_abort_rate](file:///d:/Agent-AI/benchmark/metrics.py#272-274)
**Tỷ lệ hủy chuyển làn giữa chừng**

- **Công thức**: `số lần hủy lane change / tổng số lần thử`
- **Ý nghĩa**: Hệ thống bắt đầu chuyển làn rồi dừng lại giữa chừng (vì phát hiện nguy hiểm, hoặc quyết định thay đổi). Abort không nhất thiết là lỗi (có thể là phản xạ an toàn đúng), nhưng tỷ lệ cao = không ổn định.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (soft guardrail).

---

### [stop_success_rate](file:///d:/Agent-AI/benchmark/metrics.py#201-209)
**Tỷ lệ dừng xe thành công**

- **Công thức**: `số lần dừng xe thành công / tổng số lần thử dừng`
- **Ý nghĩa**: Từ lệnh stop đến trạng thái xe dừng hoàn toàn có đúng theo yêu cầu không. "Thất bại" = xe không dừng được, hoặc dừng sai vị trí quá nhiều.
- **Giá trị mong muốn**: ↑ cao hơn tốt hơn (hard gate).

---

### [stop_overshoot_rate](file:///d:/Agent-AI/benchmark/metrics.py#211-219)
**Tỷ lệ dừng vượt điểm**

- **Công thức**: `số lần dừng xe bị qua khỏi điểm dừng / tổng số lần dừng`
- **Ý nghĩa**: Hệ thống phanh muộn hoặc phanh không đủ mạnh → xe vượt qua vạch dừng. Ở đèn đỏ hay trước người đi bộ → cực kỳ nguy hiểm.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (hard gate).

---

### [stop_overshoot_distance_m](file:///d:/Agent-AI/benchmark/metrics.py#287-292)
**Khoảng cách trung bình vượt điểm dừng (mét)**

- **Công thức**: `trung bình(max(ngưỡng overshoot - khoảng cách_thực, 0))` cho các lần dừng bị overshoot
- **Ý nghĩa**: Không chỉ biết có vượt hay không, mà còn biết vượt bao nhiêu. Vượt 0.1m khác hoàn toàn với vượt 2m.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (soft guardrail).

---

### [stop_hold_stability](file:///d:/Agent-AI/benchmark/metrics.py#294-296)
**Độ ổn định khi giữ trạng thái dừng**

- **Công thức**: `trung bình(min(số tick đã giữ dừng / số tick yêu cầu, 1.0))` qua các lần dừng
- **Ý nghĩa**: Sau khi dừng, xe có giữ yên không hay lại nhúc nhích, trôi. Một chiếc xe dừng xong rồi từ từ trôi về phía trước là lỗi nghiêm trọng.
- **Giá trị mong muốn**: ↑ cao hơn tốt hơn (hard gate).

---

### [behavior_execution_mismatch_rate](file:///d:/Agent-AI/benchmark/metrics.py#237-256)
**Tỷ lệ thực thi sai behavior**

- **Công thức**: `số frame thực thi behavior khác với behavior được yêu cầu / tổng frame`
- **Ý nghĩa**: Planner nói "chuyển làn trái" nhưng controller lại đang thực hiện "theo dõi làn thẳng" → có sự không đồng bộ giữa tầng quyết định và tầng thực thi.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (hard gate).

---

## NHÓM D — CONTROL QUALITY (Chất Lượng Điều Khiển)

---

### [lateral_oscillation_rate](file:///d:/Agent-AI/benchmark/metrics.py#306-308)
**Tỷ lệ dao động tay lái**

- **Công thức**: `số lần dấu tay lái đổi chiều / tổng số lần điều chỉnh tay lái`
- **Ý nghĩa**: Nếu tay lái liên tục đánh trái-phải-trái-phải → xe đi zic-zac, hành khách khó chịu, không an toàn. Điều khiển tốt phải mượt, ổn định.
- **Ví dụ dễ hiểu**: Tài xế mới thường "giữ tay lái" theo kiểu chỉnh liên tục nhỏ → xe không đi thẳng. Tài xế lành nghề giữ tay lái ổn định.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (soft guardrail).

---

### [trajectory_smoothness_proxy](file:///d:/Agent-AI/benchmark/metrics.py#310-312)
**Độ mượt mà quỹ đạo**

- **Công thức**: [mean(|Δsteer|) + 0.05 × mean(|acceleration_proxy|)](file:///d:/Agent-AI/benchmark/metrics.py#258-262)
- **Ý nghĩa**: Tổng hợp hai yếu tố — biến thiên tay lái (càng ít thay đổi càng mượt) và gia tốc (tăng/giảm tốc đột ngột sẽ bị phạt). Giá trị nhỏ = quỹ đạo trơn, thoải mái.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (soft guardrail).

---

### [control_jerk_proxy](file:///d:/Agent-AI/benchmark/metrics.py#314-316)
**Độ giật khi điều khiển (Jerk)**

- **Công thức**: [mean(|Δacceleration / Δt|)](file:///d:/Agent-AI/benchmark/metrics.py#258-262) — đạo hàm của gia tốc theo thời gian
- **Ý nghĩa**: Jerk là cảm giác "giật" khi xe tăng/giảm tốc đột ngột. Ví dụ: phanh gấp là jerk cao. Xe tự lái tốt phải có jerk thấp — tăng/giảm tốc từ từ.
- **Đơn vị**: m/s³
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (diagnostic only — chưa dùng để gating).

---

### [heading_settle_time_s](file:///d:/Agent-AI/benchmark/metrics.py#302-304)
**Thời gian ổn định hướng đầu xe (giây)**

- **Công thức**: Thời gian từ khi hoàn thành lane change đến khi hướng xe và vị trí so với tâm làn ổn định (3 frame liên tiếp đạt ngưỡng)
- **Ý nghĩa**: Sau khi chuyển làn xong, xe còn "lắc" một lúc trước khi đi ổn định. Thời gian lắc này càng ngắn → hệ thống điều khiển càng tốt.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (soft guardrail).

---

## NHÓM E — RUNTIME & LATENCY (Hiệu Năng Thời Gian Thực)

---

### [mean_total_loop_latency_ms](file:///d:/Agent-AI/benchmark/metrics.py#460-464)
**Latency trung bình toàn bộ vòng lặp (ms)**

- **Công thức**: `trung bình thời gian xử lý mỗi tick` từ perception đến control output
- **Ý nghĩa**: Xe đang chạy 60km/h → mỗi giây đi 16.7m. Nếu hệ thống xử lý mất 500ms → trong 0.5 giây xe đi 8.3m mà "mù". Latency thấp = phản xạ nhanh, an toàn hơn.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn.

---

### [over_budget_rate](file:///d:/Agent-AI/benchmark/metrics.py#331-333)
**Tỷ lệ vượt ngưỡng latency**

- **Công thức**: `số frame có latency > budget / tổng frame`
- **Ý nghĩa**: Hệ thống có SLA (service level agreement) về latency — ví dụ phải xử lý trong 100ms. Metric này đo tỷ lệ vi phạm. Nếu 30% frame vượt budget → hệ thống không đủ ổn định cho real-time.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn (soft guardrail).

---

### [stale_tick_rate](file:///d:/Agent-AI/benchmark/metrics.py#453-458)
**Tỷ lệ tick bị bỏ qua**

- **Công thức**: `skipped_updates / total_ticks`
- **Ý nghĩa**: Trong online mode, nếu hệ thống xử lý không kịp, một số tick bị skip — tức là có "khoảng trống thời gian" mà không có quyết định mới. Tỷ lệ cao = hệ thống bị quá tải.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn.

---

## NHÓM F — SAFETY & FALLBACK (An Toàn Dự Phòng)

---

### [fallback_activation_rate](file:///d:/Agent-AI/benchmark/metrics.py#507-519)
**Tỷ lệ kích hoạt fallback/emergency**

- **Công thức**: `số lần emergency override / tổng frame`
- **Ý nghĩa**: Hệ thống dự phòng (fallback) bật lên khi phát hiện nguy hiểm không xử lý được bằng flow thông thường. Tỷ lệ thấp = hoạt động bình thường. Tỷ lệ cao = hệ thống liên tục gặp tình huống không kiểm soát được.
- **Giá trị mong muốn**: ↓ thấp hơn tốt hơn.

---

### [fallback_reason_coverage](file:///d:/Agent-AI/benchmark/metrics.py#521-523)
**Tỷ lệ fallback có lý do rõ ràng**

- **Công thức**: `số lần fallback có ghi reason / tổng lần fallback activate`
- **Ý nghĩa**: Mỗi khi fallback bật, phải ghi rõ "tại sao" (hết thời gian, solver fail, nguy hiểm phát hiện). Nếu fallback xảy ra mà không có lý do → không thể debug, không thể cải thiện.
- **Giá trị mong muốn**: ↑ = 1.0 (hard gate).

---

### [unexpected_override_count](file:///d:/Agent-AI/benchmark/metrics.py#636-645)
**Số emergency override ngoài dự kiến**

- **Công thức**: Đếm số event `emergency_override` có classification = [unexpected](file:///d:/Agent-AI/benchmark/metrics.py#636-645)
- **Ý nghĩa**: Một số override được dự kiến trước trong test case (là hành vi đúng). Nhưng override "bất ngờ" = hệ thống làm điều không ai mong đợi → lỗi logic.
- **Giá trị mong muốn**: = 0.

---

### [arbitration_conflict_count](file:///d:/Agent-AI/benchmark/metrics.py#426-451)
**Số xung đột arbitration**

- **Công thức**: Đếm số arbitration event có classification = [unexpected](file:///d:/Agent-AI/benchmark/metrics.py#636-645) và action không phải là `continue`
- **Ý nghĩa**: Arbitration Engine (AE) phân xử khi nhiều tầng đưa ra quyết định xung đột. Conflict = AE phải can thiệp theo cách ngoài kế hoạch. Nhiều conflict = các tầng không đồng bộ tốt.
- **Giá trị mong muốn**: = 0 (soft guardrail).

---

## NHÓM G — LLM AGENT SHADOW (Quan Trọng Nhất)

---

### `agent_contract_validity_rate`
**Tỷ lệ đề xuất agent hợp lệ về schema**

- **Công thức**: `số đề xuất của agent đúng format contract / tổng đề xuất`
- **Ý nghĩa**: Agent AI (LLM) phải trả về đề xuất theo đúng schema định nghĩa (JSON schema). Nếu output sai format → hệ thống không sử dụng được dù nội dung có đúng. Đây là điều kiện cơ bản nhất: "agent có giao tiếp được không?"
- **Giá trị thực của bạn**: **1.0 (100%)** — tất cả đề xuất đều hợp lệ.
- **Giá trị mong muốn**: = 1.0 (hard gate).

---

### `agent_forbidden_intent_count`
**Số đề xuất chứa intent bị cấm**

- **Công thức**: Đếm số đề xuất của agent có chứa intent nằm trong danh sách cấm.
- **Ý nghĩa**: Agent không được phép đề xuất một số hành động nguy hiểm (ví dụ: dừng xe đột ngột ở tốc độ cao, chuyển làn khi không có không gian). Đây là safety boundary cứng.
- **Giá trị thực của bạn**: **0** — tuyệt đối an toàn.
- **Giá trị mong muốn**: = 0 (hard gate).

---

### `agent_timeout_rate`
**Tỷ lệ agent không trả lời kịp thời**

- **Công thức**: `số frame agent không trả lời trong timeout / tổng frame`
- **Ý nghĩa**: LLM API có thể chậm. Nếu agent không trả lời trong 1.5s (stub) hoặc 10s (LLM API), hệ thống phải tiếp tục với baseline — coi như agent "để trống". Tỷ lệ timeout cao = agent không đáng tin cậy cho real-time.
- **Giá trị thực của bạn**: **0.0%** — agent luôn trả lời kịp thời.
- **Giá trị mong muốn**: = 0.0 (càng thấp càng tốt).

---

### `agent_baseline_disagreement_rate`
**Tỷ lệ agent đề xuất khác với baseline**

- **Công thức**: `số frame agent đề xuất behavior khác với baseline tactical / tổng frame`
- **Ý nghĩa**: Baseline là hệ thống tự lái cũ (đã hoạt động tốt). LLM agent chạy song song trong shadow mode và đề xuất behavior. Khi agent "không đồng ý" với baseline → có thể agent đúng (phát hiện điều baseline bỏ sót) hoặc sai.
- **Giá trị thực của bạn**:
  - `ml_right_positive_core`: **3.3%** — agent hầu như đồng ý với baseline
  - `stop_follow_ambiguity_core`: **26.7%** — agent thường xuyên đề xuất khác
- **Cách diễn giải**: Sự khác biệt ở `stop_follow_ambiguity_core` cho thấy agent nhận ra độ mơ hồ của scenario và tích cực đề xuất giải pháp khác — đây là đặc tính **mong muốn** của agent thông minh.

---

### `agent_useful_disagreement_rate`
**Tỷ lệ đề xuất khác có ích**

- **Công thức**: `số lần agent không đồng ý VÀ đề xuất của agent được đánh giá là hữu ích / tổng lần không đồng ý`
- **Ý nghĩa**: Quan trọng hơn "đề xuất khác bao nhiêu" là "khi khác thì có đúng không". Nếu agent disagreement 100% mà useful rate cũng 100% → agent thực sự thêm giá trị. Nếu useful rate = 0% → agent chỉ "phá" baseline.
- **Giá trị thực của bạn**: **100%** — mọi lần không đồng ý đều có ích.
- **Đây là kết quả cần highlight nhất trong bài báo.**

---

### `agent_route_alignment_rate`
**Tỷ lệ đề xuất agent phù hợp với lộ trình**

- **Công thức**: `số frame đề xuất của agent align với route hiện tại / tổng frame`
- **Ý nghĩa**: Agent phải biết xe đang đi theo lộ trình nào và đề xuất behavior phải phù hợp với lộ trình đó. Không được đề xuất "rẽ trái" khi lộ trình yêu cầu đi thẳng.
- **Giá trị thực của bạn**: **1.0 (100%)** — agent luôn đề xuất đúng với route.
- **Giá trị mong muốn**: = 1.0 (hard gate).

---

## NHÓM H — ASSIST MODE (Stage 8)

---

### `ae_approval_rate`
**Tỷ lệ đề xuất được Arbitration Engine chấp nhận**

- **Công thức**: `số BRP được AE approve / tổng BRP`
- **Ý nghĩa**: BRP (Behavior Request Proposal) là đề xuất từ agent. AE kiểm tra tính hợp lệ và an toàn. Approval rate cao = agent đề xuất đúng, AE không cần veto nhiều.
- **Giá trị thực**: **1.0 (100%)** — AE chấp nhận mọi đề xuất.

---

### `ae_veto_rate`
**Tỷ lệ đề xuất bị từ chối**

- **Công thức**: `số BRP bị veto / tổng BRP`
- **Ý nghĩa**: AE veto = đề xuất vi phạm safety contract hoặc không khả thi. Tỷ lệ thấp = agent đề xuất an toàn.
- **Giá trị thực**: **0.0%** — không có veto nào.

---

### `mpc_guardian_override_rate`
**Tỷ lệ MPC giữ quyền kiểm soát**

- **Công thức**: `số frame MPC override đề xuất agent / tổng frame`
- **Ý nghĩa**: Trong thiết kế safety-first, MPC (Model Predictive Controller) luôn là "người gác cổng" cuối cùng. Metric = 1.0 có nghĩa là MPC luôn làm chủ — agent chỉ "tư vấn", không ra lệnh trực tiếp. Đây là đặc tính **đúng về mặt thiết kế**.
- **Giá trị thực**: **1.0** — MPC giữ quyền tuyệt đối trong giai đoạn shadow.

---

### `brp_mean_latency_ms`
**Latency trung bình của đề xuất agent (ms)**

- **Công thức**: Thời gian từ khi gửi context đến LLM đến khi nhận được BRP
- **Ý nghĩa**: Thời gian LLM API xử lý. Ngưỡng là 1500ms. Giá trị thực **2817ms** — vượt ngưỡng.
- **Giá trị thực của bạn**: **2817ms** ⚠️ — đây là bottleneck cần nêu trong bài báo.
- **Cách viết trong bài**: *"The current LLM API latency (2817ms) exceeds the real-time budget (1500ms), suggesting that on-device distillation or response caching are necessary for production deployment."*

---

### `agent_useful_assist_rate`
**Tỷ lệ đề xuất assist có ích thực sự**

- **Công thức**: `số BRP được approve VÀ đủ điều kiện để dùng / tổng BRP`
- **Ý nghĩa**: Tính toán mức độ agent thực sự "hữu ích" trong việc assist — không chỉ là trả lời đúng format mà còn phải đề xuất theo cách hệ thống có thể áp dụng thực tế.
- **Giá trị thực**: **0.0** ⚠️ — đây là limitation quan trọng ở Stage 8 hiện tại.
- **Cách viết trong bài**: *"The agent_useful_assist_rate of 0.0 in Stage 8 sandbox reflects the early-stage nature of the assist interface, where the agent's proposals are structurally valid but not yet calibrated for live assist integration."*

---

## 📌 Tóm Tắt Nhanh Cách Đọc Kết Quả

| Ký hiệu | Nghĩa |
|---|---|
| **Hard Gate** | Fail ở metric này = toàn bộ test case fail |
| **Soft Guardrail** | Warning nhưng không fail toàn bộ |
| **Diagnostic Only** | Chỉ để theo dõi, chưa dùng để gating |
| ↑ higher better | Giá trị càng cao càng tốt |
| ↓ lower better | Giá trị càng thấp càng tốt |
| = 0 / = 1.0 | Phải đạt chính xác giá trị đó |
