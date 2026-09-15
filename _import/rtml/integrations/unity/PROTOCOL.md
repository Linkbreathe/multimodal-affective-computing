# Unity ↔ Python UDP JSON 协议

- Unity → Python：`127.0.0.1:5055`
- Python → Unity：`127.0.0.1:5056`
- UTF-8 JSON，一条 UDP datagram 对应一条消息。
- `schema_version`、`unix_time_ms` 必填；Python 输出还必须含 `cycle_index`、`window_start_ms`、`window_end_ms`。
- Unity 每次 Condition 变化都必须发送新 `condition`。Python 以该消息的 `unix_time_ms` 重置 10 秒窗口原点。
- Python 每个周期恰好发送一条 `StatePrediction` 和一条 `ConditionRecommendation`。
- `shadow=true` 表示只显示/记录建议，Unity 不执行 Condition 切换。

Unity 输入示例：

```json
{"schema_version":"1.0.0","unix_time_ms":1780565300000,"participant_id":"P003","condition":"C5","head_pose":{"head_position_x":0.0,"head_position_y":1.7,"head_position_z":0.0,"head_angular_velocity_deg_s":0.2},"eye":{"gaze_direction_x":0.0,"gaze_direction_y":0.0,"gaze_direction_z":1.0,"gaze_on_painting":true},"frame_relative_path":"video_frames/frame_001234.jpg"}
```

