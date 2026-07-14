# BÁO CÁO TỔNG THỂ HỆ THỐNG & KĨ THUẬT TỐI ƯU HÓA LLM INFERENCE (Qwen3.5-2B)

## 1. Tổng Quan Bài Toán & Hạ Tầng Evaluation

* **Mô hình**: `Qwen/Qwen3.5-2B` (Dense Transformer, gốc BF16).
* **Hạ tầng BTC cấp**: 1 instance **MiG H200** (18GB VRAM, 3 Core CPU, 8GB RAM).
* **Tải thử nghiệm (Trace)**: 120 requests trong `trace-round1.jsonl` (~4.71 req/s, tổng thời gian 25.48s).
* **Mục tiêu điểm số**: 
  $$\text{Score} = 100 \times \text{ERS} \times f(\Delta)$$
  * **ERS (Latency Score)**: Tính dựa trên **TTFT** (Floor: 100ms, Ceiling: 1500ms) và **TPOT** (Floor: 20ms, Ceiling: 45ms).
  * **Accuracy Gate $f(\Delta)$**: Đánh giá độc lập trên 100 câu GPQA Diamond. Giữ sụt giảm độ chính xác $\Delta \le 10\%$ để đạt $f(\Delta) = 1.0$.

---

## 2. Kiến Trúc & Cấu Trúc Dự Án

Thư mục dự án bao gồm các thành phần chính phục vụ thử nghiệm và nộp bài:

```text
LLM_OPT/
├── analyze_trace.py       # Script phân tích chuyên sâu dữ liệu trace (Prompt, Context, System Prompt)
├── benchmark_ers.py       # Benchmark Simulator giả lập chuẩn 100% công thức ERS của BTC (SSE Streaming)
├── run_local_test.py      # Script chạy nhanh end-to-end kiểm tra healthcheck và đo ERS
├── docker-compose.yml     # Cấu hình sản phẩm chính dùng nộp bài (Chứa toàn bộ flags vLLM)
├── Dockerfile             # Đóng gói image chứa mô hình phục vụ nộp bài
└── input/
    └── trace-round1.jsonl # File trace 120 requests từ BTC
```

---

## 3. Kết Quả Phân Tích Trace (Trace Profiling Insights)

Thông qua script `analyze_trace.py`, chúng ta đã phát hiện các đặc điểm tải quan trọng:

1. **System Prompt lặp lại $100\%$**:
   * Cả 120 requests đều sử dụng duy nhất **1 System Prompt** độ dài 38,956 ký tự (**~9,739 tokens**).
   * $\rightarrow$ Đây là điểm chốt để tối ưu TTFT tiệm cận mốc 100ms nhờ **Prefix Caching**.
2. **Kích thước Context thực tế**:
   * Prompt nhỏ nhất: ~19,680 tokens.
   * Prompt lớn nhất: **~41,824 tokens**.
   * Max completion tokens requested: **200 tokens**.
   * $\rightarrow$ Context thực tế tối đa chỉ khoảng **42,024 tokens** (không cần thiết để `max-model-len=262144` gây lãng phí VRAM).

---

## 4. Bảng Chi Tiết Các Thay Đổi & Tác Dụng Kỹ Thuật

Dưới đây là tổng hợp toàn bộ cấu hình tối ưu đã được áp dụng trong file [`docker-compose.yml`](file:///c:/Users/admin/OneDrive/Desktop/LLM_OPT/docker-compose.yml):

| Tham số / Flag | Gốc (Baseline) | Đã Tối Ưu | Tác Dụng & Lý Do Kỹ Thuật |
| :--- | :--- | :--- | :--- |
| **`--max-model-len`** | `262144` | **`49152`** | Giải phóng hàng GB VRAM lãng phí cho context 256k không dùng tới. Giúp vLLM phân bổ được nhiều Block KV Cache hơn trên GPU 18GB. |
| **`--enable-prefix-caching`** | Disabled | **Enabled** | Tận dụng $100\%$ System Prompt chung (9.7k tokens). Từ request 2 trở đi, bước Prefill bỏ qua 9.7k tokens này $\rightarrow$ **Giảm TTFT cực lớn**. |
| **`--enable-chunked-prefill`** | Disabled | **Enabled** | Chia nhỏ việc prefill các prompt dài (avg 30k tokens). Tránh việc prefill chiếm trọn GPU gây khựng (stall) quá trình sinh token cho request khác $\rightarrow$ **Giữ TPOT ổn định và cực thấp**. |
| **`--kv-cache-dtype`** | `auto` (BF16) | **`fp8`** | Tiết kiệm **$50\%$ dung lượng VRAM** cho KV Cache mà không đụng vào weights mô hình. Giúp tăng dung lượng chứa concurrent requests lên gấp đôi, tránh bị out-of-memory. |
| **`--attention-backend`** | `auto` | **`FLASHINFER`** | Sử dụng kernel FlashInfer chuyên dụng cho PagedAttention trên kiến trúc NVIDIA Hopper (H200), tối ưu cả latency prefill lẫn decode. |
| **`--max-num-batched-tokens`** | `512` | **`4096`** | Tăng kích thước token được xử lý đồng thời trong 1 iteration. Phù hợp cho tải có prompt dài giúp tăng tốc độ prefill tổng thể. |
| **`--gpu-memory-utilization`**| `0.90` | **`0.92`** | Tăng mức trích xuất VRAM lên 92% (~16.5GB / 18GB VRAM) để mở rộng dung lượng cho KV cache block pool. |
| **`--disable-log-requests`** | Off | **Enabled** | Tắt ghi log từng request trên stdout/stderr. Giảm bớt CPU Overhead khi hệ thống chấm bài chỉ có **3 vCPU**. |
| **`--disable-log-stats`** | Off | **Enabled** | Tắt ghi log thống kê định kỳ để tiết kiệm chu kỳ xử lý của CPU. |

---

## 5. Định Hướng Tối Ưu Nâng Cao Tiếp Theo (Roadmap)

1. **Thử nghiệm Online Quantization (Mô hình FP8)**:
   * Chạy mô hình Qwen3.5-2B với `--quantization=fp8` để giảm lượng VRAM cho Model Weights từ 4GB xuống 2GB, dành thêm VRAM cho KV cache và đẩy throughput lên cao hơn.
2. **Fine-tune `--max-num-seqs`**:
   * Kiểm thử các mốc `32`, `64`, `128` trên benchmark local để tìm điểm cân bằng tốt nhất giữa TTFT và TPOT.
3. **Đánh giá Accuracy trên GPQA**:
   * Kiểm tra xem việc áp dụng `--kv-cache-dtype=fp8` có ảnh hưởng đến điểm số GPQA hay không (thông thường KV cache FP8 giữ được >99.5% accuracy so với gốc).
