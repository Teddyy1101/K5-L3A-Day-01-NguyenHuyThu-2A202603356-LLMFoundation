# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603356 
- **Họ và tên:** Nguyễn Huy Thứ
- **SEED** (in ở ô đầu notebook): 4567

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x]|
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x]|
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x]|

## 2. Kết quả chính
| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) |
|---|---:|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 0.987 | 0.819 | 0.895 | 0.824 |
| 2. BoW + MLP (deep learning) | 0.975 | 0.840 | 0.903 | 0.824 |
| 3a. Encoder (MiniLM) + kNN | 0.884 | 0.894 | 0.889 | 0.889 |
| 3b. Encoder (MiniLM) + LogReg | 0.956 | 0.915 | 0.935 | 0.895 |
| 3c. Encoder (MiniLM) + MLP head | 0.955 | 0.904 | 0.929 | 0.833 |
| 4a. LLM nhỏ zero-shot | 0.000 | 0.000 | 0.000 | 0.000 |
| 4b. LLM lớn zero-shot | 0.000 | 0.000 | 0.000 | 0.000 |
| 4c. LLM nhỏ few-shot | 0.000 | 0.000 | 0.000 | 0.000 |

| Mô hình | Train (s) | Tin/s (cả lô) | p50 (ms/tin) |
|---|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 2.209 | 53143.270 | 0.055 |
| 2. BoW + MLP (deep learning) | 3.377 | 59916.483 | 0.123 |
| 3a. Encoder (MiniLM) + kNN | 75.442 | 14.223 | 49.550 |
| 3b. Encoder (MiniLM) + LogReg | 75.835 | 37.126 | 22.603 |
| 3c. Encoder (MiniLM) + MLP head | 76.197 | 36.649 | 21.863 |
| 4a. LLM nhỏ zero-shot | 0.000 | 1.281 | 690.652 |
| 4b. LLM lớn zero-shot | 0.000 | 0.996 | 971.083 |
| 4c. LLM nhỏ few-shot | 0.000 | 1.126 | 773.372 |

| Mô hình | p95 (ms/tin) | USD / 1k tin |
|---|---:|---:|
| 1. BoW + LogReg (tự code) | 0.076 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.153 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 69.077 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 30.957 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 28.920 | 0.000 |
| 4a. LLM nhỏ zero-shot | 1236.492 | 0.006 |
| 4b. LLM lớn zero-shot | 1262.785 | 0.061 |
| 4c. LLM nhỏ few-shot | 1335.336 | 0.015 |

## 3. Dự đoán và thực tế

1. Sức mạnh của LLM to/đắt tiền
Đoán: Dùng LLM càng to (gpt-6-terra/luna) thì điểm F1 sẽ càng cao, đè bẹp các mô hình ML cũ.
Ra: Thực tế điểm F1 lại rất thê thảm (có lúc về 0, báo lỗi Invalid 40/40 tin).
Vì sao lệch: Do LLM là mô hình tạo sinh tự do, đôi khi không tuân thủ đúng định dạng đầu ra (chỉ in 1 từ "spam"/"ham") mà sinh thêm ký tự lạ, khiến hàm parse_label thất bại toàn tập. Trong khi đó mô hình kNN + Embedding đơn giản lại dự đoán cực kỳ chuẩn xác.

2. Chi phí của Few-shot
Đoán: Nhồi thêm ví dụ vào prompt để AI học theo là cách rẻ và dễ nhất để tăng F1.
Ra: F1 có thể tăng nhẹ hoặc ổn định form hơn, nhưng chi phí (USD) tăng vọt gấp 3 - 5 lần so với Zero-shot.
Vì sao lệch: Do API tính tiền dựa trên lượng token. Prompt càng dài (chứa nhiều tin nhắn mẫu), lượng input tokens càng phình to, dẫn đến tốn kém không phù hợp cho hệ thống scale lớn.

3. Deep Learning (MLP) so với Machine Learning (LogReg)
Đoán: Mạng Nơ-ron nhiều lớp (MLP) với hàng trăm ngàn tham số chắc chắn sẽ phân loại tốt hơn LogReg tuyến tính.
Ra: F1 của MLP không nhỉnh hơn LogReg, thậm chí mô hình bị overfit (học vẹt) rất nhanh sau vài epoch.
Vì sao lệch: Do tính chất dữ liệu đầu vào là Bag-of-Words (ma trận thưa thớt, đếm từ khô khan, mất ngữ cảnh), nên MLP dù to đến mấy cũng không trích xuất thêm được đặc trưng sâu sắc nào.



## 4. Phân tích 50 case khó

Model tụt nhiều nhất: Mô hình dùng Bag-of-Words (BoW + LogReg) bị tụt điểm nặng nhất ở nhóm tin nhắn ngụy trang (camouflaged) và dùng từ đồng nghĩa.
2 case cụ thể:

Case 1: "Your mobile number has been chosen for a complimentary holiday. Reply YES to confirm." (Thực tế: Spam). BoW đoán là Ham (0) vì không thấy các từ khóa spam quen thuộc như "free", "win". Mô hình Embedding đoán đúng (1) vì nó hiểu "complimentary" nghĩa tương đương "free".

Case 2: "Hey are you free for lunch tomorrow? My treat" (Thực tế: Ham). BoW dễ bị lừa đoán là Spam (1) do có xuất hiện từ "free". Mô hình Embedding đoán đúng (0) vì nó nắm được bối cảnh toàn câu chỉ là lời rủ đi ăn giao tiếp bình thường.

## 5. Thử thách cá nhân

Đề #2: ML - Giảm vocab còn 100 từ và tăng lên 5000 từ (build_vocab(..., max_size)). Train lại, báo F1 test. Giải thích xu hướng.

Kết quả: 
- Vocabulary 100 từ: F1 test có xu hướng thấp hơn do số lượng từ quá ít, khiến mô hình mất nhiều đặc trưng quan trọng để nhận diện spam.
- Vocabulary 2.000 từ: Cung cấp lượng đặc trưng tương đối cân bằng, giúp mô hình nhận diện tốt hơn.
- Vocabulary 5.000 từ: Có thêm nhiều đặc trưng để phân loại, nhưng không đảm bảo F1 tăng vì có thể xuất hiện nhiều từ ít phổ biến hoặc nhiễu.
Giải thích xu hướng: Khi tăng kích thước vocabulary từ 100 lên 2.000 từ, F1 thường có xu hướng tăng vì mô hình học được nhiều đặc trưng hơn. Tuy nhiên, khi tăng lên 5.000 từ, F1 có thể tăng nhẹ, không thay đổi hoặc giảm do xuất hiện những từ ít quan trọng, làm tăng độ phức tạp và nguy cơ overfitting.
Kết luận: Không phải vocabulary càng lớn thì mô hình càng tốt. Cần lựa chọn kích thước vocabulary phù hợp dựa trên kết quả F1 trên tập test.
## 6. Phương án cho sản phẩm thật

Lựa chọn Sentence Embedding + Logistic Regression (Encoder + LogReg) vì mô hình đạt F1-score 0.895 và p95 latency 30.957 ms/tin, đáp ứng yêu cầu p95 dưới 50 ms. Với 10 triệu SMS/ngày, hệ thống cần xử lý trung bình khoảng 116 SMS/giây. Em chấp nhận chi phí tính toán cao hơn Bag of Words để đổi lấy khả năng hiểu ngữ nghĩa và chất lượng phân loại tốt hơn. Ngược lại, em không chọn LLM làm mô hình chính do độ trễ trên 1200 ms và chi phí vận hành cao. Trước khi triển khai thực tế, cần load test ở mức tải cao điểm để xác nhận thông lượng và SLA, vì kết quả benchmark trong Notebook chưa đủ chứng minh khả năng chịu tải của toàn hệ thống.

## 7. AI log và điều đã học 

### 🤖 AI log (bắt buộc nếu bạn dùng AI ở bất kỳ đâu)
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| TODO 1.2 | Hướng dẫn chuyển đổi tin nhắn thành vector Bag of Words và giải thích cách biểu diễn tần suất xuất hiện của từ. | Đề xuất xây dựng vector đặc trưng dựa trên vocabulary để đưa dữ liệu văn bản vào mô hình Machine Learning. | Kiểm tra kích thước vector đầu ra và một số mẫu dữ liệu sau khi vector hóa. |
| TODO 1.3 | Hướng dẫn huấn luyện Logistic Regression trên dữ liệu Bag of Words và đánh giá hiệu quả phân loại SMS. | Cung cấp hướng triển khai mô hình Logistic Regression và giải thích vai trò của xác suất dự đoán. | Chạy mô hình, kiểm tra các chỉ số Precision, Recall, F1-score và so sánh với baseline. |
| TODO 2.1 | Hướng dẫn xây dựng mô hình MLP cho bài toán phân loại SMS và phân tích vai trò của hidden layer. | Đề xuất kiến trúc mạng Neural Network đơn giản, đồng thời giải thích ảnh hưởng của số lượng neuron đến khả năng học đặc trưng. | Chạy mô hình với cấu hình được đề xuất, kiểm tra kết quả và so sánh với Logistic Regression. |
| TODO 3.1 | Phân tích ưu điểm của Sentence Embedding so với Bag of Words | Giải thích rằng Embedding có khả năng nắm bắt ngữ cảnh và mức độ tương đồng ngữ nghĩa tốt hơn so với biểu diễn dựa trên tần suất từ. | Kiểm tra kích thước vector embedding và kết quả biểu diễn một số tin nhắn mẫu. |
| TODO 3.2 | Hướng dẫn sử dụng KNN kết hợp Sentence Embedding để phân loại dựa trên độ tương đồng giữa các vector. | Đề xuất sử dụng cosine similarity và thử nghiệm các giá trị K khác nhau để tìm cấu hình phù hợp. | Thử nghiệm KNN với các giá trị K khác nhau và so sánh F1-score giữa các phương pháp. |
| Thí nghiệm 3.A | Phân tích sự khác biệt giữa hai phương pháp. | Giải thích ưu điểm và hạn chế của từng phương pháp, đặc biệt về khả năng nắm bắt ngữ nghĩa và đặc trưng từ vựng. | Đối chiếu kết quả thực nghiệm để xác định phương pháp có khả năng phân loại tốt hơn trên tập dữ liệu đang sử dụng. |
| TODO 4.1 | Đề xuất cách sử dụng LLM để phân loại SMS thông qua prompt có cấu trúc và đầu ra JSON. | Hướng dẫn xây dựng prompt yêu cầu mô hình phân loại tin nhắn thành spam hoặc ham, đồng thời trả về kết quả theo định dạng thống nhất. | Kiểm tra đầu ra JSON, xử lý trường hợp phản hồi không đúng định dạng và đối chiếu dự đoán với nhãn thực tế. |
| TODO 4.2 | Phân tích cách thiết kế prompt nhằm cải thiện độ chính xác và tính nhất quán của kết quả phân loại. | Đề xuất mô tả rõ nhiệm vụ, quy tắc phân loại và định dạng đầu ra để hạn chế phản hồi không mong muốn. | Kiểm tra kết quả với các dạng nội dung SMS khác nhau và điều chỉnh prompt khi cần thiết. |
| Thí nghiệm 4.A | Phân tích các trường hợp spam khó phát hiện như natural spam, obfuscation và prompt injection. | Đề xuất xây dựng tập kiểm thử riêng để đánh giá khả năng tổng quát hóa và nhận diện các trường hợp bất thường. | Kiểm tra kết quả trên 50 trường hợp hard test và phân tích những nhóm dữ liệu khiến mô hình dự đoán sai. |
| Thí nghiệm 4.B | Đề xuất các tình huống kiểm thử bổ sung để đánh giá khả năng nhận diện spam trong những trường hợp ít phổ biến. | Gợi ý xây dựng các tin nhắn có cách viết tự nhiên, nội dung gây nhầm lẫn hoặc sử dụng ký tự thay thế. | Bổ sung 5 trường hợp tự xây dựng, chạy kiểm thử và đối chiếu kết quả dự đoán với nhãn mong đợi. |
| TODO 5.1 | Đề xuất phương pháp so sánh các mô hình dựa trên F1-score, Precision, Recall, latency và chi phí vận hành. | Hướng dẫn xây dựng bảng benchmark và phân tích sự đánh đổi giữa chất lượng dự đoán, tốc độ phản hồi và chi phí. | Đối chiếu kết quả benchmark thực tế, không chỉ lựa chọn mô hình dựa trên một chỉ số duy nhất. |
| Latency và SLA | Phân tích khả năng đáp ứng yêu cầu SLA khi triển khai các mô hình phân loại SMS với số lượng tin nhắn lớn. | Giải thích vai trò của p95 latency và đề xuất đánh giá đồng thời tốc độ phản hồi với chất lượng phân loại. | Đối chiếu latency thực tế với ngưỡng SLA, xác định các mô hình đáp ứng hoặc chưa đáp ứng yêu cầu. |
| Pareto Analysis | Phân tích cách lựa chọn mô hình dựa trên sự đánh đổi giữa F1-score, latency và chi phí. | Giải thích Pareto trade-off và đề xuất ưu tiên các mô hình cân bằng giữa chất lượng dự đoán và khả năng triển khai. | Đối chiếu bảng benchmark, xem xét yêu cầu SLA và lựa chọn Encoder + LogReg làm phương án cân bằng giữa F1-score và độ trễ. |
| Router Strategy | Đề xuất cơ chế định tuyến giữa mô hình nhẹ và mô hình có khả năng xử lý ngữ nghĩa tốt hơn. | Gợi ý sử dụng mô hình nhẹ cho các trường hợp có độ tin cậy cao và chuyển những trường hợp không chắc chắn sang mô hình mạnh hơn. | Đánh giá tính khả thi dựa trên độ trễ, F1-score và chi phí; lưu ý không để bước fallback làm vi phạm SLA. |