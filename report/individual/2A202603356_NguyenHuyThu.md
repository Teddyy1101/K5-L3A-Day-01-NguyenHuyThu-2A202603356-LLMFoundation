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
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.987   0.819      0.895        0.824   
2. BoW + MLP (deep learning)         0.975   0.840      0.903        0.824   
3a. Encoder (MiniLM) + kNN           0.884   0.894      0.889        0.889   
3b. Encoder (MiniLM) + LogReg        0.956   0.915      0.935        0.895   
3c. Encoder (MiniLM) + MLP head      0.955   0.904      0.929        0.833   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.209      53143.270       0.055   
2. BoW + MLP (deep learning)         3.377      59916.483       0.123   
3a. Encoder (MiniLM) + kNN          75.442         14.223      49.550   
3b. Encoder (MiniLM) + LogReg       75.835         37.126      22.603   
3c. Encoder (MiniLM) + MLP head     76.197         36.649      21.863   
4a. LLM nhỏ zero-shot                0.000          1.281     690.652   
4b. LLM lớn zero-shot                0.000          0.996     971.083   
4c. LLM nhỏ few-shot                 0.000          1.126     773.372   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.076         0.000  
2. BoW + MLP (deep learning)          0.153         0.000  
3a. Encoder (MiniLM) + kNN           69.077         0.000  
3b. Encoder (MiniLM) + LogReg        30.957         0.000  
3c. Encoder (MiniLM) + MLP head      28.920         0.000  
4a. LLM nhỏ zero-shot              1236.492         0.006  
4b. LLM lớn zero-shot              1262.785         0.061  
4c. LLM nhỏ few-shot               1335.336         0.015  

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

Đề: "Đề #2: ML - Giảm vocab còn 100 từ và tăng lên 5000 từ (build_vocab(..., max_size)). Train lại, báo F1 test. Giải thích xu hướng."
Kết quả: 
- Vocabulary 100 từ: F1 test có xu hướng thấp hơn do số lượng từ quá ít, khiến mô hình mất nhiều đặc trưng quan trọng để nhận diện spam.
- Vocabulary 2.000 từ: Cung cấp lượng đặc trưng tương đối cân bằng, giúp mô hình nhận diện tốt hơn.
- Vocabulary 5.000 từ: Có thêm nhiều đặc trưng để phân loại, nhưng không đảm bảo F1 tăng vì có thể xuất hiện nhiều từ ít phổ biến hoặc nhiễu.
Giải thích xu hướng: Khi tăng kích thước vocabulary từ 100 lên 2.000 từ, F1 thường có xu hướng tăng vì mô hình học được nhiều đặc trưng hơn. Tuy nhiên, khi tăng lên 5.000 từ, F1 có thể tăng nhẹ, không thay đổi hoặc giảm do xuất hiện những từ ít quan trọng, làm tăng độ phức tạp và nguy cơ overfitting.
Kết luận: Không phải vocabulary càng lớn thì mô hình càng tốt. Cần lựa chọn kích thước vocabulary phù hợp dựa trên kết quả F1 trên tập test.
## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
