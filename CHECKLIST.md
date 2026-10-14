# Checklist nộp bài Lab Day 10

- [x] `REPORT.md` có họ tên Trần Cao Quốc Định, MSSV 2A202602939, `dataset_version`, bảng E0-E4 và đủ 10 câu hỏi.
- [x] `REPORT.md` không còn chuỗi placeholder `<điền>`.
- [x] `report/figures/` có 6 ảnh thật, gồm box ground truth và prediction cho cả ba lớp.
- [x] `report/evidence/` có E0-E3 và hai file E4 rank 4/rank 64 do `scripts/score.py` sinh ra.
- [x] `configs/`, `scripts/` và `tests/` khớp commit gốc của đề.
- [x] Không track `data/`, `work/`, `.env` hoặc trọng số model.
- [x] Git có ít nhất 5 commit; repo GitHub đang Public trên branch `main`.
- [x] Trên VM có dataset thật: `python scripts/check_submission.py` đạt 0 FAIL, 0 WARN.

Repo GitHub: https://github.com/kentranpr4-crypto/K4-Track-4-Day10-VisionLanguageModel

Trên máy local không có `data/splits/meta.json`, checker sẽ báo 1 WARN khi đối chiếu dataset; đây là dữ liệu lớn không được commit. Kiểm tra nộp bài đầy đủ đã chạy trên VM chứa dataset và đạt 0 WARN.

Việc dán URL GitHub vào LMS là bước người nộp bài thực hiện bằng tài khoản cá nhân.
