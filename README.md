# Q-Learning Maze (학술제 예제)

이 저장소는 Q-learning으로 미로(GridWorld)에서 최적 경로를 학습하는 예제입니다.
학습 스크립트는 `src/train.py`이며, 결과(그래프, q_table.csv, 정책 이미지)는 `results/` 폴더에 저장됩니다.

## 실행 (로컬)
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
python src/visualize.py
