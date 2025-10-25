# src/make_report.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
import os

def make_report(result_dir="results"):
    pdf_path = os.path.join(result_dir, "결과_요약.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle('title', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, alignment=1)
    style_sub = ParagraphStyle('sub', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, spaceAfter=10)
    style_body = ParagraphStyle('body', parent=styles['Normal'], fontSize=12, leading=16)

    story = []
    story.append(Paragraph("Q-러닝을 이용한 최적 경로 탐색 결과 보고서", style_title))
    story.append(Spacer(1, 0.5*cm))

    # 개요
    story.append(Paragraph("📘 1. 실험 개요", style_sub))
    story.append(Paragraph("본 프로그램은 강화학습의 한 종류인 Q-러닝(Q-Learning)을 이용하여 미로 환경에서 최적의 경로를 학습하는 실험을 수행하였습니다.", style_body))
    story.append(Spacer(1, 0.5*cm))

    # 그래프들 추가
    for img_name, desc in [
        ("reward_curve.png", "학습이 진행되면서 누적 보상이 점차 증가하는 것을 확인할 수 있습니다. 이는 에이전트가 점점 더 효율적으로 경로를 탐색함을 의미합니다."),
        ("success_rate.png", "성공률 그래프는 시간이 지날수록 목표 지점에 도달하는 빈도가 높아지는 것을 보여줍니다."),
        ("policy_visual.png", "학습된 정책(Policy)은 각 칸에서 어떤 방향으로 이동하는 것이 유리한지를 화살표로 표현합니다."),
        ("path_visual.png", "최적 경로 시각화는 에이전트가 학습을 마친 후 실제로 목표까지 어떤 경로로 이동하는지를 보여줍니다.")
    ]:
        img_path = os.path.join(result_dir, img_name)
        if os.path.exists(img_path):
            story.append(Paragraph(f"📊 {img_name.split('.')[0]}", style_sub))
            story.append(Image(img_path, width=14*cm, height=8*cm))
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(desc, style_body))
            story.append(Spacer(1, 0.5*cm))

    # 요약 텍스트
    summary_path = os.path.join(result_dir, "summary.txt")
    if os.path.exists(summary_path):
        story.append(Paragraph("📄 2. 학습 요약", style_sub))
        with open(summary_path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if "," in line:
                key, val = line.strip().split(",", 1)
                story.append(Paragraph(f"{key.strip()}: {val.strip()}", style_body))
        story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("✅ 결론적으로, Q-러닝을 통해 에이전트는 시행착오를 반복하며 보상을 최대화하는 경로를 학습할 수 있음을 확인하였습니다. "
                           "이는 인간의 학습 과정과 유사한 방식으로 문제 해결 전략을 발전시키는 인공지능의 핵심 개념을 시각적으로 보여줍니다.", style_body))

    doc.build(story)
    print(f"[완료] '{pdf_path}' 생성됨")

if __name__ == "__main__":
    make_report()
