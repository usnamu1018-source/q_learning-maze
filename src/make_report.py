# src/make_report.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.units import cm
import os

def make_report(result_dir="results"):
    pdf_path = os.path.join(result_dir, "결과_요약.pdf")

    # ✅ 한국어 폰트 등록
    pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))  # 한글 명조체
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle('title', parent=styles['Heading1'],
                                 fontName='HYSMyeongJo-Medium', fontSize=18, alignment=1)
    style_sub = ParagraphStyle('sub', parent=styles['Heading2'],
                               fontName='HYSMyeongJo-Medium', fontSize=14, spaceAfter=10)
    style_body = ParagraphStyle('body', parent=styles['Normal'],
                                fontName='HYSMyeongJo-Medium', fontSize=11, leading=15)

    story = []
    story.append(Paragraph("Q-러닝을 이용한 최적 경로 탐색 결과 보고서", style_title))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("1️⃣ 실험 개요", style_sub))
    story.append(Paragraph(
        "본 실험은 강화학습의 대표 알고리즘인 Q-러닝(Q-Learning)을 이용하여 "
        "에이전트가 미로 환경에서 최적의 경로를 스스로 학습하도록 수행하였습니다.",
        style_body))
    story.append(Spacer(1, 0.5*cm))

    for img_name, desc in [
        ("reward_curve.png", "에피소드가 증가함에 따라 보상이 점차 증가하는 양상을 확인할 수 있습니다."),
        ("success_rate.png", "시간이 지날수록 목표 도달 성공률이 높아지며 학습이 안정화되는 과정을 보여줍니다."),
        ("policy_visual.png", "학습된 정책(Policy)은 각 위치에서 최적의 이동 방향을 나타냅니다."),
        ("path_visual.png", "최종적으로 에이전트가 선택한 최적 경로를 시각화한 그림입니다.")
    ]:
        img_path = os.path.join(result_dir, img_name)
        if os.path.exists(img_path):
            story.append(Paragraph(f"📊 {img_name.split('.')[0]}", style_sub))
            story.append(Image(img_path, width=14*cm, height=8*cm))
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(desc, style_body))
            story.append(Spacer(1, 0.5*cm))

    summary_path = os.path.join(result_dir, "summary.txt")
    if os.path.exists(summary_path):
        story.append(Paragraph("📄 학습 요약", style_sub))
        with open(summary_path, encoding="utf-8") as f:
            for line in f:
                if "," in line:
                    k, v = line.strip().split(",", 1)
                    story.append(Paragraph(f"{k.strip()}: {v.strip()}", style_body))
        story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(
        "✅ 결론적으로, Q-러닝은 보상을 극대화하는 방향으로 스스로 경로를 개선하며, "
        "시행착오를 통해 목표를 달성하는 학습 알고리즘임을 실험적으로 확인하였습니다.",
        style_body))

    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)
    print(f"[완료] '{pdf_path}' 생성 완료")

if __name__ == "__main__":
    make_report()
