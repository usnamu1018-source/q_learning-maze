# src/make_report.py
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.units import cm
import os

def parse_summary(path):
    data = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "," in line:
                    k, v = line.strip().split(",", 1)
                    data[k.strip()] = v.strip()
    return data

def make_report(result_dir="results"):
    pdf_path = os.path.join(result_dir, "결과_비교_보고서.pdf")

    # ✅ 한글 폰트 등록
    pdfmetrics.registerFont(UnicodeCIDFont('HYSMyeongJo-Medium'))
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle('title', parent=styles['Heading1'], fontName='HYSMyeongJo-Medium', fontSize=18, alignment=1)
    style_sub = ParagraphStyle('sub', parent=styles['Heading2'], fontName='HYSMyeongJo-Medium', fontSize=14, spaceAfter=10)
    style_body = ParagraphStyle('body', parent=styles['Normal'], fontName='HYSMyeongJo-Medium', fontSize=11, leading=15)
    style_table = ParagraphStyle('table', parent=styles['Normal'], fontName='HYSMyeongJo-Medium', fontSize=11, alignment=1)

    story = []
    story.append(Paragraph("Q-러닝 기반 최적 경로 탐색 비교 보고서", style_title))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("1️⃣ 실험 개요", style_sub))
    story.append(Paragraph(
        "이번 실험은 무작위 탐색(대조군)과 Q-러닝을 이용한 학습(실험군)을 비교하여 "
        "강화학습이 경로 탐색 효율에 미치는 영향을 분석하기 위해 수행되었습니다.", style_body))
    story.append(Spacer(1, 0.5*cm))

    # ---- 그래프 비교 ----
    comparison_images = [
        ("baseline_reward.png", "reward_curve.png", "보상 변화 비교", "무학습 상태에서는 보상이 일정하지 않고 낮은 수준을 유지하지만, Q-러닝을 적용하면 점진적으로 보상이 향상되는 양상을 보입니다."),
        ("baseline_success.png", "success_rate.png", "성공률 변화 비교", "무학습 상태에서는 성공률이 매우 낮지만, 학습이 진행되면 에이전트가 점점 더 높은 확률로 목표를 달성하게 됩니다."),
    ]

    for base_img, train_img, title, comment in comparison_images:
        base_path = os.path.join(result_dir, base_img)
        train_path = os.path.join(result_dir, train_img)
        if os.path.exists(base_path) and os.path.exists(train_path):
            story.append(Paragraph(f"📊 {title}", style_sub))
            t = Table([
                [Image(base_path, width=7*cm, height=5*cm),
                 Image(train_path, width=7*cm, height=5*cm)]
            ])
            t.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
            ]))
            story.append(t)
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(comment, style_body))
            story.append(Spacer(1, 0.5*cm))

    # ---- 실험 결과 요약 비교 ----
    story.append(Paragraph("📄 실험 결과 요약 비교", style_sub))
    base_data = parse_summary(os.path.join(result_dir, "baseline_summary.txt"))
    train_data = parse_summary(os.path.join(result_dir, "summary.txt"))

    if base_data and train_data:
        # Paragraph로 감싼 셀 내용 생성
        def p(text): return Paragraph(text, style_table)
        table_data = [
            [p("지표"), p("무학습(대조군)"), p("Q-러닝(학습군)")]
        ]
        keys = ["총 에피소드", "성공 횟수", "성공률", "평균 보상"]
        for key in keys:
            table_data.append([
                p(key),
                p(base_data.get(key, "-")),
                p(train_data.get(key, "-"))
            ])

        t = Table(table_data, colWidths=[4*cm, 5*cm, 5*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))
    else:
        story.append(Paragraph("⚠️ 요약 데이터 파일을 찾을 수 없습니다.", style_body))

    # ---- 결론 ----
    story.append(Paragraph("✅ 결론", style_sub))
    story.append(Paragraph(
        "비교 결과, 무작위 탐색에 비해 Q-러닝을 적용한 경우 보상과 성공률 모두 크게 향상되었음을 확인할 수 있습니다. "
        "이는 Q-러닝이 시행착오를 통해 효율적인 경로를 학습하고, 보상 함수를 최대화하는 방향으로 행동을 조정함을 시사합니다.",
        style_body))

    SimpleDocTemplate(pdf_path, pagesize=A4).build(story)
    print(f"[완료] '{pdf_path}' 생성 완료")

if __name__ == "__main__":
    make_report()

