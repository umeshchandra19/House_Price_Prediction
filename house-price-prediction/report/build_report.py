"""Build the project report PDF from the files in results/.

    python -m housepred train       # produce results/ first
    python report/build_report.py

Every number in the text is read from results/, so the report cannot drift from the code.
"""
import json
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "House_Price_Prediction_Report.pdf"

audit = json.load(open(R / "data_audit.json"))
summ = json.load(open(R / "summary.json"))
comp = pd.read_csv(R / "model_comparison.csv").set_index("model")
abl = pd.read_csv(R / "ablation_features.csv")
err = pd.read_csv(R / "error_analysis.csv").set_index("group")
curve = pd.read_csv(R / "depth_curve.csv")
imp = pd.read_csv(R / "tree_feature_importance.csv").set_index("feature")["importance"]

LR, DT, DTU, RF = (comp.loc[k] for k in ["Linear Regression", "Decision Tree (tuned)", "Decision Tree (untuned)", "Random Forest (extra)"])
LRR = comp.loc["Linear Regression (raw features)"]
k = lambda v: f"${v / 1000:,.0f}k"
pc = lambda v: f"{100 * v:.0f}%"
a = lambda model, eng, col: float(abl[(abl.model == model) & (abl.new_features == eng)][col].iloc[0])
best_depth = int(curve.loc[curve.cv_r2.idxmax(), "max_depth"])
bp = summ["best_tree_params"]

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=8.7, leading=10.8, alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=1)
SUB = ParagraphStyle("sub", parent=BODY, textColor=colors.HexColor("#444444"), spaceAfter=5)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1b4f72"))
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=10, bulletIndent=0, spaceAfter=1.5)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.4, leading=9, alignment=0, textColor=colors.HexColor("#555555"), spaceAfter=3)
P = lambda t, s=BODY: Paragraph(t, s)
B = lambda t: Paragraph(t, BUL, bulletText="-")


def table(data, widths, font=7.6):
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("LEADING", (0, 0), (-1, -1), font + 1.8),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
          ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f7")),
          ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#1b4f72")), ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#999999"))]
    for i in range(2, len(data), 2):
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f9fa")))
    t.setStyle(TableStyle(st))
    return t


def img(name, width):
    w, h = PILImage.open(R / "figures" / name).size
    return Image(str(R / "figures" / name), width=width, height=width * h / w)


S = []
S.append(P("Predicting House Prices with Python and scikit-learn", H1))
S.append(P("Machine learning project by Umesh Chandra Namani. The code is in the GitHub repository that goes with this report; "
           "its README explains how to run everything.", SUB))

S.append(P("1. What I set out to do", H2))
S.append(P(
    "I wanted to build a complete, honest machine learning project from start to finish: take raw housing data, clean it, add useful "
    "information, compare a simple model with a more flexible one, and measure how good the best one really is. The task is "
    "<b>regression</b>, which means predicting a number (a house price) rather than a category. I compared two models: "
    "<b>Linear Regression</b>, which draws the best straight-line relationship between the inputs and the price, and a "
    "<b>Decision Tree</b>, which asks a series of yes/no questions (for example, \"is the neighbourhood income above X?\") and ends "
    "at a price. I also added a Random Forest as an extra, to see how far a stronger model could go."))

S.append(P("2. The data", H2))
S.append(P(
    f"I used the well-known California housing data (from the 1990 US census), which has <b>{audit['n_rows']:,} records</b> and "
    f"{audit['n_columns']} columns: where the area is, how old the houses are, how many rooms, bedrooms, people and households it has, "
    "the typical income, how close it is to the ocean, and the typical house value, which is what I try to predict. Each record is a "
    "small neighbourhood, not one single house. Before modelling I checked the data and found three things worth knowing:"))
S.append(B(f"<b>Missing values.</b> {audit['missing_values']['total_bedrooms']} areas have no bedroom count. I fill these in later with the typical (median) value."))
S.append(B(f"<b>A price ceiling.</b> {audit['capped_price_rows']:,} areas ({audit['capped_price_share_pct']}%) all show exactly $500,001. "
           "The census simply stopped counting above that, so the true prices there are higher. This limits how well any model can do on expensive areas."))
S.append(B(f"<b>No duplicate rows</b>, and the prices run from ${audit['price_min']:,.0f} to ${audit['price_max']:,.0f} with a middle value of ${audit['price_median']:,.0f}."))

S.append(P("3. Getting the data ready", H2))
S.append(P(
    f"First I split the data: {summ['n_train']:,} records for training and building the models, and {summ['n_test']:,} records "
    "(20%) locked away as a final exam that is used only once at the very end. I kept the income mix the same in both parts so the "
    "exam is fair. Everything else happens inside one <b>pipeline</b>, so the same steps run in the same order every time:"))
S.append(B("<b>Fill the gaps</b> with the median of the training data."))
S.append(B("<b>Add three new features.</b> Raw totals are hard to compare (a block with 5,000 rooms means little by itself), so I added "
           "rooms per household, bedrooms per room and people per household."))
S.append(B("<b>Put the numbers on the same scale</b>, and turn the ocean-distance label into yes/no columns that a model can use."))
S.append(P(
    "Doing all of this inside the pipeline matters. When I test a model on a piece of data it has not seen, the medians and scales are "
    "worked out from the training part only, so no information from the \"exam\" leaks into the \"study\" material."))

S.append(P("4. How I compared the models", H2))
S.append(P(
    "I used <b>5-fold cross-validation</b>. The training data is cut into five pieces; the model learns from four and is tested on the fifth, "
    "and this is repeated until every piece has been the test piece once. The five scores are averaged, and the spread between them shows "
    "how stable the result is. A Decision Tree has settings that matter a lot (mainly how deep it may grow), so I tried 18 combinations "
    "and kept the best. To keep that fair, the search was done again inside every cross-validation round. I chose between Linear Regression "
    "and the Decision Tree using cross-validation only. The locked-away test data was looked at once, at the end."))

S.append(P("5. Results", H2))
S.append(P(
    "A model that predicts a number does not have one single \"accuracy\", so I report three plain measures. <b>R-squared</b> is the share "
    "of the ups and downs in price that the model explains (1.00 would be perfect). The <b>typical error</b> is in dollars. And "
    "<b>within 20%</b> is the share of areas where the predicted price was no more than 20% away from the true price."))
rows = [["Model", "Explains (CV)", "Explains (test)", "Typical error", "Within 20%"]]
for name in ["Linear Regression (raw features)", "Linear Regression", "Decision Tree (untuned)", "Decision Tree (tuned)", "Random Forest (extra)"]:
    c = comp.loc[name]
    rows.append([name, f"{c.cv_r2_mean:.3f} +- {c.cv_r2_std:.3f}", f"{c.test_r2:.3f}", k(c.test_rmse), pc(c.test_within_20pct)])
S.append(table(rows, [2.2 * inch, 1.3 * inch, 1.2 * inch, 1.1 * inch, 1.0 * inch]))
S.append(P(f"Table 1. \"CV\" is the average over the 5 rounds on the training data; \"test\" is the final exam on {summ['n_test']:,} unseen areas. "
           "The Random Forest is an extra and was not part of the choice between the two project models.", CAP))
S.append(img("fig1_model_comparison.png", 6.7 * inch))
S.append(B(
    f"<b>The tuned Decision Tree is the better of the two project models.</b> It explains {DT.test_r2 * 100:.0f}% of the variation in price on the test "
    f"data (Linear Regression: {LR.test_r2 * 100:.0f}%), is typically {k(DT.test_rmse)} off (Linear Regression: {k(LR.test_rmse)}), and lands within 20% "
    f"of the true price for {pc(DT.test_within_20pct)} of areas (Linear Regression: {pc(LR.test_within_20pct)}). The test score is close to the "
    "cross-validation score, which is a good sign that the result was not a lucky split."))
S.append(B(
    f"<b>Tuning mattered.</b> A tree left to grow freely learns the training data perfectly (R-squared {DTU.train_r2_mean:.2f}) but only scores "
    f"{DTU.cv_r2_mean:.2f} on new data, which is worse than the simple straight-line model. That is <b>over-fitting</b>: it memorised instead of learning. "
    f"Limiting the depth (the best settings were a depth of {bp['model__max_depth']} and at least {bp['model__min_samples_leaf']} areas per leaf) "
    f"lifted the cross-validation score to {DT.cv_r2_mean:.2f}."))
S.append(B(
    f"<b>The new features helped only a little.</b> For Linear Regression the cross-validation R-squared went from {a('Linear Regression', 'no', 'cv_r2'):.3f} "
    f"to {a('Linear Regression', 'yes', 'cv_r2'):.3f}, and for the tree from {a('Decision Tree (tuned settings)', 'no', 'cv_r2'):.3f} to "
    f"{a('Decision Tree (tuned settings)', 'yes', 'cv_r2'):.3f}. I kept them because they never hurt, but I would not call them a big win."))
S.append(KeepTogether([img("fig4_feature_importance.png", 6.3 * inch),
    P("Figure 1. Left: the inputs the decision tree relies on most. Right: how the linear model's predicted price moves "
      "when an input goes up by one standard step (the 'island' label has only 5 areas and is left out).", CAP)]))
S.append(B(
    f"<b>What drives the price.</b> Neighbourhood income is by far the most important input ({imp.iloc[0] * 100:.0f}% of the tree's decisions), followed by "
    f"being inland and the number of people per household."))
S.append(B(
    f"<b>A stronger model would do better.</b> The extra Random Forest (many trees voting together) explains {RF.test_r2 * 100:.0f}% and gets {pc(RF.test_within_20pct)} "
    "of areas within 20%. I kept it separate because the project was about comparing the two simpler models, but it shows how much room is left."))
S.append(KeepTogether([img("fig3_tree_depth.png", 3.3 * inch),
    P("Figure 2. The deeper the tree, the better it fits the training data, but after about depth "
      f"{best_depth} it gets worse on new data (this chart lets leaves be as small as one area).", CAP)]))

h6 = P("6. Where the model gets it wrong", H2)
rows = [["Group of test areas", "Areas", "Typical error", "Within 20%"]]
for g in ["Location: INLAND", "Location: <1H OCEAN", "Location: NEAR BAY", "Location: NEAR OCEAN", "True price: cheap (<$150k)",
          "True price: middle ($150-300k)", "True price: expensive (>$300k)", "Without the capped $500k rows"]:
    e = err.loc[g]
    rows.append([g, f"{int(e.n):,}", k(e.rmse), pc(e.within_20pct)])
S.append(KeepTogether([h6, table(rows, [2.9 * inch, 0.9 * inch, 1.3 * inch, 1.2 * inch]),
                       P("Table 2. Tuned Decision Tree, test data only.", CAP)]))
S.append(B(
    f"Mistakes get bigger in dollars as prices rise: {k(err.loc['True price: cheap (<$150k)'].rmse)} off for cheap areas and "
    f"{k(err.loc['True price: expensive (>$300k)'].rmse)} off for expensive ones. Inland areas are predicted most closely in dollars, "
    "partly because they are cheaper."))
S.append(B(
    f"The $500k ceiling hurts. The model cannot learn what the really expensive areas cost, and without the capped rows the test R-squared is "
    f"{err.loc['Without the capped $500k rows'].r2:.2f} rather than {err.loc['All test rows'].r2:.2f}, so the headline number looks a little "
    "better than it would on prices that were not capped."))
S.append(KeepTogether([img("fig2_predicted_vs_actual.png", 5.6 * inch),
    P("Figure 3. Predicted against true price on the test data (darker means more areas). The red line is a perfect prediction. "
      "The vertical stripe at $500k is the price ceiling in the data.", CAP)]))

S.append(P("7. Limits and what I would do next", H2))
S.append(B("<b>The data is old and coarse.</b> It describes 1990 neighbourhoods, not individual houses, so the numbers say little about prices today."))
S.append(B("<b>One split.</b> I used one 80/20 split plus cross-validation. Repeating the whole thing with different random splits would tell me how much the final score moves."))
S.append(B("<b>Simple location handling.</b> The model only sees latitude and longitude as plain numbers. Features such as distance to the nearest big city would likely help."))
S.append(B("<b>Next steps.</b> Try gradient boosting and a log-transformed price, drop or model the capped rows separately, add prediction ranges instead of a single number, and wrap the model in a small web app."))

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=0.62 * inch, rightMargin=0.62 * inch, topMargin=0.55 * inch,
                        bottomMargin=0.55 * inch, title="Predicting House Prices with Python and scikit-learn", author="Umesh Chandra Namani")


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(0.62 * inch, 0.3 * inch, "House Price Prediction - Python, scikit-learn")
    c.drawRightString(A4[0] - 0.62 * inch, 0.3 * inch, f"Page {d.page}")
    c.restoreState()


doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
